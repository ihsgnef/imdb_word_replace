import os
import sys
import time
import glob
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.neighbors import KDTree

from torchtext import data
from torchtext import datasets

from models import ConvModel, LSTMModel

PLH = '<unk>'

def parse_args_common():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='bar')
    parser.add_argument('--fix_embedding', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='results/inject/')
    parser.add_argument('--data_cache', type=str, 
                        default=os.path.join(os.getcwd(), '.data_cache'))
    parser.add_argument('--wv_cache', type=str, 
            default=os.path.join(os.getcwd(), '.vector_cache/wv.pt'))
    parser.add_argument('--wv_type', type=str, default='glove.42B')
    parser.add_argument('--resume_snapshot', type=str, default='')
    return parser

def parse_args_lstm():
    parser = parse_args_common()
    parser.add_argument('--model_class', default='LSTMModel')
    parser.add_argument('--model_name', default='lstm')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    return args

def parse_args_conv():
    parser = parse_args_common()
    parser.add_argument('--model_class', default='ConvModel')
    parser.add_argument('--model_name', default='conv')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--filter_sizes', type=int, nargs='+', default=[3,4,5])
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    return args

def makedirs(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def inject_n(n, dataset):
    '''randomly inject this special symbol to each example in the dataset'''
    for i in range(len(dataset.examples)):
        t = dataset.examples[i].text
        for j in range(n):
            pos = random.randint(0, len(t))
            t = t[:pos] + [PLH] + t[pos:]
        dataset.examples[i].text = t
    return dataset

def setup(args, inject_train=None, inject_dev=None, inject_test=None):
    torch.cuda.set_device(args.gpu)

    ### setup data
    TEXT = data.Field()
    LABEL = data.Field(sequential=False)
    
    train_set, dev_set, test_set = datasets.SST.splits(
            TEXT, LABEL, fine_grained=False, train_subtrees=True,
            filter_pred=lambda x: x.label != 'neutral')

    ### inject special place holders to the datasets
    if inject_train is not None:
        train_set = inject_train(train_set)
    if inject_dev is not None:
        dev_set = inject_dev(dev_set)
    if inject_test is not None:
        test_set = inject_test(test_set)

    TEXT.build_vocab(train_set)
    LABEL.build_vocab(train_set)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train_set, dev_set, test_set), 
            batch_size=args.batch_size, device=args.gpu)

    # load word vectors
    if args.wv_type:
        if os.path.isfile(args.wv_cache):
            TEXT.vocab.vectors = torch.load(args.wv_cache)
        else:
            TEXT.vocab.load_vectors(wv_dir=args.data_cache,
                    wv_type=args.wv_type, wv_dim=args.embed_size)
            makedirs(os.path.dirname(args.wv_cache))
            torch.save(TEXT.vocab.vectors, args.wv_cache)

    args.vocab_size = len(TEXT.vocab)
    args.embed_size = TEXT.vocab.vectors.size(1)
    args.output_size = len(LABEL.vocab)
    print('vocab size', args.vocab_size)
    print('embed size', args.embed_size)
    print('output size', args.output_size)

    ### setup model
    if args.resume_snapshot:
        print('loading snapshot', args.resume_snapshot)
        model = torch.load(args.resume_snapshot, 
                map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = globals()[args.model_class](args)
    
        if args.wv_type:
            model.embed.weight.data = TEXT.vocab.vectors
        
        if args.gpu >= 0:
            model.cuda()

    return args, TEXT, LABEL, train_iter, dev_iter, test_iter, model

def train(args):
    inject_func = partial(inject_n, 1)
    args, TEXT, LABEL, train_iter, dev_iter, test_iter, model = setup(
            args, inject_train=inject_func, inject_dev=inject_func,
            inject_test=inject_func)

    ### training setup
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda w: w.requires_grad, model.parameters())
    optim = torch.optim.Adam(parameters, lr=args.lr)
    train_iter.repeat = False
    makedirs(args.save_path)
    iterations = 0
    best_dev_acc = 0

    for epoch in range(args.epochs):
        train_iter.init_epoch()
        train_loss = 0
        n_correct = n_total = 0
        report_string = ''
        for batch_idx, batch in enumerate(tqdm(train_iter, leave=False)):
            model.train()
            optim.zero_grad()
            y = model(batch.text)

            # update model
            loss = criterion(y, batch.label)
            loss.backward()
            optim.step()

            # update stats
            train_loss += loss.data[0]
            predictions = torch.max(y, 1)[1].view(batch.label.size())
            n_correct += (predictions.data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100 * n_correct / n_total

            iterations += 1
            
            # update snapshot
            if iterations % args.save_every == 0:
                snapshot_prefix = os.path.join(
                        args.save_path, args.model_name + '_snapshot')
                snapshot_path = snapshot_prefix + \
                        '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(
                                train_acc, loss.data[0], iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            # evaluation
            if iterations % args.dev_every == 0:
                model.eval()
                dev_iter.init_epoch()
                n_dev_correct = n_dev_total = 0
                dev_loss = 0
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    y = model(dev_batch.text)
                    predictions = torch.max(y, 1)[1].view(dev_batch.label.size())
                    n_dev_correct += (predictions.data == dev_batch.label.data).sum()
                    n_dev_total += dev_batch.batch_size
                    dev_loss += criterion(y, dev_batch.label).data[0]
                dev_acc = 100 * n_dev_correct / n_dev_total

                if report_string:
                    report_string += '\n'
                report_string += '{0} {1} {2} {3}'.format(
                        epoch, iterations, dev_loss / n_dev_total, dev_acc)

                # update best eval accuracy
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    snapshot_prefix = os.path.join(
                            args.save_path, args.model_name + '_best_snapshot')
                    snapshot_path = snapshot_prefix + \
                        '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(
                                dev_acc, dev_loss / n_dev_total, iterations)
                    torch.save(model, snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)

            # elif iterations % args.log_every == 0:
            #     print(epoch, iterations, i, len(train_iter), train_loss /
            #             n_total, train_acc)
        if report_string:
            print(report_string)

def evaluate(args):
    snapshot_prefix = os.path.join(
            args.save_path, args.model_name + '_best_snapshot')
    args.resume_snapshot = glob.glob(snapshot_prefix + '*')[0]
    inject_one = partial(inject_n, 1)
    inject_more = partial(inject_n, 3)
    args, TEXT, LABEL, train_iter, dev_iter, test_iter, model = setup(
            args, inject_train=inject_one, inject_dev=inject_more,
            inject_test=inject_more)

    def eval(iterator):
        model.eval()
        iterator.init_epoch()
        n_dev_correct = n_dev_total = 0
        dev_loss = 0
        criterion = nn.CrossEntropyLoss()
        for dev_batch_idx, dev_batch in enumerate(iterator):
            y = model(dev_batch.text)
            predictions = torch.max(y, 1)[1].view(dev_batch.label.size())
            n_dev_correct += (predictions.data == dev_batch.label.data).sum()
            n_dev_total += dev_batch.batch_size
            dev_loss += criterion(y, dev_batch.label).data[0]
        dev_acc = 100 * n_dev_correct / n_dev_total
        return dev_loss / n_dev_total, dev_acc

    dev_loss, dev_acc = eval(dev_iter)
    print('dev {0} {1}'.format(dev_loss, dev_acc))

    test_loss, test_acc = eval(test_iter)
    print('test {0} {1}'.format(test_loss, test_acc))

def clean_sentence(sent):
    if isinstance(sent, str):
        sent = sent.split(' ')
    useless_words = ['<pad>', '.', ',']
    sent = [x for x in sent if x not in useless_words]
    sent = ' '.join(sent).lower()
    return sent

def inject_and_replace(args):

    def to_sentence(words):
        words = to_numpy(words)
        words = [TEXT.vocab.itos[w] for w in words]
        words = [w for w in words if w != '<pad>']
        return ' '.join(words)

    def to_numpy(x):
        if isinstance(x, Variable):
            x = x.data
        try:
            if x.is_cuda:
                x = x.cpu()
        except AttributeError:
            pass
        if isinstance(x, torch.LongTensor) or isinstance(x, torch.FloatTensor):
            x = x.numpy()
        return x

    args.n_inject = 1
    args.eps = 5
    args.norm = np.inf
    snapshot_prefix = os.path.join(
            args.save_path, args.model_name + '_best_snapshot')
    args.resume_snapshot = glob.glob(snapshot_prefix + '*')[0]
    inject_func = partial(inject_n, args.n_inject)
    args, TEXT, LABEL, train_iter, dev_iter, test_iter, model = setup(
            args, inject_dev=inject_func, inject_test=inject_func)
    plh_idx = TEXT.vocab.stoi[PLH]
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    iterator = dev_iter
    iterator.init_epoch()
    iterator.train = True

    tree = KDTree(TEXT.vocab.vectors.numpy())
    print('KDTree built for {} words'.format(len(TEXT.vocab)))

    extracted_grads = {}
    def extract_grad_hook(name):
        def hook(grad):
            extracted_grads[name] = grad
        return hook

    checkpoint = []
    n_correct = n_total = n_words = n_correct_injected = 0
    for batch_idx, batch in enumerate(iterator):
        if batch_idx > 0:
            break
        preds = []
        y = model(batch.text, extract_grad_hook('inputs_embed'))
        preds.append(torch.max(y, 1)[1].view(batch.label.size()))
        n_correct += (preds[0].data == batch.label.data).sum()
        n_total += batch.batch_size
        n_words += batch.text.size(0) * batch.batch_size
    
        loss = criterion(y, batch.label)
        model.zero_grad()
        loss.backward()

        grads = extracted_grads['inputs_embed'].transpose(0, 1)
        grads = grads.data.cpu().numpy()
        text = batch.text.transpose(0, 1).data.cpu().numpy()
        y = y.data.cpu().numpy()

        batch_text = batch.text.data.clone().transpose(0, 1)
        for i in range(batch.batch_size):
            text_i_list = text[i].tolist()
            j = random.randint(0, len(text_i_list) - 1)
            old_embed = TEXT.vocab.vectors[text[i][j]].numpy()
            word_grad = grads[i][j]
            word_grad /= np.linalg.norm(word_grad, ord=args.norm)

            # ordered by norm of gradient and perturbed by gradient
            new_embed_grd = old_embed + word_grad * args.eps
            _, inds = tree.query([new_embed_grd], k=2)
            repl = inds[0][0] if inds[0][0] != text[i][j] else inds[0][1]

            k = text_i_list.index(plh_idx)
            batch_text[i][k] = repl.item()

            print(to_sentence(text[i]))
            print(to_sentence(batch_text[i]))

        y = model(Variable(batch_text).transpose(0, 1))
        preds.append(torch.max(y, 1)[1].view(batch.label.size()))
        n_correct_injected += (preds[1].data == batch.label.data).sum()

    print(n_correct, n_total, n_correct_injected)

if __name__ == '__main__':
    args = parse_args_lstm()
    globals()[args.mode](args)
