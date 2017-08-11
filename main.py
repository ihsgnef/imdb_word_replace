import os
import sys
import time
import glob
import argparse
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.neighbors import KDTree

from torchtext import data
from torchtext import datasets


extracted_grads = {}
def extract_grad_hook(name):
    def hook(grad):
        extracted_grads[name] = grad
    return hook

class ConvModel(nn.Module):
    
    def __init__(self, cfg):
        super(ConvModel, self).__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        if cfg.fix_embedding:
            self.embed.weight.requires_grad = False
        convs = []
        for i, filter_size in enumerate(cfg.filter_sizes):
            pad = filter_size // 2
            conv = nn.Sequential(
                    nn.Conv1d(cfg.embed_size, cfg.hidden_size, filter_size, padding=pad),
                    nn.ReLU()
                    )
            convs.append(conv)
        self.convs = nn.ModuleList(convs)
        self.fconns = nn.Sequential(
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size * len(cfg.filter_sizes), cfg.output_size)
                )

    def forward(self, inputs, extract_inputs_embed_grad=False):
        inputs_embed = self.embed(inputs) 
        if extract_inputs_embed_grad:
            inputs_embed.register_hook(extract_grad_hook('inputs_embed'))
        # length, batch_size, embed_size -> batch_size, embed_dim, length
        inputs_embed = inputs_embed.transpose(0, 1).transpose(1, 2)
        mots = []
        for conv in self.convs:
            conv_out = conv(inputs_embed)
            mot, _ = conv_out.max(2)
            mots.append(mot.squeeze(2))
        mots = torch.cat(mots, 1)
        output = self.fconns(mots)
        return output
                    

class LSTMModel(nn.Module):

    def __init__(self, cfg):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        if cfg.fix_embedding:
            self.embed.weight.requires_grad = False
        self.lstm = nn.LSTM(cfg.embed_size, cfg.hidden_size, cfg.num_layers)
        self.fconns = nn.Sequential( 
                nn.BatchNorm1d(cfg.hidden_size),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size, cfg.output_size)
                )

    def forward(self, inputs, extract_inputs_embed_grad=False):
        length, batch_size = inputs.size()
        hidden = self.init_hidden(batch_size)
        inputs_embed = self.embed(inputs)
        if extract_inputs_embed_grad:
            inputs_embed.register_hook(extract_grad_hook('inputs_embed'))
        output, hidden = self.lstm(inputs_embed, hidden)
        output = output[-1]
        output = self.fconns(output)
        return output
        
    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.embed.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        w = next(self.parameters()).data
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        return (Variable(w.new(num_layers, batch_size, hidden_size).zero_()),
                Variable(w.new(num_layers, batch_size, hidden_size).zero_()))


def makedirs(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
def parse_args_common():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_embedding', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='results')
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

parse_args = parse_args_lstm

def setup(args):
    torch.cuda.set_device(args.gpu)

    ### setup data
    TEXT = data.Field()
    LABEL = data.Field(sequential=False)
    
    train_set, dev_set, test_set = datasets.SST.splits(
            TEXT, LABEL, fine_grained=True, train_subtrees=True,
            filter_pred=lambda x: x.label != 'neutral')
    
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

def train():
    args = parse_args()
    args, TEXT, LABEL, train_iter, dev_iter, test_iter, model = setup(args)

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

def test():
    args = parse_args()
    snapshot_prefix = os.path.join(
            args.save_path, args.model_name + '_best_snapshot')
    args.resume_snapshot = glob.glob(snapshot_prefix + '*')[0]
    args, TEXT, LABEL, train_iter, dev_iter, test_iter, model = setup(args)

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

def foo():
    args = parse_args()
    snapshot_prefix = os.path.join(
            args.save_path, args.model_name + '_best_snapshot')
    args.resume_snapshot = glob.glob(snapshot_prefix + '*')[0]
    args, TEXT, LABEL, train_iter, dev_iter, test_iter, model = setup(args)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    dev_iter.init_epoch()
    dev_iter.train = True
    
    out = open('output.txt', 'w')
    pkl = []

    for batch_idx, batch in enumerate(tqdm(dev_iter)):
        y = model(batch.text, extract_inputs_embed_grad=True)
        loss = criterion(y, batch.label)
        model.zero_grad()
        loss.backward()
        grad = extracted_grads['inputs_embed'].sum(dim=2).squeeze(2)
        grad = grad.transpose(0, 1).data.cpu().numpy()
        text = batch.text.transpose(0, 1)
        y = y.data.cpu().numpy()
        prediction = np.argmax(y, axis=1)
        label = batch.label.data.cpu().numpy()
        
        for i in range(batch.batch_size):
            scores = np.abs(grad[i])
            order = np.argsort(scores)[::-1]
            words = text[i].data.cpu().numpy()
            words = [TEXT.vocab.itos[x] for x in words]
            sorted_sent = [words[j] for j in order]
            sent = [x for x in words]
            pkl.append([])
            pkl[-1].append(clean_sentence(sent))
            pkl[-1].append(clean_sentence(sorted_sent))
            pkl[-1].append(LABEL.vocab.itos[label[i]])
            pkl[-1].append(LABEL.vocab.itos[prediction[i]])
            out.write('\n'.join(pkl[-1]) + '\n\n')
            pkl[-1].append(order)
    with open('output.pkl', 'wb') as f:
        pickle.dump(pkl, f)

def bar():
    args = parse_args()
    args.n_replace = 1
    args.eps = 50000
    snapshot_prefix = os.path.join(
            args.save_path, args.model_name + '_best_snapshot')
    args.resume_snapshot = glob.glob(snapshot_prefix + '*')[0]
    args, TEXT, LABEL, train_iter, dev_iter, test_iter, model = setup(args)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    dev_iter.init_epoch()
    dev_iter.train = True

    tree = KDTree(TEXT.vocab.vectors.numpy())
    print('KDTree built for {} words'.format(len(TEXT.vocab)))

    # use better exclude list
    exclude_list = list(range(20))

    n_correct = n_total = n_new_correct = 0
    for batch_idx, batch in enumerate(dev_iter):
        y = model(batch.text, extract_inputs_embed_grad=True)
        predictions = torch.max(y, 1)[1].view(batch.label.size())
        n_correct += (predictions.data == batch.label.data).sum()
        n_total += batch.batch_size

        loss = criterion(y, batch.label)
        model.zero_grad()
        loss.backward()

        grads = extracted_grads['inputs_embed'].transpose(0, 1)
        grads = grads.data.cpu()
        scores = grads.sum(dim=2).squeeze(2).numpy()
        grads = grads.numpy()
        text = batch.text.transpose(0, 1).data.cpu().numpy()
        y = y.data.cpu().numpy()

        batch_text = batch.text.data
        for i in range(batch.batch_size):
            order = np.argsort(np.abs(scores[i]))[::-1]
            excluded_idxs = [j for j, x in enumerate(text[i]) if x in exclude_list]
            order = [j for j in order if j not in excluded_idxs][:args.n_replace]

            for j in order:
                old_embedding = TEXT.vocab.vectors[text[i][j]].numpy()
                word_grad = grads[i][j]
                new_embedding = old_embedding + word_grad * args.eps
                dists, inds = tree.query(new_embedding, k=2)
                inds = inds[0]
                replace = inds[0] if inds[0] != text[i][j] else inds[1]
                batch_text[j][i] = replace.item()

        batch_text = Variable(batch_text)
        y = model(batch_text, extract_inputs_embed_grad=True)
        predictions = torch.max(y, 1)[1].view(batch.label.size())
        n_new_correct += (predictions.data == batch.label.data).sum()
        print(n_correct / n_total, n_new_correct / n_total)
        
                
if __name__ == '__main__':
    bar()
