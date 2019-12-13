import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(0)

class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


class UnsupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc):
        super(UnsupervisedGraphSage, self).__init__()
        self.enc = enc
        # self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, num_neg=5):
        scores = self.enc(nodes)
        pos_sample = self.enc.pos_sample(nodes)
        pos_embed = self.enc(pos_sample)

        neg_sample = self.enc.neg_sample(nodes, num_neg)
        neg_embed = [self.enc(neg_sample[i]) for i in range(num_neg)]

        loss_pos = torch.diag(-torch.log(torch.sigmoid(scores.t().mm(pos_embed))))

        loss_neg = sum([torch.diag(-torch.log(torch.sigmoid(-scores.t().mm(neg_embed[i])))) for i in range(num_neg)]) / len(nodes)
        # print(loss_pos.shape, loss_pos.mean().item(), loss_neg.shape, loss_neg.mean().item())
        loss_batch = torch.mean(loss_pos + loss_neg)
        return loss_batch


class UnsupervisedGraphSageCls(nn.Module):
    def __init__(self, num_classes, enc):
        super(UnsupervisedGraphSageCls, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            #feat_data[i,:] = map(float, info[1:-1])
            feat_data[i, :] = [float(x) for x in info[1:-1]]

            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("../cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def un_cora(model_d=128, cuda=False):
    np.random.seed(233)
    random.seed(233)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data),
                                   requires_grad=False)
    if cuda:
        features.cuda()

    agg1 = MeanAggregator(features, cuda=cuda)
    enc1 = Encoder(features, 1433, model_d, adj_lists, agg1, gcn=False, cuda=cuda)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=cuda)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(),
                   enc1.embed_dim,
                   128,
                   adj_lists,
                   agg2,
                   base_model=enc1,
                   gcn=False,
                   cuda=cuda)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = UnsupervisedGraphSage(7, enc2)
    if cuda:
        graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    # test = rand_indices[:1000]
    test_size = 1000
    train_size = num_nodes
    val = rand_indices[:test_size]
    train = list(rand_indices[test_size:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       graphsage.parameters()),
                                lr=0.2)
    times = []

    epoch_num = 10
    batch_size = 512
    for batch in range(epoch_num * int(train_size / batch_size)):
        # batch_size

        batch_nodes = train[:batch_size]
        # shuffle：random list on itself return none
        random.shuffle(train)
            
        start_time = time.time()
        # set all gradient to be 0
        optimizer.zero_grad()
        # loss 包含 forward
        loss = graphsage.loss(
            batch_nodes)

        loss.backward()
        # refresh params' gradient
        optimizer.step()

        end_time = time.time()
        times.append(end_time - start_time)
        print('[%f / %d]' % (float(batch / int(train_size / batch_size)), epoch_num), 'batch:', batch, 'loss:', loss.item())

    # print('############ Training Classification ##########')
    # cls_model = UnsupervisedGraphSageCls(7, enc2)
    # cls_optimizer = torch.optim.SGD([cls_model.weight], lr=0.1)
    # for batch in range(epoch_num * batch_size):
    #     # batch_size
    #     batch_nodes = train[:batch_size]
    #     # shuffle：random list on itself return none
    #     random.shuffle(train)
    #     start_time = time.time()
    #     # set all gradient to be 0
    #     cls_optimizer.zero_grad()
    #     # loss 包含 forward
    #     loss = cls_model.loss(
    #         batch_nodes,
    #         Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
    #
    #     loss.backward()
    #     # refresh params' gradient
    #     cls_optimizer.step()
    #
    #     end_time = time.time()
    #     times.append(end_time - start_time)
    #     # print(batch, loss.item())
    #
    # val_output = cls_model.forward(val)
    # print(
    #     "Validation F1:",
    #     f1_score(labels[val],
    #              val_output.data.numpy().argmax(axis=1),
    #              average="micro"))
    # print("Average batch time:", np.mean(times))
    # 1
    # return f1_score(labels[val],
    #              val_output.data.numpy().argmax(axis=1),
    #              average="micro")


def run_cora():
    np.random.seed(233)
    random.seed(233)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data),
                                   requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(),
                   enc1.embed_dim,
                   128,
                   adj_lists,
                   agg2,
                   base_model=enc1,
                   gcn=False,
                   cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
    ungraphsage = UnsupervisedGraphSage(7, enc2)
    #    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    # test = rand_indices[:1000]
    val = rand_indices[:1000]
    train = list(rand_indices[1000:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       graphsage.parameters()),
                                lr=0.7)
    times = []
    for batch in range(200):
        # batch_size
        batch_nodes = train[:256]
        # shuffle：random list on itself return none
        random.shuffle(train)
        start_time = time.time()
        # set all gradient to be 0
        optimizer.zero_grad()
        # loss 包含 forward
        loss = graphsage.loss(
            batch_nodes,
            Variable(torch.LongTensor(labels[np.array(batch_nodes)])))

        if batch % 20 == 0:
            loss2 = ungraphsage.loss(batch_nodes)
            print('[{}/{}] Training: cls loss = {}; uns loss = {}'.format(batch, 200, loss, loss2))

        loss.backward()
        # refresh params' gradient
        optimizer.step()

        end_time = time.time()
        times.append(end_time - start_time)
        # print(batch, loss.item())

    val_output = graphsage.forward(val)
    print(
        "Validation F1:",
        f1_score(labels[val],
                 val_output.data.numpy().argmax(axis=1),
                 average="micro"))
    print("Average batch time:", np.mean(times))
    1
    return f1_score(labels[val],
                 val_output.data.numpy().argmax(axis=1),
                 average="micro")


def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("../pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {
            entry.split(":")[1]: i - 1
            for i, entry in enumerate(fp.readline().split("\t"))
        }
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("../pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_pubmed():
    np.random.seed(233)
    random.seed(233)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data),
                                   requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(),
                   enc1.embed_dim,
                   128,
                   adj_lists,
                   agg2,
                   base_model=enc1,
                   gcn=True,
                   cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
    #    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    # test = rand_indices[:1000]
    val = rand_indices[:4000]
    train = list(rand_indices[4000:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       graphsage.parameters()),
                                lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(
            batch_nodes,
            Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        # print(batch, loss.item())

    val_output = graphsage.forward(val)
    # print(
    #     "Validation F1:",
    #     f1_score(labels[val],
    #              val_output.data.numpy().argmax(axis=1),
    #              average="micro"))
    # print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    un_cora()
    # run_cora()
    # f, a = 0, 0
    # for i in range(10):
    #     ret = run_cora()
    #     # f = max(f, ret[0])
    #     # a = max(a, ret[1])
    #     print(i, ret)
    # print(f, a)
