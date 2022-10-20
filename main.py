import math

import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datasets_torch.handcrafted import Handcrafted
from datasets_torch.nodeset import NodeSet
import torch
from model.visgnn import GCN
from model.node_encoder import NodeEncoder, twinTransformer
import configparser
from utils.loss import triplet_loss, contrastive_loss
import logging
from utils.graph_operate import test_on_insert
from utils.visualize_graph import *
from utils.utils import sample_path, sample_pair, checkpoint, sample_triples, sample_n_nodes, adjust_moco_momentum
from tensorboardX import SummaryWriter
import clip
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--box_dim", type=int, default=128)
parser.add_argument("--regularization_loss", type=bool, default=True)
parser.add_argument("--gpu_id", type=str, default='0,1')
parser.add_argument("--saved_model_path", type=str, default='model.pth')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_imgs_per_node", type=int, default=100)
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--vis_graph", action="store_true")
parser.add_argument("--parallel", action="store_true")

def get_dataset(root, dataset):
    return Handcrafted(root)


def prepare(args, step_size=600, gamma=0.2, parallel=True):
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    # model = GCN([1024, 512, 512, 1024], 3).to(device)
    model = twinTransformer(args.box_dim, args.max_imgs_per_node + 1)
    # prep = model.preprocess
    model.to(device)
    optimizer = torch.optim.Adam(params=model.query_transformer.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if parallel:
        model = torch.nn.parallel.DataParallel(model)
    return model, optimizer, scheduler, device


def train(model, dataset, optimizer, scheduler, device, args):
    writer = SummaryWriter(comment='NodeEncoder')
    model.train()
    sample_nums = 50
    total_iters = 0
    for e in range(20):
        pe, pp, pn = 0, 0, 0
        for se in tqdm.tqdm(range(20000 // sample_nums)):
            # paths = sample_path(dataset.raw_graph, 4)
            # pos = [sample_pair(random.choice(paths)) for _ in range(sample_nums)]
            # neg = [sample_pair(tuple(random.sample(paths, k=2))) for _ in range(sample_nums)]
            # pos = np.array(pos)
            # neg = np.array(neg)
            # triples = sample_triples(dataset.raw_graph, sample_nums, n_num=8)
            # triples = np.array(triples)  # sample_nums x (2+n_num)
            # batch = np.hstack([pos[:, 0], pos[:, 1], neg[:, 0], neg[:, 1]])
            # batch = np.hstack(triples)
            batch, con_m = sample_n_nodes(sample_nums, dataset.id_to_father)

            optimizer.zero_grad()
            text, img = [], []

            for i in batch:
                inputs = dataset[i]
                _id, _name, _text, _imgs = inputs
                text = _text if len(text) == 0 else torch.cat([text, _text])
                img = _imgs if len(img) == 0 else torch.cat([img, _imgs])
            text = text.to(device)
            img = img.to(device)
            q, k = model(text, img, adjust_moco_momentum(e, 0.99, 20))

            loss, pos_n = contrastive_loss(q, k, con_m, batch, args.regularization_loss)
            # pe += loss.item()
            # pp += p
            # pn += n

            # loss = contrastive_loss(outs)
            pe += loss.item()
            pp += pos_n
            total_iters += 1
            if total_iters % 100 == 0:
                print(q[random.randint(0, q.shape[0] - 1)])
            if total_iters % 10 == 0:
                print(e, total_iters, pe / 10)
                # writer.add_scalar('pos num', pp / 10, total_iters)
                # writer.add_scalar('n loss', pn / 10, total_iters)
                writer.add_scalar('total loss', pe / 10, total_iters)
                print(e, pp / 10)
                # pe, pp, pn = 0, 0, 0
                pe = 0
                pp = 0
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
            else:
                return
            scheduler.step()
        # __test(dataset, model)
        checkpoint(args.saved_model_path, model)
    writer.close()


def __test(dataset, model):
    test_on_insert(dataset, model, 'datasets_json/handcrafted_test.json', args.box_dim)


def main(args):
    # dataset = get_dataset('/data/home10b/xw/visualCon/datasets_json/handcrafted')
    model, optimizer, scheduler, device = prepare(args, parallel=args.parallel)
    dataset = NodeSet('/data/home10b/xw/visualCon/datasets_json/',
                      '/data/home10b/xw/visualCon/handcrafted', max_imgs_per_node=args.max_imgs_per_node)
    if args.vis_graph:
        vis_graph(get_adj_matrix(dataset.id_to_children), dataset.id_to_name)
    if args.train:
        train(model, dataset, optimizer, scheduler, device, args)
    if args.test:
        model.load_state_dict(torch.load(args.saved_model_path))
        __test(dataset, model)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    main(args)
