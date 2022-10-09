import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from datasets_torch.handcrafted import Handcrafted
from datasets_torch.nodeset import NodeSet
import torch
from model.visgnn import GCN
from model.node_encoder import NodeEncoder
import configparser
from utils.loss import contrastive_loss
import logging
from utils.graph_operate import test_on_insert
from utils.utils import sample_path, sample_pair, checkpoint
from tensorboardX import SummaryWriter
import clip
import random
import numpy as np


def get_dataset(root, dataset):
    return Handcrafted(root)


def prepare(model_name='', lr=0.001, step_size=5, gamma=0.1, parallel=False):
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    # model = GCN([1024, 512, 512, 1024], 3).to(device)
    model = NodeEncoder(256)
    prep = model.preprocess
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if parallel:
        model = torch.nn.parallel.DataParallel(model)
    return model, prep, optimizer, scheduler, device


def train(model, dataset, optimizer, device):
    writer = SummaryWriter(comment='NodeEncoder')
    model.train()
    sample_nums = 3
    total_iters = 0
    for e in range(40):
        pe, pp, pn = 0, 0, 0
        for se in tqdm.tqdm(range(300 // sample_nums)):
            paths = sample_path(dataset.raw_graph, 4)
            pos = [sample_pair(random.choice(paths)) for _ in range(sample_nums)]
            neg = [sample_pair(tuple(random.sample(paths, k=2))) for _ in range(sample_nums)]
            pos = np.array(pos)
            neg = np.array(neg)
            batch = np.hstack([pos[:, 0], pos[:, 1], neg[:, 0], neg[:, 1]])

            optimizer.zero_grad()
            outs = []
            for i in batch:
                inputs = dataset[i]
                _id, name, text, imgs = inputs
                text = text.to(device)
                imgs = imgs.to(device)
                outs.append(model(text, imgs))

            loss, p, n = contrastive_loss(outs)
            total_iters += 1

            pe += loss.item()
            pp += p
            pn += n
            if total_iters % 10 == 0:
                print(e, total_iters, pe / 10, pp / 10, pn / 10)
                writer.add_scalar('p loss', pp / 10, total_iters)
                writer.add_scalar('n loss', pn / 10, total_iters)
                writer.add_scalar('total loss', pe / 10, total_iters)
                pe, pp, pn = 0, 0, 0
            loss.backward()
            optimizer.step()
        checkpoint('model.pth', model)
    writer.close()
    # test_on_insert('/data/home10b/xw/visualCon/datasets_json/saved_handcrafted_test.json', data.raw_graph, model(data),
    #                model)


def main():
    # dataset = get_dataset('/data/home10b/xw/visualCon/datasets_json/handcrafted')
    model, prep, optimizer, scheduler, device = prepare(parallel=True)
    dataset = NodeSet('/data/home10b/xw/visualCon/datasets_json/',
                      '/data/home10b/xw/visualCon/handcrafted',
                      prep, clip.tokenize)

    train(model, dataset, optimizer, device)


if __name__ == '__main__':
    main()
