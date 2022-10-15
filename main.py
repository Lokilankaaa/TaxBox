import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from datasets_torch.handcrafted import Handcrafted
from datasets_torch.nodeset import NodeSet
import torch
from model.visgnn import GCN
from model.node_encoder import NodeEncoder
import configparser
from utils.loss import triplet_loss, contrastive_loss
import logging
from utils.graph_operate import test_on_insert
from utils.visualize_graph import *
from utils.utils import sample_path, sample_pair, checkpoint, sample_triples
from tensorboardX import SummaryWriter
import clip
import random
import numpy as np


def get_dataset(root, dataset):
    return Handcrafted(root)


def prepare(model_name='', lr=0.001, step_size=5, gamma=0.1, parallel=True):
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    # model = GCN([1024, 512, 512, 1024], 3).to(device)
    model = NodeEncoder(128)
    prep = model.preprocess
    model.to(device)
    optimizer = torch.optim.Adam(params=[
        {'params': model.clip.parameters(), 'lr': 0.1 * lr},
        # {'params': model.project_text.parameters(), 'lr': lr},
        # {'params': model.project_image.parameters(), 'lr': lr},
        {'params': model.project_box.parameters(), 'lr': lr},
        {'params': model.fusion_module.parameters(), 'lr': lr}
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if parallel:
        model = torch.nn.parallel.DataParallel(model)
    return model, prep, optimizer, scheduler, device


def train(model, dataset, optimizer, device):
    writer = SummaryWriter(comment='NodeEncoder')
    model.train()
    sample_nums = 2
    total_iters = 0
    for e in range(20):
        pe, pp, pn = 0, 0, 0
        for se in tqdm.tqdm(range(1200 // sample_nums)):
            # paths = sample_path(dataset.raw_graph, 4)
            # pos = [sample_pair(random.choice(paths)) for _ in range(sample_nums)]
            # neg = [sample_pair(tuple(random.sample(paths, k=2))) for _ in range(sample_nums)]
            # pos = np.array(pos)
            # neg = np.array(neg)
            triples = sample_triples(dataset.raw_graph, sample_nums, n_num=8)
            triples = np.array(triples)  # sample_nums x (2+n_num)
            # batch = np.hstack([pos[:, 0], pos[:, 1], neg[:, 0], neg[:, 1]])
            batch = np.hstack(triples)

            optimizer.zero_grad()
            text, img = [], []

            for i in batch:
                inputs = dataset[i]
                _id, _name, _text, _imgs = inputs
                text = _text if len(text) == 0 else torch.cat([text, _text])
                img = _imgs if len(img) == 0 else torch.cat([img, _imgs])
            text = text.to(device)
            img = img.to(device)
            outs = model(text, img)

            loss = contrastive_loss(outs, sample_nums)
            # pe += loss.item()
            # pp += p
            # pn += n

            # loss = contrastive_loss(outs)
            pe += loss.item()
            total_iters += 1
            if total_iters % 10 == 0:
                print(e, total_iters, pe / 10)
                # writer.add_scalar('p loss', pp / 10, total_iters)
                # writer.add_scalar('n loss', pn / 10, total_iters)
                writer.add_scalar('total loss', pe / 10, total_iters)
                # print(e, pe / 10)
                # pe, pp, pn = 0, 0, 0
                pe = 0
            loss.backward()
            optimizer.step()
        __test(dataset, model)
        checkpoint('model.pth', model)
    writer.close()
    # test_on_insert('/data/home10b/xw/visualCon/datasets_json/saved_handcrafted_test.json', data.raw_graph, model(data),
    #                model)


def __test(dataset, model):
    test_on_insert(dataset, model, 'datasets_json/saved_handcrafted_test.json')


def main():
    # dataset = get_dataset('/data/home10b/xw/visualCon/datasets_json/handcrafted')
    model, prep, optimizer, scheduler, device = prepare(parallel=True)
    dataset = NodeSet('/data/home10b/xw/visualCon/datasets_json/',
                      '/data/home10b/xw/visualCon/handcrafted',
                      prep, clip.tokenize)
    vis_graph(get_adj_matrix(dataset.id_to_children), dataset.id_to_name)
    # train(model, dataset, optimizer, device)
    # model.load_state_dict(torch.load('model.pth'))
    # __test(dataset, model)


if __name__ == '__main__':
    main()
