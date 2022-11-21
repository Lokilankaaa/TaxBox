import math
import pprint

import tqdm
import os

from utils.mkdataset import split_tree_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# from datasets_torch.handcrafted import Handcrafted
from datasets_torch.nodeset import NodeSet
from datasets_torch.treeset import TreeSet
import torch
from model.node_encoder import NodeEncoder, twinTransformer, MultimodalNodeEncoder, TreeKiller
import configparser
from utils.loss import triplet_loss, contrastive_loss, adaptive_BCE, cls_loss
import logging
from utils.graph_operate import test_on_insert, test_entailment
from utils.visualize_graph import *
from utils.metric import TreeMetric
from utils.utils import sample_path, sample_pair, checkpoint, sample_triples, sample_n_nodes, adjust_moco_momentum
from tensorboardX import SummaryWriter
from utils.scheduler import get_scheduler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--box_dim", type=int, default=128)
parser.add_argument("--regularization_loss", type=bool, default=True)
parser.add_argument("--gpu_id", type=str, default='0,1')
parser.add_argument("--saved_model_path", type=str, default='model.pth')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--max_imgs_per_node", type=int, default=100)
parser.add_argument("--sample_nums", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--vis_graph", action="store_true")
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--load_dataset_pt", type=str, default='imagenet_dataset.pt')


# def get_dataset(root, dataset):
#     return Handcrafted(root)


def prepare(args, parallel=True):
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    # model = GCN([1024, 512, 512, 1024], 3).to(device)
    # model = twinTransformer(args.box_dim, args.max_imgs_per_node + 1)
    model = TreeKiller(args.box_dim, 512)
    # prep = model.preprocess
    model.to(device)
    optimizer = torch.optim.Adam(params=[
        {'params': model.node_encoder.parameters(), 'lr': args.lr},
        {'params': model.struct_encoder.parameters(), 'lr': args.lr},
        {'params': model.box_decoder.parameters(), 'lr': args.lr},
        {'params': model.scorer.parameters(), 'lr': args.lr}
    ], lr=args.lr)
    total_iters = 206 * args.epoch
    scheduler = get_scheduler(optimizer, 'warmupcosine', 0.06 * total_iters, total_iters)
    if parallel:
        model.node_encoder = torch.nn.parallel.DataParallel(model.node_encoder)
        model.struct_encoder.fusion = torch.nn.parallel.DataParallel(model.struct_encoder.fusion)
        model.box_decoder = torch.nn.parallel.DataParallel(model.box_decoder)
        model.scorer = torch.nn.parallel.DataParallel(model.scorer)
    return model, optimizer, scheduler, device


def train(model, dataset, optimizer, scheduler, device, args):
    writer = SummaryWriter(comment='NodeEncoder')
    metric = TreeMetric()
    model.train()
    # sample_nums = args.sample_nums
    total_iters = 0
    bs = []
    for e in range(args.epoch):
        pe = 0
        dataset.shuffle()
        for i, batch in enumerate(tqdm.tqdm(dataset)):
            # paths = sample_path(dataset.raw_graph, 4)
            # pos = [sample_pair(random.choice(paths)) for _ in range(sample_nums)]
            # neg = [sample_pair(tuple(random.sample(paths, k=2))) for _ in range(sample_nums)]
            # pos = np.array(pos)
            # neg = np.array(neg)
            # triples = sample_triples(dataset.raw_graph, sample_nums, n_num=8)
            # triples = np.array(triples)  # sample_nums x (2+n_num)
            # batch = np.hstack([pos[:, 0], pos[:, 1], neg[:, 0], neg[:, 1]])
            # batch = np.hstack(triples)
            # batch, con_m = sample_n_nodes(sample_nums, dataset.id_to_father)

            optimizer.zero_grad()

            g, node_features, leaves_embeds, _, new_to_old, path_sim = batch
            bs.append(batch)
            node_features = node_features.to(device)
            paired_nodes = list(range(node_features.shape[0]))
            boxes, scores = model(node_features, leaves_embeds, paired_nodes, g)
            loss = adaptive_BCE(paired_nodes, boxes, g, path_sim) + cls_loss(scores, paired_nodes, g)

            # updating boxes storage
            if i + 1 in dataset.get_milestone():
                print('milestone--' + str(i))
                model.eval()
                model.change_mode()
                for b in bs:
                    g, node_features, leaves_embeds, _, new_to_old, path_sim = b
                    with torch.no_grad():
                        fused = model(node_features, leaves_embeds, paired_nodes, g).detach()
                dataset.update_boxes(fused.squeeze(0), new_to_old)
                model.change_mode()
                model.train()
            # text, img = [], []
            #
            # for i in batch:
            #     inputs = dataset[i]
            #     _id, _name, _text, _imgs = inputs
            #     text = _text if len(text) == 0 else torch.cat([text, _text])
            #     img = _imgs if len(img) == 0 else torch.cat([img, _imgs])
            # text = text.to(device)
            # img = img.to(device)
            # q, k = model(text, img)

            # loss, pos_n = contrastive_loss(q, k, con_m, batch, args.regularization_loss)
            # pe += loss.item()
            # pp += p
            # pn += n

            # loss = contrastive_loss(outs)
            pe += loss.item()
            # pp += pos_n
            total_iters += 1
            # if total_iters % 100 == 0:
            #     print(q[random.randint(0, q.shape[0] - 1)])
            if total_iters % 10 == 0:
                print(e, total_iters, pe / 10)
                writer.add_scalar('total loss', pe / 10, total_iters)
                pe = 0
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
            else:
                return
            scheduler.step()
        bs = []
        __test(dataset, model, metric, 'eval')
        dataset.clear_boxes()
        checkpoint(args.saved_model_path, model)
    __test(dataset, model, metric, 'test')
    writer.close()


def __test(dataset, model, metric, mode):
    assert mode in ('eval', 'test')
    print('{}....'.format(mode))
    metric.clear()
    dataset.clear_boxes()
    dataset.change_mode('train')
    model.eval()
    model.change_mode()
    with torch.no_grad():
        for b in dataset:
            g, node_features, leaves_embeds, _, new_to_old, path_sim = b
            fused = model(node_features, leaves_embeds, None, g).detach()
            dataset.update_boxes(fused.squeeze(0), new_to_old)
        boxes = []
        new_to_old = []
        old_to_new = [-1] * (max(dataset.train) + 1)
        for k, v in dataset.fused_embeddings.items():
            old_to_new[k] = len(new_to_old)
            new_to_old.append(k)
            boxes.append(model.box_decoder(v.unsqueeze(0)))
        boxes = torch.cat(boxes)
        old_to_new = torch.Tensor(old_to_new).type(torch.long)
        dataset.change_mode(mode)
        for n in tqdm.tqdm(dataset):
            node_feature, gt_path, idx = n
            node_feature = node_feature.unsqueeze(0).to(boxes[0].device)
            feature = model.node_encoder(node_feature[:, 0, :], node_feature[:, 1, :].unsqueeze(0))
            box = model.decode_box(feature)
            scores = model.scorer(box, boxes.unsqueeze(0)).view(boxes.shape[0])
            metric.update(scores, gt_path, new_to_old, dataset._tree)
            # novel_in_c, s_in_f = test_entailment(boxes, box, dataset.fs_pairs, old_to_new)
            # metric.update((novel_in_c, s_in_f), gt_path, new_to_old, dataset._tree, dataset.fs_pairs)
    pprint.pprint(metric.show_results())
    dataset.change_mode('train')
    model.change_mode()
    model.train()


def main(args):
    # dataset = get_dataset('/data/home10b/xw/visualCon/datasets_json/handcrafted')
    model, optimizer, scheduler, device = prepare(args, parallel=args.parallel)
    # dataset = NodeSet('/data/home10b/xw/visualCon/datasets_json/',
    #                   '/data/home10b/xw/visualCon/handcrafted', max_imgs_per_node=args.max_imgs_per_node)
    if args.load_dataset_pt is not None:
        d = torch.load(args.load_dataset_pt)
        dataset = TreeSet(d['whole'], d['g'], d['names'], d['descriptions'], d['train'], d['eva'], d['test'])
    else:
        whole_g, G, names, descriptions, tra, test, eva = split_tree_dataset(
            '/data/home10b/xw/visualCon/datasets_json/imagenet_dataset.json')
        dataset = TreeSet(whole_g, G, names, descriptions, tra, test, eva)
    if args.vis_graph:
        vis_graph(get_adj_matrix(dataset.id_to_children), dataset.id_to_name)
    if args.train:
        train(model, dataset, optimizer, scheduler, device, args)
    if args.test:
        if os.path.exists(args.saved_model_path):
            model.load_state_dict(torch.load(args.saved_model_path))
        metric = TreeMetric()
        __test(dataset, model, metric, 'test')


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    main(args)
