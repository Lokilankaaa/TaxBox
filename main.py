import pprint

import tqdm
import os

# from datasets_torch.handcrafted import Handcrafted
# from datasets_torch.nodeset import NodeSet
from datasets_torch.treeset import TreeSet
import torch
from model.node_encoder import NodeEncoder, twinTransformer, MultimodalNodeEncoder, TreeKiller, TMNModel, BoxTax
import configparser
from utils.loss import triplet_loss, contrastive_loss, adaptive_BCE, insertion_loss, tmnloss, box_constraint_loss, \
    attachment_loss, ranking_loss
import logging
from utils.graph_operate import test_on_insert, test_entailment
from utils.visualize_graph import *
from utils.metric import TreeMetric
from utils.utils import sample_path, sample_pair, checkpoint, sample_triples, sample_n_nodes, adjust_moco_momentum, \
    save_state, rearrange, collate
from tensorboardX import SummaryWriter
from utils.scheduler import get_scheduler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--box_dim", type=int, default=128)
parser.add_argument("--regularization_loss", type=bool, default=True)
# parser.add_argument("--gpu_id", type=str, default='0,1')
parser.add_argument("--saved_model_path", type=str, default='model.pth')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--vis_graph", action="store_true")
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--load_dataset_pt", type=str, default='wn_small.pt',
                    choices=['mesh.pt', 'wn_food.pt', 'wn_small.pt', 'wn_whole.pt', 'wn_verb.pt', 'mag_cs.pt',
                             'mag_psy.pt'])
parser.add_argument("--resume", action="store_true")
parser.add_argument("--config", type=str, default='')

start_e = 0


def parse_args(args):
    if args.config != '' and os.path.exists(args.config):
        config = configparser.ConfigParser()
        config.read(args.config)
        args.hidden_size = config['model']['hidden_size']
        args.box_dim = config['model']['box_dim']
        # args.

    def model_sum(model):
        num_params = sum(p.numel() for p in model.parameters())
        print("number of params: {}".format(num_params))

    def prepare(args, dataset):
        global start_e
        device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

        model = BoxTax(args.hidden_size, args.box_dim)
        model_sum(model)
        model.to(device)
        optimizer = torch.optim.Adam(params=[
            # {'params': model.node_encoder.parameters(), 'lr': args.lr},
            # {'params': model.struct_encoder.parameters(), 'lr': args.lr},
            # {'params': model.box_decoder_k.parameters(), 'lr': 2 * args.lr},
            # {'params': model.box_decoder_q.parameters(), 'lr': 2 * args.lr},
            # {'params': model.scorer.parameters(), 'lr': 5 * args.lr}
            {'params': model.parameters()}
        ], lr=args.lr)
        total_iters = len(dataset) / args.batch_size * args.epoch
        scheduler = get_scheduler(optimizer, 'plateau', int(3 / args.epoch * total_iters), total_iters)
        if args.parallel:
            model = torch.nn.parallel.DataParallel(model)
            # model.node_encoder = torch.nn.parallel.DataParallel(model.node_encoder)
            # model.struct_encoder.fusion = torch.nn.parallel.DataParallel(model.struct_encoder.fusion)
            # model.box_decoder_k = torch.nn.parallel.DataParallel(model.box_decoder_k)
            # model.box_decoder_q = torch.nn.parallel.DataParallel(model.box_decoder_q)
            # model.scorer = torch.nn.parallel.DataParallel(model.scorer)
        if args.resume:
            if os.path.exists(args.saved_model_path):
                model.load_state_dict(torch.load(args.saved_model_path))
            if os.path.exists('state_' + args.saved_model_path):
                d = torch.load('state_' + args.saved_model_path)
                optimizer.load_state_dict(d['optimizer'])
                # scheduler.load_state_dict(d['scheduler'])
                start_e = d['e']
        return model, optimizer, scheduler, device

    def train(model, dataset, optimizer, scheduler, device, args):
        writer = SummaryWriter(comment='NodeEncoder')
        i_metric = TreeMetric()
        a_metric = TreeMetric()
        model.train()
        # sample_nums = args.sample_nums
        best = 99999999
        total_iters, cum_tot_loss, cum_i_loss, cum_a_loss, cum_b_loss, cum_r_loss = 0, 0, 0, 0, 0, 0
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=True, num_workers=0)
        for e in range(args.epoch):
            # torch.cuda.empty_cache()
            if e < start_e:
                continue

            # dataset.shuffle()

            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                # if i % 10 == 0:
                #     torch.cuda.empty_cache()
                optimizer.zero_grad()

                queries, p_datas, c_datas, labels, sims, reaches, i_idx, rank_sims = batch
                # inputs, labels, sims, reaches, i_idx, rank_sims = batch
                # inputs = inputs.to(device)
                queries = queries.to(device)
                labels = labels.to(device)
                sims = sims.reshape(-1, 2).to(device)
                i_idx = i_idx.to(device)
                reaches = reaches.reshape(-1, 4).to(device)

                if args.model == 'boxtax':
                    boxes, scores = model(queries, p_datas, c_datas, i_idx)
                    # boxes, scores = model(inputs, i_idx)
                    loss_i = insertion_loss(scores, labels, i_idx)
                    # loss_a = attachment_loss(a_scores, labels.reshape(-1, 3)[torch.logical_not(i_idx)]) if i_idx.sum() < \
                    #                                                                                        i_idx.shape[0] else 0
                    loss_r = ranking_loss(scores[0], labels, i_idx, rank_sims)
                    loss_box = box_constraint_loss(boxes.reshape(boxes.shape[0] * boxes.shape[1], 3, -1), sims, reaches)
                    loss = loss_box + loss_i + loss_r
                elif args.model == 'tmn':
                    q, p, c = inputs.chunk(3, 2)
                    out = model(q, p, c)
                    loss = tmnloss(out, labels)
                # updating boxes storage

                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
                else:
                    return
                cum_tot_loss += loss.item()
                cum_i_loss += loss_i.item() if hasattr(loss_i, 'item') else 0
                # cum_a_loss += loss_a.item() if hasattr(loss_a, 'item') else 0
                cum_b_loss += loss_box.item()
                cum_r_loss += loss_r.item()

                total_iters += 1
                if total_iters % 10 == 0:
                    print("Epoch {},{}. loss:{}, i_loss:{}, b_loss:{}, a_loss:{}, r_loss:{}".format(e, total_iters,
                                                                                                    cum_tot_loss / 10,
                                                                                                    cum_i_loss / 10,
                                                                                                    cum_b_loss / 10,
                                                                                                    cum_a_loss / 10,
                                                                                                    cum_r_loss / 10))
                    writer.add_scalar('total loss', cum_tot_loss / 10, total_iters)
                    writer.add_scalar('i_loss', cum_i_loss / 10, total_iters)
                    writer.add_scalar('b_loss', cum_b_loss / 10, total_iters)
                    cum_tot_loss, cum_i_loss, cum_b_loss, cum_a_loss, cum_r_loss = 0, 0, 0, 0, 0

                # scheduler.step()
            bs = []
            # torch.cuda.empty_cache()
            res = __test(dataset, model, i_metric, a_metric, 'eval', device)
            scheduler.step(res['micro_mr'])
            for k, v in res.items():
                writer.add_scalar(k, v, e)
            dataset.clear_boxes()
            if best > res['macro_mr']:
                best = res['macro_mr']
                checkpoint(args.saved_model_path, model)
                save_state('state_' + args.saved_model_path, scheduler, optimizer, e)
        model.load_state_dict(torch.load(args.saved_model_path))
        __test(dataset, model, i_metric, a_metric, 'test', device)
        writer.close()

    def __test(dataset, model, i_metric, a_metric, mode, device):
        assert mode in ('eval', 'test')
        print('{}....'.format(mode))
        i_metric.clear()
        a_metric.clear()
        dataset.clear_boxes()
        dataset.change_mode('train')
        model.eval()
        with torch.no_grad():
            for b in tqdm.tqdm(dataset.train, desc='generate training embeds'):
                p_datas, _ = dataset.generate_pyg_data([(b, -1)], -2, 0)
                emb = model.forward_graph(p_datas[0].to(device))[0].unsqueeze(0)
                # emb = dataset._database[b][0].unsqueeze(0).to(device).float()
                dataset.update_boxes(emb, [b])
            boxes = []
            new_to_old = []
            old_to_new = [-1] * (max(dataset.train) + 2)
            for k, v in dataset.fused_embeddings.items():
                old_to_new[k] = len(new_to_old)
                new_to_old.append(k)
                boxes.append(v.unsqueeze(0))

            boxes = torch.cat(boxes, dim=0)

            old_to_new[-1] = len(new_to_old)
            new_to_old.append(len(dataset.whole.nodes()))
            boxes = torch.cat([boxes, torch.zeros_like(boxes[0]).unsqueeze(0).to(device)])
            boxes = model.box_decoder_k(boxes)

            # old_to_new[len(dataset.whole.nodes())] = len(new_to_old)
            # new_to_old.append(len(old_to_new) - 1)

            # dummy_box = torch.zeros(1, boxes.shape[-1]).to(boxes.device)
            # _boxes = torch.cat([boxes, dummy_box], dim=0)  # boxes: n * (n+1) * d
            edge = torch.Tensor(list([[old_to_new[e[0]], old_to_new[e[1]]] for e in dataset.edges])).type(
                torch.long).to(
                device)
            # edge_dummy = torch.Tensor(
            #     list([[old_to_new[n], old_to_new[len(dataset.whole.nodes())]] for n in dataset._tree.nodes()]))
            # edge = torch.cat([edge, edge_dummy]).type(torch.long).to(device)
            f = boxes[edge[:, 0], :].unsqueeze(1)
            s = boxes[edge[:, 1], :].unsqueeze(1)
            att_idx = edge[:, 1] == old_to_new[-1]
            ins_idx = torch.logical_not(att_idx)

            # old_to_new = torch.Tensor(old_to_new).type(torch.long)
            # edge = dataset.edges
            dataset.change_mode(mode)
            dataset.shuffle()
            for i, n in enumerate(tqdm.tqdm(dataset)):
                # if mode == 'eval' and i > 300:
                #     break
                node_feature, gt_path, labels = n
                node_feature = node_feature[0].unsqueeze(0).to(boxes.device).float()
                feature = model.box_decoder_q(node_feature)

                # feature = model.node_encoder(node_feature[:, 0, :], node_feature[:, 1, :].unsqueeze(0))
                # box = model.box_decoder_q(feature)
                feature = feature.expand(f.shape)

                scores = model.scorer(feature, f, s)
                # a_scores = model.a_scorer(feature[att_idx], f[att_idx])
                # scores = torch.cat([i_scores, a_scores], dim=0)
                # scores = torch.zeros_like(ins_idx).float().to(feature.device)
                # scores[ins_idx] += i_scores.squeeze(-1)
                # scores[att_idx] += a_scores.squeeze(-1)
                scores = model.mul_sim(scores, feature.squeeze(1).unsqueeze(0), f.squeeze(1).unsqueeze(0),
                                       s.squeeze(1).unsqueeze(0), ins_idx.unsqueeze(0)).squeeze(0)

                first = torch.argmax(scores).item()
                scores, labels = rearrange(scores, dataset.edges, labels)
                i_metric.update(scores, labels, gt_path, new_to_old, first, dataset._tree, edge)
                # a_metric.update(a_scores.squeeze(1), gt_path, new_to_old, old_to_new, dataset._tree, edge[-dummy_idx:])
        pprint.pprint(i_metric.show_results())
        dataset.change_mode('train')
        # model.change_mode()
        model.train()
        # torch.cuda.empty_cache()
        return i_metric.show_results()  # {'insertion': i_metric.show_results(), 'attach': a_metric.show_results()}

    def main(args):
        # dataset = get_dataset('/data/home10b/xw/visualCon/datasets_json/handcrafted')
        # dataset = NodeSet('/data/home10b/xw/visualCon/datasets_json/',
        #                   '/data/home10b/xw/visualCon/handcrafted', max_imgs_per_node=args.max_imgs_per_node)
        if args.load_dataset_pt is not None:
            dataset = TreeSet(args.load_dataset_pt, args.load_dataset_pt.split('.')[0])
        # else:
        #     whole_g, G, names, descriptions, tra, test, eva = split_tree_dataset(
        #         '/data/home10b/xw/visualCon/datasets_json/imagenet_dataset.json')
        #     dataset = TreeSet(whole_g, G, names, descriptions, tra, test, eva)
        model, optimizer, scheduler, device = prepare(args, dataset)
        if args.vis_graph:
            vis_graph(get_adj_matrix(dataset.id_to_children), dataset.id_to_name)
        if args.train:
            train(model, dataset, optimizer, scheduler, device, args)
        if args.test:
            if os.path.exists(args.saved_model_path):
                model.load_state_dict(torch.load(args.saved_model_path))
            i_metric = TreeMetric()
            a_metric = TreeMetric()
            __test(dataset, model, i_metric, a_metric, 'test', device)

    if __name__ == '__main__':
        args = parser.parse_args()
        main(args)
