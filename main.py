import pprint
from copy import deepcopy
from queue import Queue

import tqdm
import os

from rich.progress import track

from datasets_torch.treeset import TreeSet
import torch
import torch.profiler
from model.taxbox import TaxBox, TaxBoxWithPLM
import configparser
from utils.loss import insertion_loss, box_constraint_loss, \
    attachment_loss, ranking_loss
from utils.visualize_graph import *
from utils.metric import TreeMetric
from utils.utils import checkpoint, sample_triples, sample_n_nodes, adjust_moco_momentum, \
    save_state, rearrange, collate, obtain_ranks, rescore_by_chatgpt, partition_array
from tensorboardX import SummaryWriter
from utils.scheduler import get_scheduler
import argparse
import multiprocessing as mp

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=32)
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
                             'mag_psy.pt', 'bamboo.pt'])
parser.add_argument("--model", type=str, default='taxbox', choices=['taxbox', 'taxboxp'])
parser.add_argument("--resume", action="store_true")
parser.add_argument("--config", type=str, default='')
parser.add_argument("--loss_r", type=float, default=1)
parser.add_argument("--loss_b", type=float, default=1)
parser.add_argument("--graph", action="store_true")
parser.add_argument("--sep_count", action="store_true")
parser.add_argument("--mp_test", action="store_true")

start_e = 0


def parse_args(args):
    if args.config != '' and os.path.exists(args.config):
        config = configparser.ConfigParser()
        config.read(args.config)
        args.hidden_size = config['model']['hidden_size']
        args.box_dim = config['model']['box_dim']
        # args.


def model_sum(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of model params: {}".format(num_params))


def prepare(args, dataset):
    global start_e
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    if args.model == 'taxbox':
        model = TaxBox(args.hidden_size, args.box_dim, graph_embed=args.graph)
    else:
        model = TaxBoxWithPLM(args.hidden_size, args.box_dim, graph_embed=args.graph)
    model_sum(model)
    model.set_device(device)
    model.to(device)
    optimizer = torch.optim.Adam(params=[
        {'params': model.parameters()}
    ], lr=args.lr)
    total_iters = len(dataset) / args.batch_size * args.epoch
    scheduler = get_scheduler(optimizer, 'plateau', 3, args.epoch)
    if args.parallel:
        model = torch.nn.parallel.DataParallel(model)
        # model.box_decoder_k = torch.nn.parallel.DataParallel(model.box_decoder_k)
        # model.box_decoder_q = torch.nn.parallel.DataParallel(model.box_decoder_q)
        # model.scorer = torch.nn.parallel.DataParallel(model.scorer)
    if args.resume:
        if os.path.exists(args.saved_model_path):
            with open(args.saved_model_path, 'rb') as f:
                model.load_state_dict(torch.load(f))
        if os.path.exists('state_' + args.saved_model_path):
            with open('state_' + args.saved_model_path, 'rb') as f:
                d = torch.load(f)
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
    best = 0
    total_iters, cum_tot_loss, cum_i_loss, cum_a_loss, cum_b_loss, cum_r_loss = 0, 0, 0, 0, 0, 0
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=True, num_workers=8)
    for e in range(args.epoch):
        # torch.cuda.empty_cache()
        if e < start_e:
            continue

        # dataset.shuffle()
        # with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        #         record_shapes=False
        # ) as prof:
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            # if i >= (1+1+3) * 2:
            #     break
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

            boxes, scores = model(queries, p_datas, c_datas, i_idx)
            # boxes, scores = model(inputs, i_idx)
            loss_i = insertion_loss(scores, labels, i_idx)
            # loss_a = attachment_loss(a_scores, labels.reshape(-1, 3)[torch.logical_not(i_idx)]) if i_idx.sum() < \
            #                                                                                        i_idx.shape[0] else 0
            loss_r = ranking_loss(scores[0], labels, i_idx, rank_sims)
            loss_box = box_constraint_loss(boxes.reshape(boxes.shape[0] * boxes.shape[1], 3, -1), sims, reaches)
            loss = args.loss_b * loss_box + loss_i + args.loss_r * loss_r

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
            else:
                return

            # prof.step()

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

        res = __test(dataset, model, i_metric, a_metric, 'eval', device, sep_count=args.sep_count)
        scheduler.step(res['mrr'])
        for k, v in res.items():
            writer.add_scalar(k, v, e)
        dataset.clear_boxes()
        if best < res['mrr']:
            best = res['mrr']
            checkpoint(args.saved_model_path, model)
            save_state('state_' + args.saved_model_path, scheduler, optimizer, e)
    model.load_state_dict(torch.load(args.saved_model_path))
    __test(dataset, model, i_metric, a_metric, 'test', device, sep_count=args.sep_count)
    writer.close()


def _mp_test(q, dataset, model, edge_t, edges, f, s, att_idx, ins_idx, mode, rescore=False):
    for i, n in enumerate(track(dataset, description='{}'.format(mode))):
        node_feature, gt_path, labels, a_idx, query = n
        node_feature = node_feature[0].unsqueeze(0).to(f.device).float()
        feature = model.box_decoder_q(node_feature)

        feature = feature.expand(f.shape)

        scores = model.scorer(feature, f, s)
        scores = model.mul_sim(scores, feature.squeeze(1).unsqueeze(0), f.squeeze(1).unsqueeze(0),
                               s.squeeze(1).unsqueeze(0), ins_idx.unsqueeze(0)).squeeze(0)
        if rescore:
            scores = rescore_by_chatgpt(scores, att_idx, ins_idx, dataset, edge_t, query.split('@')[0], k=20)

        scores, labels = rearrange(scores, edges, labels)
        rank = obtain_ranks(scores, labels)[0]
        q.put(rank)


def __test(dataset, model, i_metric, a_metric, mode, device, sep_count=False):
    assert mode in ('eval', 'test')
    i_metric.clear()
    a_metric.clear()
    dataset.clear_boxes()
    dataset.change_mode('train')
    model.eval()
    with torch.no_grad():
        for b in track(dataset.train, description='Generating seed embeds'):
            p_datas, _ = dataset.generate_pyg_data([(b, -1)], -2, -1)
            emb = model.forward_graph(p_datas[0].to(device))[0].unsqueeze(0) if model.graph_embed else p_datas[0].x[
                0].reshape(1, -1).to(device)
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

        edge = torch.Tensor(list([[old_to_new[e[0]], old_to_new[e[1]]] for e in dataset.edges])).type(
            torch.long).to(
            device)
        edge_t = torch.Tensor(list([e for e in dataset.edges])).type(
            torch.long).to(device)

        f = boxes[edge[:, 0], :].unsqueeze(1)
        s = boxes[edge[:, 1], :].unsqueeze(1)
        att_idx = edge[:, 1] == old_to_new[-1]
        ins_idx = torch.logical_not(att_idx)

        # old_to_new = torch.Tensor(old_to_new).type(torch.long)
        # edge = dataset.edges
        dataset.change_mode(mode)
        # dataset.shuffle()
        import time
        t1 = time.time()

        if args.mp_test:
            p = partition_array(dataset, 10)
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            q = mp.Queue()
            processes = []
            for pp in p:
                processes.append(mp.Process(target=_mp_test, args=(
                    q, pp, model, edge_t, deepcopy(dataset.edges), f, s, att_idx, ins_idx, mode)))

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            while not q.empty():
                rank = q.get()
                i_metric.update(rank)
        else:
            for i, n in enumerate(track(dataset, description='{}'.format(mode))):
                if mode == 'eval' and i > 300:
                    break
                node_feature, gt_path, labels, a_idx, query = n
                node_feature = node_feature[0].unsqueeze(0).to(boxes.device).float()
                feature = model.box_decoder_q(node_feature)

                # feature = model.node_encoder(node_feature[:, 0, :], node_feature[:, 1, :].unsqueeze(0))
                # box = model.box_decoder_q(feature)
                feature = feature.expand(f.shape)

                scores = model.scorer(feature, f, s)
                scores = model.mul_sim(scores, feature.squeeze(1).unsqueeze(0), f.squeeze(1).unsqueeze(0),
                                       s.squeeze(1).unsqueeze(0), ins_idx.unsqueeze(0)).squeeze(0)
                # scores = rescore_by_chatgpt(scores, att_idx, ins_idx, dataset, edge_t, query.split('@')[0], k=20)

                scores, labels = rearrange(scores, dataset.edges, labels)
                rank = obtain_ranks(scores, labels)[0]

                if sep_count:
                    if sum(a_idx) == len(a_idx):
                        rank_a = np.array(rank)[np.array(a_idx, dtype=bool)]
                        a_metric.update(rank_a)
                    rank_i = np.array(rank)[np.array(a_idx, dtype=bool) == False]
                    i_metric.update(rank_i)
                else:
                    i_metric.update(rank)
                # a_metric.update(a_scores.squeeze(1), gt_path, new_to_old, old_to_new, dataset._tree, edge[-dummy_idx:])

        t2 = time.time()
        print((t2 - t1) / 1000)
    pprint.pprint(i_metric.show_results())
    if sep_count:
        pprint.pprint(a_metric.show_results())
    dataset.change_mode('train')
    # model.change_mode()
    model.train()
    # torch.cuda.empty_cache()
    return a_metric.show_results() if sep_count else i_metric.show_results()  # {'insertion': i_metric.show_results(), 'attach': a_metric.show_results()}


def main(args):
    if args.load_dataset_pt is not None:
        if args.model == 'taxbox':
            dataset = TreeSet(args.load_dataset_pt, args.load_dataset_pt.split('.')[0], sample_size=args.sample_size)
        else:
            dataset = TreeSetWithText(args.load_dataset_pt, args.load_dataset_pt.split('.')[0],
                                      sample_size=args.sample_size)

    model, optimizer, scheduler, device = prepare(args, dataset)
    if args.vis_graph:
        vis_graph(get_adj_matrix(dataset.id_to_children), dataset.id_to_name)
    if args.train:
        train(model, dataset, optimizer, scheduler, device, args)
    if args.test:
        if os.path.exists(args.saved_model_path):
            with open(args.saved_model_path, 'rb') as f:
                model.load_state_dict(torch.load(f))
        i_metric = TreeMetric()
        a_metric = TreeMetric()
        __test(dataset, model, i_metric, a_metric, 'test', device, sep_count=args.sep_count)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
