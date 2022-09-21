import tqdm

from datasets_torch.handcrafted import Handcrafted
import torch
from model.visgnn import GCN
import configparser
from utils.loss import contrastive_loss
import logging
from utils.graph_operate import test_on_insert
from tensorboardX import SummaryWriter


def get_dataset(root):
    return Handcrafted(root)


def prepare(model_name='', lr=0.0001, step_size=20, gamma=0.1, parallel=False):
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    model = GCN([1024, 512, 512, 1024], 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if parallel:
        model = torch.nn.parallel.DataParallel(model)
    return model, optimizer, scheduler, device


def train(model, dataset, optimizer, device):
    data = dataset[0].to(device)
    writer = SummaryWriter(comment='visgnn')
    model.train()
    for e in tqdm.tqdm(range(200)):
        optimizer.zero_grad()
        out = model(data)
        loss, p, n = contrastive_loss(out, 4, 15, data.raw_graph)
        writer.add_scalar('p loss', p, e)
        writer.add_scalar('n loss', n, e)
        writer.add_scalar('total loss', loss.item(), e)
        # print(e, loss)
        loss.backward()
        optimizer.step()
    writer.close()
    test_on_insert('/data/home10b/xw/visualCon/datasets_json/saved_handcrafted_test.json', data.raw_graph, model(data),
                   model)


def main():
    dataset = get_dataset('/data/home10b/xw/visualCon/datasets_json/handcrafted')
    model, optimizer, scheduler, device = prepare()
    train(model, dataset, optimizer, device)


if __name__ == '__main__':
    main()
