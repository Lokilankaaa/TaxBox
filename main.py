import tqdm

from datasets_torch.handcrafted import Handcrafted
import torch
from model.visgnn import GCN
import configparser
from utils.loss import contrastive_loss

from torch_geometric.datasets import TUDataset


def get_dataset(root):
    return Handcrafted(root)
    # return TUDataset(root='/tmp/ENZYMES', name='ENZYMES')


def prepare(model_name='', lr=0.01, step_size=30, gamma=0.1, parallel=False):
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    model = GCN(1024, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if parallel:
        model = torch.nn.parallel.DataParallel(model)
    return model, optimizer, scheduler, device


def train(model, dataset, optimizer, device):
    data = dataset[0].to(device)
    model.train()
    for e in tqdm.tqdm(range(100)):
        optimizer.zero_grad()
        out = model(data)
        loss = contrastive_loss(out, 4, 10, data.raw_graph)
        # print(e, loss)
        loss.backward()
        optimizer.step()


def main():
    dataset = get_dataset('/data/home10b/xw/visualCon/datasets_json/handcrafted')
    model, optimizer, scheduler, device = prepare()
    train(model, dataset, optimizer, device)


if __name__ == '__main__':
    main()
