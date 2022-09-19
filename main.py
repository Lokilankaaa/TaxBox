from datasets_torch.handcrafted import Handcrafted
import torch
from model.gnn import GCN
import configparser
from utils.loss import contrastive_loss


def get_dataset(root):
    return Handcrafted(root)


def prepare(model_name='', lr=0.01, step_size=10, gamma=0.1, parallel=True):
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    model = GCN(1024, 3).to(device)
    if parallel:
        model = torch.nn.parallel.DataParallel(model)
    optimizer = torch.optim.Adam(model.params(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return model, optimizer, scheduler, device


def train(model, dataset, optimizer, device):
    data = dataset[0].to(device)
    model.train()
    for e in range(20):
        optimizer.zero_grad()
        out = model(data)
        loss = contrastive_loss()
        loss.backward()
        optimizer.step()


def main():
    dataset = get_dataset('/data/home10b/xw/visualCon/datasets_json/handcrafted')
    model, optimizer, scheduler, device = prepare()
    train(model, dataset, optimizer, device)


if __name__ == '__main__':
    main()
