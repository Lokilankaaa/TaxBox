from typing import Union
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, norm, kurtosis
from scipy.cluster import vq, hierarchy
import torch
import os
import numpy as np
import clip
import random
import torch.nn.functional as F

EPS = 1e-13


def batch_load_img(imgs_, transform, max_k=50):
    imgs = random.choices(imgs_, k=max_k) if len(imgs_) > max_k else imgs_
    inputs = []
    while len(imgs) != 0:
        i = imgs[0]
        imgs.remove(i)
        try:
            inputs.append(transform(Image.open(i).convert('RGB')).unsqueeze(0))
        except:
            imgs.append(random.choice(imgs_))
    return inputs


def get_graph_box_embedding(dataset, model, save=True, load=True):
    if load and os.path.exists('embeddings.npy'):
        embeddings = np.load('embeddings.npy')
        return embeddings

    embeddings = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    for d in dataset:
        _, _, t, i = d
        t = t.to(device)
        i = i.to(device)
        model.eval()
        with torch.no_grad():
            out = model(t, i)
            embeddings.append(out.cpu().numpy())

    embeddings = np.array(embeddings).reshape(-1, 256)
    if save:
        np.save('embeddings.npy', embeddings)

    return embeddings


def checkpoint(path_to_save, model):
    print('saving model to ' + path_to_save)
    if type(model) == torch.nn.DataParallel:
        sd = model.module.state_dict()
    else:
        sd = model.state_dict()
    torch.save(sd, path_to_save)


def sample_pair(path, relation='descendant'):
    assert relation in ('son', 'descendant')
    if type(path) != tuple:
        pathx = path
        path_len = len(pathx)
        anchor = random.randint(0, path_len - 2)
        if relation == 'son':
            return pathx[anchor], pathx[anchor + 1]
        else:
            return pathx[anchor], pathx[random.randint(anchor + 1, path_len - 1)]
    else:
        pathx, pathy = path
        start = check_common_path(pathx, pathy)
        anchorx = random.randint(start, len(pathx) - 1)
        anchory = random.randint(start, len(pathy) - 1)
        return pathx[anchorx], pathy[anchory]


def check_common_path(seq_x, seq_y):
    i = 0
    while seq_x[i] == seq_y[i]:
        i += 1
    return i


def check_same_path(seq_x, seq_y):
    # if type(seq_x) == list:
    #     seq_x = np.array(seq_x, dtype=np.int32)
    #     if seq_x.ndim == 1:
    #         seq_x = np.expand_dims(seq_x, axis=0)
    if type(seq_y) == list:
        seq_y = np.array(seq_y, dtype=np.int32)
    for s in seq_x:
        s = np.array(s, dtype=np.int32)
        if len(s) == len(seq_y) and (s - seq_y).sum(-1) == 0:
            return True

    return False


def sample_triples(raw_graph, num=1, n_num=4):
    pointer = []
    paths = sample_path(raw_graph, num, pointer)
    triple = []
    for i, p in enumerate(paths):
        anchor = random.choice(range(1, len(p)))
        pos = random.choice(range(0, anchor))
        head = pointer[i][anchor - 1]
        npath = []
        negs = []
        while len(negs) < n_num:
            if len(head['children']) > 1:
                nhead = random.choice(head['children'])
                nstart = nhead['id']
                while p[anchor] == nstart:
                    nhead = random.choice(head['children'])
                    nstart = nhead['id']
                npath.append(nstart)
                while len(nhead['children']) > 0:
                    nhead = random.choice(nhead['children'])
                    npath.append(nhead['id'])
            else:
                npath = sample_path(raw_graph)[0]
                while check_same_path([p], npath):
                    npath = sample_path(raw_graph)[0]
                common_i = check_common_path(npath, p)
                npath = npath[common_i:]
            neg = random.choice(npath)
            negs.append(neg)
        triple.append([p[anchor], p[pos]] + negs)
    return triple


def sample_path(raw_graph, num=1, pointer=None):
    paths = []
    while len(paths) < num:
        path = []
        pp = []
        head = raw_graph
        path.append(head['id'])
        if pointer is not None:
            pp.append(head)
        while len(head['children']) > 0:
            head = random.choice(head['children'])
            path.append(head['id'])
            if pointer is not None:
                pp.append(head)
        if len(paths) != 0 and check_same_path(paths, path):
            continue
        paths.append(path)
        if pointer is not None:
            pointer.append(pp)
    return paths


def hard_volume(x: torch.Tensor, box_mode=False):
    assert x.dim() == 2
    z, Z = x.chunk(2, -1)
    if box_mode:
        box_length = Z - z
    else:
        box_length = Z * 2
    return box_length.prod(-1)


def softplus(x, t):
    # EPS in case of nan
    return F.softplus(x, t)


def soft_volume(x: torch.Tensor, t=1, box_mode=False):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    assert x.dim() == 2
    z, Z = x.chunk(2, -1)
    if box_mode:
        box_length = Z - z
    else:
        box_length = Z * 2
    return softplus(box_length, 1 / t).clamp(min=EPS)


def bessel_approx_volume(x, t=1, box_mode=False):
    EULER_GAMMA = 0.57721566490153286060
    if x.dim() == 1:
        x = x.unsqueeze(0)
    z, Z = x.chunk(2, -1)
    if box_mode:
        box_length = Z - z
    else:
        box_length = Z * 2
    return softplus(box_length - 2 * EULER_GAMMA * t, 1 / t).clamp(min=EPS)


def hard_intersection(x, y, box_mode=False):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    assert x.dim() == 2, y.dim() == 2
    xz, xZ = x.chunk(2, -1)
    yz, yZ = y.chunk(2, -1)
    if box_mode:
        Z = torch.min(xZ, yZ)
        z = torch.max(xz, yz)
        return torch.cat([z, Z], dim=-1)
    else:
        Z = torch.min(xz + xZ, yz + yZ)
        z = torch.max(xz - xZ, yz - yZ)
        cen, off = (z + Z) / 2, (Z - z) / 2
        return torch.cat([cen, off], dim=-1)


def log_sum_exp(x, y):
    return torch.logaddexp(x, y)


def gumbel_intersection(x, y, beta=1, box_mode=False):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    xz, xZ = x.chunk(2, -1)
    yz, yZ = y.chunk(2, -1)
    if box_mode:
        z = -beta * (log_sum_exp(-xZ / beta, -yZ / beta))
        Z = beta * (log_sum_exp(xz / beta, yz / beta))
        return torch.cat([z, Z], dim=-1)
    else:
        r = -beta * (log_sum_exp(-(xz + xZ) / beta, -(yz + yZ) / beta))
        l = beta * (log_sum_exp((xz - xZ) / beta, (yz - yZ) / beta))
        cen, off = (l + r) / 2, (r - l) / 2
        return torch.cat([cen, off], dim=-1)


# log p(x|y)
def log_conditional_prob(x, y, box_mode=False):
    inter = hard_intersection(x, y, box_mode=box_mode)
    log_prob = (torch.log(soft_volume(inter, box_mode=box_mode)) - torch.log(soft_volume(y, box_mode=box_mode))).sum(
        -1).clamp(max=-EPS)
    # torch.log(soft_volume(inter)
    return log_prob


# p(x | y)
def conditional_prob(x, y, box_mode=False):
    return (soft_volume(hard_intersection(x, y, box_mode=box_mode), box_mode=box_mode) / soft_volume(y,
                                                                                                     box_mode=box_mode)).prod(
        -1).clamp(min=EPS)


def extract_feature(_model: torch.nn.Module, preprocess, imgs: Union[str, list[str]], label, device):
    # if label in ['flower', 'shirt', 'banana', 'watermelon', 'apple', 'cat']:
    #     return
    inputs = []
    outputs = []
    imgs = [imgs] if type(imgs) == str else imgs
    for i in imgs:
        try:
            inputs.append(preprocess(i).to(device))
        except:
            continue

    # print(label, len(inputs))

    # inputs = torch.stack(inputs).squeeze(1)
    def get_batch_feature(_features, batch_size):
        feature_num = len(_features)
        assert batch_size <= feature_num
        start = 0
        end = batch_size
        while start < feature_num:
            yield _features[start: end]
            start += batch_size
            end = end + batch_size if end <= feature_num else feature_num

    for inputs in get_batch_feature(inputs, 8):
        inputs = torch.stack(inputs).squeeze(1).to(device)
        with torch.no_grad():
            features = _model(inputs).cpu().numpy()
        outputs.append(features)
    res = np.vstack(outputs)
    np.save("datasets_json/handcrafted/raw/features/" + label + ".npy", res)
    return res


def extract_features_for_imgs(_model, preprocess, path_to_imglist, l=None):
    img_list = [os.path.join(path_to_imglist, p) for p in os.listdir(path_to_imglist)]
    f = []
    if l is not None:
        label = os.path.join(path_to_imglist, l)
        return extract_feature(_model, preprocess,
                               [os.path.join(label, p) for p in os.listdir(label)],
                               l, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    else:
        for label in img_list:
            f.append(extract_feature(_model, preprocess, [os.path.join(label, p) for p in os.listdir(label)],
                                     label.split('/')[-1],
                                     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

        return f


def visualize_distribution(path_to_npy):
    features = np.load(path_to_npy).T
    print(features.shape)
    show_dims = random.sample(list(range(0, features.shape[0])), 8)
    for dim in show_dims:
        feature = features[dim, :]
        norm_feature = (feature - feature.min()) / (feature.max() - feature.min())
        density = gaussian_kde(norm_feature)
        x = np.linspace(0, 1, 10000)
        y = density(x)
        plt.plot(x, y, label='feature_distribution')
        plt.plot(x, norm.pdf(x, norm_feature.mean(), norm_feature.std()), label='normal_distribution')
        plt.title('mean:' + str(norm_feature.mean()) + '-' + 'var:' + str(norm_feature.var()))
        plt.legend(loc='upper right')
        plt.show()


def cluster_feature_dim(path_to_npy):
    features = np.load(path_to_npy).T

    def cluster_one_dim_with_kurtosis(feature):
        norm_feature = (feature - feature.min()) / (feature.max() - feature.min())
        k = kurtosis(norm_feature)
        return k

    kurtosis_each_dim = np.array(list(map(cluster_one_dim_with_kurtosis, features)))
    vq.kmeans(kurtosis_each_dim, 2)


def retrieve_model(model_name, device):
    model_dict = {'resnet50': 'nvidia_resnet50', 'clip': 'ViT-B/32'}
    assert model_name in model_dict.keys()
    _model, _preprocess = None, None
    if model_name == 'resnet50':
        _model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_dict['resnet50'], pretrained=True)
        _model_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
        _preprocess = _model_utils.prepare_input_from_uri
        _model.fc = torch.nn.Identity()
    elif model_name == 'clip':
        _model, _pre = clip.load('ViT-B/32', device)

        def _p(uri):
            i = Image.open(uri).convert('RGB')
            return _pre(i)

        _preprocess = _p

    _model.eval().to(device)
    return _model if model_name != 'clip' else _model.encode_image, _preprocess


if __name__ == '__main__':
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, preprocess = retrieve_model('resnet50', device)
    # extract_features_for_imgs(model, preprocess, 'handcrafted')
    visualize_distribution('features/apple.npy')
