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
from box_embeddings.modules.intersection import HardIntersection, GumbelIntersection
from box_embeddings.modules.volume import HardVolume, SoftVolume, BesselApproxVolume

EPS = 1e-4


def checkpoint(path_to_save, model):
    torch.save(path_to_save, model.state_dict())


def sample_pos_and_negs():
    pass


def sample_pair(path, relation='son'):
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
    if type(seq_x) == list:
        seq_x = np.array(seq_x, dtype=np.int32)
    if type(seq_y) == list:
        seq_y = np.array(seq_y, dtype=np.int32)

    return ((seq_x - seq_y).sum(-1) == 0).sum() != 0


def sample_path(raw_graph, num=1):
    paths = []
    while len(paths) < num:
        path = []
        head = raw_graph['object']
        path.append(head['id'])
        while 'children' in head.keys():
            selected = random.choice(head['children'])
            head = selected[list(selected.keys())[0]]
            path.append(head['id'])
        if len(paths) != 0 and check_same_path(paths, path):
            continue
        paths.append(path)
    return paths


def hard_volume(x: torch.Tensor):
    assert x.dim() == 2
    features_len = x.shape[-1] // 2
    box_offset = x[:, features_len:]
    return 2. * box_offset.prod(-1)


def softplus(x, t):
    # EPS in case of nan
    return F.softplus(x, t)


def soft_volume(x: torch.Tensor, t=1):
    assert x.dim() == 2
    features_len = x.shape[-1] // 2
    box_length = x[:, features_len:] * 2
    return (t * softplus(box_length, t)).clamp(min=EPS)


def hard_intersection(x, y):
    assert x.dim() == 2, y.dim() == 2
    x_center, x_offset = x.chunk(2, -1)
    y_center, y_offset = y.chunk(2, -1)
    r = torch.min(x_center + x_offset, y_center + y_offset)
    l = torch.max(x_center - x_offset, y_center - y_offset)
    cen, off = (l + r) / 2, (r - l) / 2
    return torch.cat([cen, off], dim=-1)


# log p(x|y)
def log_conditional_prob(x, y):
    inter = hard_intersection(x, y)
    log_prob = torch.log(soft_volume(inter) / soft_volume(y)).sum(-1).clamp(max=-EPS)
    return log_prob


# p(x | y)
def conditional_prob(x, y):
    return (soft_volume(hard_intersection(x, y)) / soft_volume(y)).prod(-1)


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
    np.save("datasets_json/handcrafted/raw/features/" + label + ".npy",res)
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
