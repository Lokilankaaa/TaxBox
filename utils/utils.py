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
from box_embeddings.modules.intersection import HardIntersection, GumbelIntersection
from box_embeddings.modules.volume import HardVolume, SoftVolume, BesselApproxVolume


def check_common_path(seq_x, seq_y):
    i = 0
    while seq_x[i] == seq_y[i]:
        i += 1
    return i


def sample_path(raw_graph, num=1):
    paths = []
    for i in range(num):
        path = []
        head = raw_graph['object']
        path.append(head['id'])
        while 'children' in head.keys():
            selected = random.choice(head['children'])
            head = selected[selected.keys()[0]]
            path.append(head['id'])
        paths.append(path)
    return paths


def hard_volume(x: torch.Tensor):
    assert x.dim() == 1
    features_len = len(x)
    box_offset = x[features_len:]
    return 2. * box_offset.prod()


def softplus(x, t):
    return torch.log(torch.Tensor(1) + torch.exp(x * 2 / t))


def soft_volume(x: torch.Tensor, t):
    assert x.dim() == 1
    features_len = len(x)
    box_offset = x[features_len:]
    return softplus(box_offset, t).prod()


def hard_intersection(x, y):
    assert x.dim() == 1, y.dim() == 1
    features_len = len(x)
    l = torch.max(x[:features_len] - x[features_len:], y[:features_len] - y[features_len:])
    r = torch.min(x[:features_len] + x[features_len:], y[:features_len] + y[features_len:])
    cen, off = (l + r) / 2, (r - l) / 2
    return torch.stack([cen, off])


# p(x | y)
def conditional_prob(x, y):
    return hard_volume(hard_intersection(x, y)) / hard_volume(y)


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
    print(label, len(inputs))

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
    np.save("features/" + label + ".npy", np.vstack(outputs))


def extract_features_for_imgs(_model, preprocess, path_to_imglist):
    img_list = [os.path.join(path_to_imglist, p) for p in os.listdir(path_to_imglist)]
    for label in img_list:
        extract_feature(_model, preprocess, [os.path.join(label, p) for p in os.listdir(label)], label.split('/')[-1],
                        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), )


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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, preprocess = retrieve_model('resnet50', device)
    extract_features_for_imgs(model, preprocess, 'handcrafted')
    visualize_distribution('features/apple.npy')
