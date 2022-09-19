from .utils import conditional_prob, hard_volume, soft_volume
from .utils import sample_path, check_common_path


def contrastive_loss(x, num_path, num_samples, raw_graph):
    paths = sample_path(raw_graph, num_path)
    pos =
