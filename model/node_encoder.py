import torch
from transformers import BertTokenizer, BertModel


class NodeEncoder(torch.nn.Module):
    def __init__(self):
        super(NodeEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = 128
        self.text_encoder = BertModel()
        self.image_feature_extractor = None
