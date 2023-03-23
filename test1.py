import os
import fasttext.util
import fasttext
import torch
import time
import rich
import rtoml
import clip

from rich.progress import track
from rich.prompt import Prompt
from rich.table import Column, Table
from sys import argv

from transformers import AutoTokenizer, CLIPTextModel

from datasets_torch.treeset import TreeSet
from main import model_sum
from model.taxbox import TaxBox
from utils.utils import rescore_by_chatgpt


class demoEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.dataset = None
        self.edges = None
        self.candidates = None
        self.ins_candidates = None
        self.att_candidates = None
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.taxonomy_box_p, self.taxonomy_box_c = None, None
        self.taxid_o2n = None
        self.taxid_n2o = None
        self.ins_idx = None
        self.embedding_model = None
        self.tokenizer = None
        self.embed_type = self.cfg['MODEL']['embed_type']
        self.prompt = Prompt
        self.console = rich.console.Console()

        self.init_model()
        self.init_dataset()
        self.init_embedding_model()
        self.encode_taxonomy()

    def init_model(self):
        model = TaxBox(self.cfg['MODEL']['hidden_size'], self.cfg['MODEL']['box_dim'], self.cfg['MODEL']['graph_embed'])
        model_sum(model)
        model.to(self.device)
        if os.path.exists(self.cfg['MODEL']['model_path']):
            model.load_state_dict(torch.load(self.cfg['MODEL']['model_path']))
            model.eval()
            self.model = model
        else:
            print('No such file: {}'.format(self.cfg['MODEL']['model_path']))
            exit()

    def init_dataset(self):
        if os.path.exists(self.cfg['DATASET']['data_path']):
            dataset = TreeSet(self.cfg['DATASET']['data_path'], self.cfg['DATASET']['data_path'].split('.')[0])
            self.dataset = dataset

        else:
            print('No such file: {}'.format(self.cfg['DATASET']['data_path']))
            exit()

    def init_embedding_model(self):
        if self.embed_type == 'fasttext':
            if os.path.exists(self.cfg['MODEL']['embedding_path']):
                self.embedding_model = fasttext.load_model(self.cfg['MODEL']['embedding_path'])
        elif self.embed_type == 'clip':
            model, _ = clip.load('ViT-B/32', device='cuda')
            model.eval()
            self.embedding_model = model

        elif self.embed_type == 'clip_t':
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            model.eval()
            self.embedding_model = model

    def encode_taxonomy(self):
        self.dataset.change_mode('train')
        with torch.no_grad():
            for b in track(self.dataset.train, description='Generating taxonomy embeds'):
                p_datas, _ = self.dataset.generate_pyg_data([(b, -1)], -2, -1)
                emb = self.model.forward_graph(p_datas[0].to(self.device))[0].unsqueeze(0) if \
                    self.model.graph_embed else p_datas[0].x[0].reshape(1, -1).to(self.device)
                self.dataset.update_boxes(emb, [b])
            boxes = []
            new_to_old = []
            old_to_new = [-1] * (max(self.dataset.train) + 2)
            for k, v in self.dataset.fused_embeddings.items():
                old_to_new[k] = len(new_to_old)
                new_to_old.append(k)
                boxes.append(v.unsqueeze(0))
            boxes = torch.cat(boxes, dim=0)

            old_to_new[-1] = len(new_to_old)
            new_to_old.append(len(self.dataset.whole.nodes()))
            boxes = torch.cat([boxes, torch.zeros_like(boxes[0]).unsqueeze(0).to(self.device)])
            boxes = self.model.box_decoder_k(boxes)
            self.candidates = torch.Tensor(list(self.dataset.edges)).long()
            edge = torch.Tensor(list([[old_to_new[e[0]], old_to_new[e[1]]] for e in self.candidates])).type(
                torch.long).to(self.device)
            att_idx = edge[:, 1] == old_to_new[-1]
            self.ins_idx = torch.logical_not(att_idx)
            self.ins_candidates = self.candidates[self.ins_idx.cpu()]
            self.att_candidates = self.candidates[att_idx.cpu()]
            self.taxonomy_box_p = boxes[edge[:, 0], :].unsqueeze(1)
            self.taxonomy_box_c = boxes[edge[:, 1], :].unsqueeze(1)
            self.taxid_o2n = old_to_new
            self.taxid_n2o = new_to_old
            self.edges = torch.Tensor(list([e for e in self.dataset.edges])).type(
                torch.long).to(self.device)

    def visualize_taxonomy(self):
        pass

    def fasttext_embed(self, text):
        return torch.Tensor(self.embedding_model[text]).unsqueeze(0).to(self.device)

    def clipt_embed(self, text, desc):
        text = text + ', ' + desc
        return torch.nn.functional.normalize(
            (self.embedding_model(
                **self.tokenizer(text, padding=True, return_tensors='pt', truncation=True).to(
                    self.device))['pooler_output']), p=2, dim=-1)

    def clip_embed(self, text, desc):
        text = text + ', ' + desc
        return torch.nn.functional.normalize(
            (self.embedding_model.encode_text(clip.tokenize(text).to(self.device)).float()), p=2, dim=-1)

    def query(self, text, desc='', k=10, sep=False):
        with torch.no_grad():
            if self.embed_type == 'fasttext':
                query_embed = self.fasttext_embed(text)
            elif self.embed_type == 'clip':
                query_embed = self.clip_embed(text, desc)
            elif self.embed_type == 'clip_t':
                query_embed = self.clipt_embed(text, desc)
            else:
                query_embed = self.dataset.embeds[self.dataset.names.index(text)].to(self.device)

        st = time.time()
        query_box = self.model.box_decoder_q(query_embed)
        query_box = query_box.expand(self.taxonomy_box_p.shape)
        scores = self.model.scorer(query_box, self.taxonomy_box_p, self.taxonomy_box_c)
        scores = self.model.mul_sim(scores, query_box.squeeze(1).unsqueeze(0),
                                    self.taxonomy_box_p.squeeze(1).unsqueeze(0),
                                    self.taxonomy_box_c.squeeze(1).unsqueeze(0), self.ins_idx.unsqueeze(0)).squeeze(0)

        scores = rescore_by_chatgpt(scores, torch.logical_not(self.ins_idx), self.ins_idx, self.dataset,
                                    self.edges, text, k=20)

        if sep:
            i_scores = scores[self.ins_idx]
            a_scores = scores[torch.logical_not(self.ins_idx)]
            i_topk = [self.ins_candidates[idx] for idx in i_scores.topk(k)[1]]
            a_topk = [self.att_candidates[idx] for idx in a_scores.topk(k)[1]]
            et = time.time()
            i_topk_tuples = map(
                lambda x: tuple([self.dataset.names[x[0]], self.dataset.names[x[1]]]), i_topk)
            a_topk_tuples = map(
                lambda x: tuple([self.dataset.names[x[0]], 'NONE']), a_topk)
            return {'i_topk': list(i_topk_tuples), 'a_topk': list(a_topk_tuples), 'duration': et - st}
        else:
            topk = scores.topk(k)
            et = time.time()
            topk = [self.candidates[idx] for idx in topk[1]]
            topk_tuples = map(
                lambda x: tuple([self.dataset.names[x[0]], self.dataset.names[x[1]] if x[1] != -1 else 'NOUN']), topk)

            return {'i_topk': list(topk_tuples), 'duration': et - st}

    def construct_res_table(self, res):
        table = Table(show_header=True, header_style="bold magenta")
        if 'a_topk' in res:
            table.add_column('Insertion Topk')
            table.add_column('Attachment Topk')
        else:
            table.add_column('Topk parent')
            table.add_column('Topk child')

        candidates = zip(res['i_topk'], res['a_topk']) if 'a_topk' in res else res['i_topk']
        for c1, c2 in candidates:
            table.add_row(str(c1), str(c2))
        return table

    def interactive_env(self):
        self.console.print('This is a demo system for taxonomy completion based on',
                           '[bold red]{}[/bold red]'.format(self.cfg['DATASET']['dataset_name']))
        while True:
            command = self.prompt.ask('Type command', choices=["query", "quit"])
            if command == 'quit':
                break
            elif command == 'vis':
                if len(self.dataset.train) < 2000:
                    self.visualize_taxonomy()
                else:
                    rich.print('Taxonomy is too large to visualize.')
            else:
                query = self.prompt.ask('Type query concept name')
                desc = self.prompt.ask(
                    'Type query description') if self.embed_type == 'clip' or self.embed_type == 'clip_t' else ''
                k = rich.prompt.IntPrompt.ask('Type topk number', default=10)
                res = self.query(query, desc, k, sep=True)
                self.console.print('Fetch top{} results in {:.4f}s:'.format(k, res['duration']))
                self.console.print(self.construct_res_table(res))


if __name__ == '__main__':
    cfg_path = argv[1]
    cfg = rtoml.load(open(cfg_path, 'r'))
    demo = demoEngine(cfg)
    demo.interactive_env()
