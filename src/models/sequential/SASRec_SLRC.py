# -*- coding: UTF-8 -*-

""" SASRec_SLRC
CMD example:
    python main.py --model_name SASRec_SLRC --emb_size 32 --lr 1e-4 --l2 1e-5 --num_layers 1 --num_heads 1 --history_max 20 \
    --dataset "MMTD"
"""
import torch
import torch.nn as nn
import torch.distributions
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers


class SASRec_SLRC(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--time_scalar', type=int, default=3600*24*7,
                            help='Time scalar for time intervals.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self.time_scalar = args.time_scalar
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

        self.global_alpha = nn.Parameter(torch.tensor(1.0))
        self.alphas = nn.Embedding(self.item_num, 1)
        self.pis = nn.Embedding(self.item_num, 1)
        self.betas = nn.Embedding(self.item_num, 1)
        self.sigmas = nn.Embedding(self.item_num, 1)
        self.mus = nn.Embedding(self.item_num, 1)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']
        r_intervals = feed_dict['relational_interval']
        alphas = self.global_alpha + self.alphas(i_ids)
        pis, mus = self.pis(i_ids) + 0.5, self.mus(i_ids) + 1
        betas = (self.betas(i_ids) + 1).clamp(min=1e-10, max=10)
        sigmas = (self.sigmas(i_ids) + 1).clamp(min=1e-10, max=10)
        mask = (r_intervals >= 0).float()
        delta_t = r_intervals * mask
        norm_dist = torch.distributions.normal.Normal(mus, sigmas)
        exp_dist = torch.distributions.exponential.Exponential(betas, validate_args=False)
        decay = pis * exp_dist.log_prob(delta_t).exp() + (1 - pis) * norm_dist.log_prob(delta_t).exp()
        excitation = (alphas * decay * mask).sum(-1)

        history = feed_dict['history_items']
        lengths = feed_dict['lengths']
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors


        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)

        for block in self.transformer_block:
            his_vectors = block(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()

        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]

        i_vectors = self.i_embeddings(i_ids)
        base_intensity = (his_vector[:, None, :] * i_vectors).sum(-1)
        prediction = base_intensity + excitation
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

    class Dataset(SequentialModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            user_id, time = self.data['user_id'][index], self.data['time'][index]
            history_item, history_time = feed_dict['all_history_items'], feed_dict['all_history_times']
            relational_interval = list()
            for i, target_item in enumerate(feed_dict['item_id']):
                interval = np.ones(5, dtype=float) * -1
                t = 0
                for j in range(len(history_item))[::-1]:
                    if history_item[j] == target_item:
                        interval[t] = (time - history_time[j]) / self.model.time_scalar
                        t = t + 1
                        if t == 5:
                            break
                relational_interval.append(interval)
            feed_dict['relational_interval'] = np.array(relational_interval, dtype=np.float32)
            return feed_dict
