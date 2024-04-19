# -*- coding: UTF-8 -*-
""" MF-SLRC
CMD example:
    python main.py --model_name MF_SLRC --emb_size 32 --lr 1e-3 --l2 1e-6 --dataset "MMTD"
"""
import torch
import torch.nn as nn
import torch.distributions
import numpy as np

from models.BaseModel import SequentialModel


class MF_SLRC(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--time_scalar', type=int, default=3600*24*7,
                            help='Time scalar for time intervals.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.time_scalar = args.time_scalar
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.global_alpha = nn.Parameter(torch.tensor(1.0))
        self.alphas = nn.Embedding(self.item_num, 1)
        self.pis = nn.Embedding(self.item_num, 1)
        self.betas = nn.Embedding(self.item_num, 1)
        self.sigmas = nn.Embedding(self.item_num, 1)
        self.mus = nn.Embedding(self.item_num, 1)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']
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

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        base_intensity = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(-1)
        base_intensity = base_intensity
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
                        t=t+1
                        if t==5:
                            break
                relational_interval.append(interval)
            feed_dict['relational_interval'] = np.array(relational_interval, dtype=np.float32)
            return feed_dict
