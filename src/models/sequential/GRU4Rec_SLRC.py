# -*- coding: UTF-8 -*-
""" GRU4Rec-SLRC
CMD example:
    python main.py --model_name GRU4Rec_SLRC --emb_size 32 --lr 1e-3 --l2 1e-6 --hidden_size 64 --history_max 20 --dataset "MMTD"
"""
import torch
import torch.nn as nn
import torch.distributions
import numpy as np

from models.BaseModel import SequentialModel


class GRU4Rec_SLRC(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'hidden_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in GRU.')
        parser.add_argument('--time_scalar', type=int, default=3600*24*7,
                            help='Time scalar for time intervals.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.time_scalar = args.time_scalar
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.emb_size)

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
        his_vectors = self.i_embeddings(history)

        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)

        output, hidden = self.rnn(history_packed, None)

        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = hidden[-1].index_select(dim=0, index=unsort_idx)


        pred_vectors = self.i_embeddings(i_ids)
        rnn_vector = self.out(rnn_vector)
        base_intensity = (rnn_vector[:, None, :] * pred_vectors).sum(-1)
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
