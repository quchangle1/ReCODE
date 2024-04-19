# -*- coding: UTF-8 -*-

""" NCF_SLRC
CMD example:
    python main.py --model_name NCF_SLRC --emb_size 32 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 --dataset "MMTD"
"""
import torch
import torch.nn as nn
import torch.distributions
import numpy as np

from models.BaseModel import SequentialModel


class NCF_SLRC(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'layers']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        parser.add_argument('--time_scalar', type=int, default=3600*24*7,
                            help='Time scalar for time intervals.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.layers = eval(args.layers)
        self.time_scalar = args.time_scalar
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.mlp_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mlp_i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.emb_size
        for i, layer_size in enumerate(self.layers):
            self.mlp.append(nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.prediction = nn.Linear(pre_size + self.emb_size, 1, bias=False)

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

        u_ids = feed_dict['user_id']

        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))

        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        mlp_u_vectors = self.mlp_u_embeddings(u_ids)
        mlp_i_vectors = self.mlp_i_embeddings(i_ids)

        mf_vector = mf_u_vectors * mf_i_vectors
        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        base_intensity = self.prediction(output_vector)
        base_intensity = base_intensity.squeeze()
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
