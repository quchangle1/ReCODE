# -*- coding: UTF-8 -*-

""" MF
CMD example:
    python main.py --model_name MF --emb_size 32 --lr 5e-4 --l2 1e-7 --dataset "MMTD"
"""

import torch.nn as nn

from models.BaseModel import GeneralModel


class MF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']
        i_ids = feed_dict['item_id']
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)

        prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
