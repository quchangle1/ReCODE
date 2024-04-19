# -*- coding: UTF-8 -*-

""" GRU4Rec
CMD example:
    python main.py --model_name GRU4Rec --emb_size 32 --hidden_size 64 --lr 1e-3 --l2 1e-5 --history_max 20 --dataset "MMTD"
"""

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel


class GRU4Rec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'hidden_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in GRU.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']
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
        prediction = (rnn_vector[:, None, :] * pred_vectors).sum(-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
