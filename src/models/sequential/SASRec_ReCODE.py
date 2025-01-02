# -*- coding: UTF-8 -*-

""" SASRec_ReCODE
CMD example:
    python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np

from torchdiffeq import odeint_adjoint as odeint
from models.BaseModel import SequentialModel
from utils import layers

class ODEFunc(nn.Module):

    def __init__(self,hidden_size):
        super(ODEFunc, self).__init__()
        self.hidden_size=hidden_size
        self.net = nn.Sequential(
            nn.Linear(16, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 16),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class SASRec_ReCODE(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads','hidden_size','method','steps','time_scalar']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in GRU.')
        parser.add_argument('--method', type=str, default='euler',
                            help="the method for ODE")
        parser.add_argument('--steps', type=int, default=77,
                            help="the num of step for ODE")
        parser.add_argument('--time_scalar', type=int, default=3600 * 24 * 7,
                            help='Time scalar for time intervals.')
        parser.add_argument('--step_size', type=int, default=10,
                            help='step_size for ODE') 
        
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self.hidden_size = args.hidden_size
        self.method = args.method
        self.steps = args.steps
        self.time_scalar = args.time_scalar
        self.step_size = args.step_size
        
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.model = ODEFunc(self.hidden_size).to(self.device)
        
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

        self.en = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 16)
        )

        self.o_net = nn.Sequential(
            nn.Linear(16, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, feed_dict):
        self.check_list = []
        items = feed_dict['item_id']
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

        i_vectors = self.i_embeddings(items)
        base_intensity = (his_vector[:, None, :] * i_vectors).sum(-1)

        items_expand = items.reshape(items.shape[0] * items.shape[1])
        time_interval = torch.ceil(feed_dict["relational_interval"] / self.time_scalar).type(
            torch.LongTensor)
        time_interval = time_interval.reshape(-1, time_interval.shape[-1]).cuda()
        u_vector = his_vector
        i_vector = self.i_embeddings(items_expand)
        u_vector = torch.cat([u_vector] * items.shape[1], dim=0)
        init_state = [u_vector, i_vector]
        init_state = torch.cat(init_state, dim=-1)

        init = self.en(init_state)
        init = nn.functional.normalize(init)

        t = torch.linspace(0, self.steps - 1, self.steps).to(self.device)

        
        outputs = odeint(self.model, init, t, method=self.method, options={"perturb": "True", "step_size": self.step_size})
        outputs = outputs.transpose(0, 1)

        outputs = torch.squeeze(self.o_net(outputs))
        p = torch.zeros(outputs.shape[0]).cuda()
        p = p.unsqueeze(1)
        outputs = torch.cat((p, outputs), 1)
        score = torch.gather(outputs, dim=1, index=time_interval)
        excit1 = score.reshape(items.shape[0], items.shape[1], -1)
        excitation = (excit1).sum(-1)

        prediction = base_intensity + excitation

        return {'prediction': prediction.view(batch_size, -1)}

    class Dataset(SequentialModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            user_id, time = self.data['user_id'][index], self.data['time'][index]
            history_item, history_time = feed_dict['all_history_items'], feed_dict['all_history_times']
            relational_interval = list()
            for i, target_item in enumerate(feed_dict['item_id']):
                interval = np.ones(5, dtype=float) * 0
                t = 0
                for j in range(len(history_item))[::-1]:
                    if history_item[j] == target_item:
                        interval[t] = (time - history_time[j])
                        t = t + 1
                        if t == 5:
                            break
                relational_interval.append(interval)
            feed_dict['relational_interval'] = np.array(relational_interval, dtype=np.float32)
            return feed_dict
