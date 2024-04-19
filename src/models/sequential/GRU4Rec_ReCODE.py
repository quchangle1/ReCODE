# -*- coding: UTF-8 -*-
""" GRU4Rec-ReCODE
CMD example:
python main.py --model_name GRU4Rec_ReCODE --emb_size 32 --hidden_size 64 --steps 77 --method 'euler' \
    --history_max 20 --lr 1e-4 --l2 1e-6 --dataset "MMTD"

"""
import torch
import torch.nn as nn
import torch.distributions
import numpy as np

from torchdiffeq import odeint_adjoint as odeint
from models.BaseModel import SequentialModel

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

class GRU4Rec_ReCODE(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size','hidden_size','steps','method']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in GRU.')
        parser.add_argument('--method', type=str, default='euler',
                            help="the method for ODE")
        parser.add_argument('--steps', type=int, default=51,
                            help="the num of step for ODE")
        parser.add_argument('--time_scalar', type=int, default=3600*24*7,
                            help='Time scalar for time intervals.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.hidden_size=args.hidden_size
        self.method=args.method
        self.steps=args.steps
        self.time_scalar = args.time_scalar

        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

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

        self.rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []

        items = feed_dict['item_id']
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

        pred_vectors = self.i_embeddings(items)
        rnn_vector = self.out(rnn_vector)
        base_intensity = (rnn_vector[:, None, :] * pred_vectors).sum(-1)

        items_expand = items.reshape(items.shape[0] * items.shape[1])
        time_interval = torch.ceil(feed_dict["relational_interval"] /self.time_scalar).type(
            torch.LongTensor)
        time_interval = time_interval.reshape(-1, time_interval.shape[-1]).cuda()
        u_vector = rnn_vector
        i_vector = self.i_embeddings(items_expand)
        u_vector = torch.cat([u_vector] * items.shape[1], dim=0)
        init_state = [u_vector, i_vector]
        init_state = torch.cat(init_state, dim=-1)

        init = self.en(init_state)
        t = torch.linspace(0, self.steps - 1, self.steps).to(self.device)
        model = ODEFunc(self.hidden_size).to(self.device)
        outputs = odeint(model, init, t, method=self.method,options={"perturb": "True", "step_size": self.steps})
        outputs = outputs.transpose(0, 1)

        outputs = torch.squeeze(self.o_net(outputs))
        p = torch.zeros(outputs.shape[0]).cuda()
        p = p.unsqueeze(1)
        outputs = torch.cat((p, outputs), 1)
        score = torch.gather(outputs, dim=1, index=time_interval)
        excit1 = score.reshape(items.shape[0], items.shape[1], -1)
        excitation = (excit1).sum(-1)

        prediction = base_intensity + excitation
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

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
