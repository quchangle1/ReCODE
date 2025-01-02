# -*- coding: UTF-8 -*-

""" NCF-ReCODE
CMD example:
    python main.py --model_name NCF_ReCODE --emb_size 32 --hidden_size 64 --steps 77 --method 'euler' \
    --layers '[64]' --lr 1e-4 --l2 1e-6 --dropout 0.2 --dataset "MMTD"
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

class NCF_ReCODE(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'hidden_size','method','steps','layers','time_scalar']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in GRU.')
        parser.add_argument('--method', type=str, default='euler',
                            help="the method for ODE")
        parser.add_argument('--steps', type=int, default=77,
                            help="the num of step for ODE")
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        parser.add_argument('--time_scalar', type=int, default=3600*24*7,
                            help='Time scalar for time intervals.')
        parser.add_argument('--step_size', type=int, default=10,
                            help='step_size for ODE')
        
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.layers = eval(args.layers)
        self.hidden_size = args.hidden_size
        self.method = args.method
        self.steps = args.steps
        self.time_scalar = args.time_scalar
        self.step_size = args.step_size
        
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
        self.model = ODEFunc(self.hidden_size).to(self.device)
        
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

        users = feed_dict["user_id"]
        items = feed_dict["item_id"]
        items_expand = items.reshape(items.shape[0] * items.shape[1])
        time_interval = torch.ceil(feed_dict["relational_interval"] / self.time_scalar).type(
            torch.LongTensor)
        time_interval = time_interval.reshape(-1, time_interval.shape[-1]).cuda()
        u_vector = self.mf_u_embeddings(users)
        i_vector = self.mf_i_embeddings(items_expand)
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

        users = users.unsqueeze(-1).repeat((1, items.shape[1]))

        mf_u_vectors = self.mf_u_embeddings(users)
        mf_i_vectors = self.mf_i_embeddings(items)
        mlp_u_vectors = self.mlp_u_embeddings(users)
        mlp_i_vectors = self.mlp_i_embeddings(items)

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
