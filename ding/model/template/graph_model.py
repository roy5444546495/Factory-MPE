from typing import Union, List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from ding.utils import list_split, squeeze, MODEL_REGISTRY
from ding.torch_utils.network.nn_module import fc_block, MLP
from ding.torch_utils.network.transformer import ScaledDotProductAttention
from ding.torch_utils import to_tensor, tensor_to_list
from .q_learning import DRQN
import math 
import itertools
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, 
            input_dim, 
            hidden_dim, 
            output_dim
        ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x 
        
class GraphWeight(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim: int = 8, 
                 output_dim: int = 8
        ):
        super(GraphWeight, self).__init__()
        self._hidden_dim = hidden_dim
        self.q = GCN(input_dim, hidden_dim, output_dim)
        self.k = GCN(input_dim, hidden_dim, output_dim)

        return 

    def remove_diag(self, attention_map):
        T, B, A, _ = attention_map.shape
        mask = torch.ones(attention_map.shape, dtype=torch.bool, device=attention_map.device)
        for i in range(A):
            mask[:, :, i, i] = False
        attention_map = attention_map[mask]
        
        return attention_map.reshape(T, B, A, A-1)

    def forward(self, agent_state, edges):
        # >>> agent_state = (T, B, A, N)
        T, B, A, N = agent_state.shape
        query = self.q(agent_state, edges)
        key = self.k(agent_state, edges)
        key = key.reshape(T, B, self._hidden_dim, A)
        attention_map = torch.matmul(query, key) 
        attention_map /= math.sqrt(1)
        attention_map = F.softmax(attention_map, dim=-1)

        return self.remove_diag(attention_map)    

class GraphEncoder(nn.Module):
    def __init__(self):
        super(GraphEncoder, self).__init__()
        return 

    def forward(self, attention_map, agent_relation):
        T, B, A, _ = agent_relation.shape
        agent_relation = agent_relation.reshape(T, B, A, (A-1), 2)
        attention_map = attention_map.unsqueeze(0).unsqueeze(0).unsqueeze(-1) 
        weight_agent_relation = attention_map * agent_relation
        
        return weight_agent_relation.reshape(T, B, A, -1)

class Attention(nn.Module):
    def __init__(self, input_dim = 8, hidden_dim = 8):
        super(Attention, self).__init__()
        self._hidden_dim = hidden_dim
        self.q = fc_block(input_dim, hidden_dim)
        self.k = fc_block(input_dim, hidden_dim)

    def forward(self, agent_state):
        T, B, A, N = agent_state.shape
        query = self.q(agent_state)
        key = self.k(agent_state)
        key = key.reshape(T, B, self._hidden_dim, A)
        attention_map = torch.matmul(query, key)  # T, B, A, hidden_dim
        attention_map /= math.sqrt(1)
        attention_map = F.softmax(attention_map, dim=-1)
        return attention_map    # T, B, A, A

class NeighborhoodEncoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim: int = 8, 
                 output_dim: int = 8
        ):
        super(NeighborhoodEncoder, self).__init__()
        self._hidden_dim = hidden_dim
        self.encoder = GCN(input_dim, hidden_dim, output_dim)
        return 

    def forward(self, neigborhood_state, edges):
        neighborhood_emb = self.encoder(neigborhood_state, edges)
        neighborhood_emb = torch.mean(neighborhood_emb, dim=-2, keepdim=False)  
        
        return neighborhood_emb

class NeighborhoodWeight(nn.Module):
    def __init__(self):
        super(NeighborhoodWeight, self).__init__()
        return 

    def forward(self, attention_map, agent_relation):
        T, B, A, _ = agent_relation.shape
        agent_relation = agent_relation.reshape(T, B, A, (A-1), 2)
        attention_map = attention_map.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        weight_agent_relation = attention_map * agent_relation
        
        return weight_agent_relation.reshape(T, B, A, -1)

class JointQ(nn.Module):
    def __init__(self, q_input_size, embedding_size, hidden_dim):
        super(JointQ, self).__init__()
        self.Qact = nn.Sequential(
            nn.Linear(q_input_size, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size), nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )
        self.Qst =  nn.Linear(hidden_dim, 1)
        
        return 
    def forward(self, state, act):
        return self.Qst(state) + self.Qact(act)
    
@MODEL_REGISTRY.register('graph_model')
class GraphModel(nn.Module):
    """
    Overview:
        Graph network
    Interface:
        __init__, forward
    """

    def __init__(
            self,
            agent_num: int,
            obs_shape: int,
            global_obs_shape: int,
            action_shape: int,
            hidden_size_list: list,
            embedding_size: int,
            lstm_type: str = 'gru',
            dueling: bool = False, 
    ) -> None:
        super(GraphModel, self).__init__()
        region_emb_shape = 10
        self._agent_num = agent_num
        self._act = nn.ReLU()
        self._q_network = DRQN(obs_shape + (self._agent_num - 1) * 2 + region_emb_shape, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling)
        self._region_encoder = nn.Linear(obs_shape, region_emb_shape)
        q_input_size = global_obs_shape + hidden_size_list[-1] + action_shape
        self.Q = JointQ(q_input_size, embedding_size, hidden_dim = 8)
        self._transformer = Attention()
        self._graph_encoder = GraphEncoder()
        self._graph_weight = GraphWeight(obs_shape)
        self._neb_encoder = NeighborhoodEncoder(obs_shape)
        # V(s)
        self.V = nn.Sequential(
            nn.Linear(global_obs_shape, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size),
            nn.ReLU(), nn.Linear(embedding_size, 1)
        )
        ae_input = hidden_size_list[-1] + action_shape
        self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input), nn.ReLU(), nn.Linear(ae_input, ae_input))

    def compute_clip_loss(self, region_embs, labels, temperature=1):
        T, B, A, N = region_embs.shape
        region_embs = F.normalize(region_embs, p=2, dim=-1)
        sim_matrix = torch.matmul(region_embs, region_embs.transpose(-1, -2)) / temperature
        same_label = (labels.unsqueeze(-1) == labels.unsqueeze(-2)).float()
        eye_mask = torch.eye(A, device=region_embs.device).float().view(1, 1, A, A)
        
        exp_sim = torch.exp(sim_matrix) * (1 - eye_mask)
        row_sums = exp_sim.sum(dim=-1, keepdim=True)
        
        pos_mask = same_label * (1 - eye_mask)
        pos_loss_elements = -torch.log(exp_sim / (row_sums + 1e-8)) * pos_mask
        num_pos = pos_mask.sum(dim=(-1, -2))
        pos_loss = pos_loss_elements.sum(dim=(-1, -2)) / torch.clamp(num_pos, min=1)
        neg_mask = (1 - same_label) * (1 - eye_mask)
        neg_loss_elements = -torch.log(1 - exp_sim / (row_sums + 1e-8)) * neg_mask
        num_neg = neg_mask.sum(dim=(-1, -2))
        neg_loss = neg_loss_elements.sum(dim=(-1, -2)) / torch.clamp(num_neg, min=1)
        
        clip_loss = pos_loss + neg_loss
        return clip_loss
    
    def forward(self, data: dict, single_step: bool = True) -> dict:
        agent_state, global_state, prev_state = data['obs']['agent_state'], data['obs']['global_state'], data[
            'prev_state']     
        region_state =  data['obs']['region_state']
        region_label = data['obs']['region_label']
        action = data.get('action', None)
        if single_step:
            agent_state, global_state = agent_state.unsqueeze(0), global_state.unsqueeze(0)
            region_state = region_state.unsqueeze(0)
            region_label = region_label.unsqueeze(0)
        T, B, A = agent_state.shape[:3]
        assert len(prev_state) == B and all(
            [len(p) == A for p in prev_state]
        ), '{}-{}-{}-{}'.format([type(p) for p in prev_state], B, A, len(prev_state[0]))
        if len(agent_state.shape) == 3:
            agent_state = agent_state.unsqueeze(0)
        agent_relation = agent_state[:, :, :, 4: 4 + (self._agent_num-1) * 2]                 
        edges = torch.tensor(list(itertools.combinations(range(A), 2))).t().contiguous().to(agent_state.device)
        region_emb = self._region_encoder(agent_state)
        clip_loss = self.compute_clip_loss(region_emb, region_label)
        attention_map = self._graph_weight(agent_state, edges)
        message_state = self._graph_encoder(attention_map, agent_relation)               
        agent_state = torch.cat([agent_state, message_state, region_emb], dim = -1)                  
        prev_state = reduce(lambda x, y: x + y, prev_state)
        agent_state = agent_state.reshape(T, -1, *agent_state.shape[3:])                     
        output = self._q_network({'obs': agent_state, 'prev_state': prev_state, 'enable_fast_timestep': True})
        agent_q, next_state = output['logit'], output['next_state']     
        next_state, _ = list_split(next_state, step=A)
        agent_q = agent_q.reshape(T, B, A, -1)
        if action is None:
            # For target forward process
            if len(data['obs']['action_mask'].shape) == 3:
                action_mask = data['obs']['action_mask'].unsqueeze(0)
            else:
                action_mask = data['obs']['action_mask']
            agent_q[action_mask == 0.0] = -9999999
            action = agent_q.argmax(dim=-1)
        agent_q_act = torch.gather(agent_q, dim=-1, index=action.unsqueeze(-1))
        agent_q_act = agent_q_act.squeeze(-1)  # T, B, A

        hidden_states = output['hidden_state'].reshape(T * B, A, -1)
        action = action.reshape(T * B, A).unsqueeze(-1)
        action_onehot = torch.zeros(size=(T * B, A, agent_q.shape[-1]), device=action.device)
        action_onehot = action_onehot.scatter(2, action, 1)
        agent_state_action_input = torch.cat([hidden_states, action_onehot], dim=2)
        agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(T * B * A, -1)).reshape(T * B, A, -1)
        agent_state_action_encoding = agent_state_action_encoding.sum(dim=1)  # Sum across agents
        inputs = torch.cat([global_state.reshape(T * B, -1), agent_state_action_encoding], dim=1)
        neighborhood_emb = self._neb_encoder(region_state, edges)
        attention_map = self._transformer(neighborhood_emb)[:, :, :, 0].unsqueeze(2)
        weight_neighborhood_emb = torch.matmul(attention_map, neighborhood_emb).reshape(T, B, -1)
        q_outputs = self.Q(weight_neighborhood_emb.reshape(T * B, -1), inputs).reshape(T, B)
        v_outputs = self.V(global_state.reshape(T * B, -1))
        v_outputs = v_outputs.reshape(T, B)
        if single_step:
            q_outputs, agent_q, agent_q_act, v_outputs = q_outputs.squeeze(0), agent_q.squeeze(0), agent_q_act.squeeze(
                0
            ), v_outputs.squeeze(0)
            clip_loss = clip_loss.squeeze(0)
        return {
            'total_q': q_outputs,
            'logit': agent_q,
            'agent_q_act': agent_q_act,
            'vs': v_outputs,
            'next_state': next_state,
            'action_mask': data['obs']['action_mask'],
            'clip_loss': clip_loss
        }