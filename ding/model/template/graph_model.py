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
from dizoo.multiagent_particle.envs.qwen import QwenAgent
from dizoo.multiagent_particle.envs.doubao import DoubaoAgent
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
        self._agent =  DoubaoAgent()
        self._total_forward_count = 0
        self._agent_num = agent_num
        self._act = nn.ReLU()
        self._q_network = DRQN(obs_shape + (self._agent_num - 1) * 2, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling)
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

    def compute_klloss(self, p, q):
        """
        计算两个张量之间的 KL 散度。
        
        参数:
            p: 形状为 (Batch, agent_num) 的张量，表示概率分布。
            q: 形状为 (Batch, agent_num) 的张量，表示概率分布。
        
        返回:
            KL 散度值，形状为 (Batch,)
        """
        # 确保输入是概率分布（每行的和为 1）
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        
        # 计算 KL 散度
        klloss = F.kl_div(p.log(), q, reduction='none').sum(dim=-1)
        return klloss
    
    def forward(self, data: dict, single_step: bool = True) -> dict:
        self._total_forward_count += 1
        agent_state, global_state, prev_state = data['obs']['agent_state'], data['obs']['global_state'], data['prev_state']     
        neighborhood_state =  data['obs']['neighborhood_state']
        # reward_coef =  data['obs']['rew_coef']
        

        
        action = data.get('action', None)
        if single_step:
            agent_state, global_state = agent_state.unsqueeze(0), global_state.unsqueeze(0)
            neighborhood_state = neighborhood_state.unsqueeze(0)
            # reward_coef = reward_coef.unsqueeze(0)

            if True:
                # Usable info
                n_agent = data['obs']['n_agent']
                n_target = data['obs']['n_target']
                num_landmarks = data['obs']['num_landmarks']
                timestep = data['obs']['timestep']
                llm_exploration = data['obs']['llm_exploration']
                timestep = timestep[0].cpu().numpy()
                num_landmarks = num_landmarks[0].cpu().numpy()
                n_agent = n_agent[0].cpu().numpy()
                n_target = n_target[0].cpu().numpy()
                llm_exploration = llm_exploration[0].cpu().numpy()
                # print(f"Before calling get_teacher_policy: agent_state.shape = {agent_state.shape}")
                teacher_policy = self._agent.get_teacher_policy(agent_state, self._total_forward_count, timestep, n_agent, n_target, num_landmarks)

        T, B, A = agent_state.shape[:3]
        # print(f"Prev = {prev_state}")
        assert len(prev_state) == B and all(
            [len(p) == A for p in prev_state]
        ), '{}-{}-{}-{}'.format([type(p) for p in prev_state], B, A, len(prev_state[0]))
        if len(agent_state.shape) == 3:
            agent_state = agent_state.unsqueeze(0)
        agent_relation = agent_state[:, :, :, 4: 4 + (self._agent_num-1) * 2]                 
        edges = torch.tensor(list(itertools.combinations(range(A), 2))).t().contiguous().to(agent_state.device)
        attention_map = self._graph_weight(agent_state, edges)
        message_state = self._graph_encoder(attention_map, agent_relation)               
        agent_state = torch.cat([agent_state, message_state], dim = -1)                  
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
        agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(T * B * A,
                                                                                            -1)).reshape(T * B, A, -1)
        agent_state_action_encoding = agent_state_action_encoding.sum(dim=1)  # Sum across agents
        inputs = torch.cat([global_state.reshape(T * B, -1), agent_state_action_encoding], dim=1)
        neighborhood_emb = self._neb_encoder(neighborhood_state, edges)
        attention_map = self._transformer(neighborhood_emb)[:, :, :, 0].unsqueeze(2)
        weight_neighborhood_emb = torch.matmul(attention_map, neighborhood_emb).reshape(T, B, -1)
        q_outputs = self.Q(weight_neighborhood_emb.reshape(T * B, -1), inputs).reshape(T, B)
        v_outputs = self.V(global_state.reshape(T * B, -1))
        v_outputs = v_outputs.reshape(T, B)
        if single_step:
            q_outputs, agent_q, agent_q_act, v_outputs = q_outputs.squeeze(0), agent_q.squeeze(0), agent_q_act.squeeze(0), v_outputs.squeeze(0)
        
        result = {
            'total_q': q_outputs,
            'logit': agent_q,
            'agent_q_act': agent_q_act,
            'vs': v_outputs,
            'next_state': next_state,
            'action_mask': data['obs']['action_mask'],
        }
        if single_step:
            if llm_exploration:
                result.update({'teacher_policy': teacher_policy})

        return result