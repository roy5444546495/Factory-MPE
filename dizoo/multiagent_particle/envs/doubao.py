"""Doubao Agent."""
import torch
from typing import Dict
from http import HTTPStatus                        

class DoubaoAgent:
    # 每个Episode前T步允许蒸馏
    __MAX_TEACH_STEP = -1
    
    # 最多允许教师模型蒸馏的K次参数
    __MAX_FORWARD_COUNT = -1
    def __init__(
            self,
        ):
        pass 

    def tensor_to_description(self, tensor, n_agent, n_target, num_landmarks):
        """
        tensor: torch.Tensor of shape [A, D]
        n_agent: 己方控制的无人车智能体数量
        n_target: 目标点数量
        num_landmarks: 障碍物数量
        
        状态向量结构假设：
        [agent.state.p_pos] + [agent.state.p_vel] + other_pos(相对坐标) + other_vel + target_pos + target_vel + entity_pos
        
        返回：
        tuple: (智能体状态描述, 目标点绝对位置描述)
        """
        descriptions = []
        target_abs_positions = []  # 存储目标点的绝对坐标
        entity_positions = []  # 存储障碍物的绝对坐标
        
        for agent_idx in range(tensor.shape[0]):
            agent_data = tensor[agent_idx].cpu().numpy()
            
            # 解析各部分数据
            p_pos = agent_data[0:2]  # 智能体绝对坐标
            p_vel = agent_data[2:4]
            other_relative_pos = agent_data[4: 4 + 2 * (n_agent + n_target - 1)].reshape(-1, 2)  # 其他单位的相对坐标

            # 构建智能体描述
            desc = f"无人车{agent_idx+1}:\n"
            desc += f"- 速度: ({p_vel[0]:.2f}, {p_vel[1]:.2f}) m/s\n"
            desc += f"- 当前位置(绝对坐标): ({p_pos[0]:.2f}, {p_pos[1]:.2f})\n"
            
            # 添加其他单位位置（相对坐标）
            if len(other_relative_pos) > 0:
                desc += "- 观测到其他单位的相对位置:\n"
                for i, rel_pos in enumerate(other_relative_pos):
                    # 计算绝对坐标 = 自身位置 + 相对位置
                    abs_pos = (p_pos[0] + rel_pos[0], p_pos[1] + rel_pos[1])
                    
                    if i < n_agent - 1:
                        desc += f"  队友无人车: 相对位置({rel_pos[0]:.2f}, {rel_pos[1]:.2f})\n"
                    else:
                        desc += f"  目标点: 相对位置({rel_pos[0]:.2f}, {rel_pos[1]:.2f})\n"
                        
                        # 记录目标点的绝对坐标
                        if agent_idx == 0:
                            target_abs_positions.append(f"目标点{len(target_abs_positions)+1}: 绝对位置({abs_pos[0]:.2f}, {abs_pos[1]:.2f})")
            
            descriptions.append(desc.strip())
            
            # 记录障碍物的绝对坐标
            entity_pos = agent_data[-2 * num_landmarks:]
            if agent_idx == 0:
                for lm_idx in range(num_landmarks):
                    rel_lm_pos = entity_pos[2 * lm_idx: 2 * lm_idx + 2]
                    abs_lm_pos = (p_pos[0] + rel_lm_pos[0], p_pos[1] + rel_lm_pos[1])
                    entity_positions.append(f"障碍物{lm_idx + 1}: 绝对位置({abs_lm_pos[0]:.2f}, {abs_lm_pos[1]:.2f})")
        
        # 障碍物描述
        entity_desc = f"\n环境障碍物:\n" + "\n".join(entity_positions) if entity_positions else ""
        
        # 目标点绝对位置汇总描述
        target_abs_desc = "\n目标点绝对位置汇总:\n" + "\n".join(target_abs_positions) if target_abs_positions else ""
        
        # 组合完整描述
        full_description = "\n\n".join(descriptions) + entity_desc
        
        return full_description, target_abs_desc
    
    def get_teacher_policy_call(self, agent_state, n_agent, n_target, num_landmarks):
        n_agent, n_target, num_landmarks = int(n_agent), int(n_target), int(num_landmarks)
        task_description = f"我们正在训练一个多智能体协同导航系统，环境中有{n_agent}个无人车智能体（由RL控制）需要协作导航到{n_target}个目标点，导航过程需要注意避障和团队协调"
        
        action_space_description = """
            每个无人车智能体可执行以下5种基础移动动作：
            1. [不动]：保持当前位置不变
            2. [左移]：沿x轴负方向移动1个单位
            3. [右移]：沿x轴正方向移动1个单位 
            4. [下移]：沿y轴负方向移动1个单位
            5. [上移]：沿y轴正方向移动1个单位
        """
        
        state_description, target_description = self.tensor_to_description(agent_state, n_agent, n_target, num_landmarks)
        
        if True:
            prompt = f"""
                # 多智能体协同导航任务决策请求
                
                ## 任务描述
                {task_description}
                
                ## 动作空间描述
                {action_space_description}

                ## 多智能体系统的当前状态
                {state_description}
                
                ## 目标点位置信息
                {target_description}

                ## 决策需求
                请根据以下约束生成RL所控制的{n_agent}个无人车智能体的动作概率分布：
                1. 高效导航：引导无人车快速到达目标点
                2. 避障安全：避免与环境障碍物发生碰撞
                3. 团队协作：多车之间相互协调，避免拥堵和碰撞
                4. 概率约束：每个智能体的动作概率分布和为1
                
                ## 输出格式要求
                首先简要说明决策逻辑，然后按照以下JSON格式输出智能体的动作概率分布（确保JSON格式可被正确解析）：
                {{
                    "agent1": [prob1, prob2, prob3, prob4, prob5],
                    "agent2": [prob1, prob2, prob3, prob4, prob5],
                    ...
                }}
                
                其中，5个概率值依次对应：[不动, 左移, 右移, 下移, 上移]
                
                ## 错误示范（会导致JSON解析失败）
                ❌ 错误1：value使用字典而非列表
                {{"agent1": {{"action": "left", "probability": 0.5}}}}
                
                ❌ 错误2：JSON中包含注释
                {{
                    "agent1": [0.2, 0.2, 0.2, 0.2, 0.2], // 随机分配
                    "agent2": [0.1, 0.3, 0.3, 0.2, 0.1]
                }}
                
                ❌ 错误3：概率和不为1
                {{"agent1": [0.1, 0.2, 0.3, 0.3, 0.3]}}
                
                ## 正确示范
                ✅ 决策逻辑：agent1向右靠近目标点，agent2保持观望
                {{
                    "agent1": [0.0, 0.0, 0.9, 0.0, 0.1],
                    "agent2": [0.8, 0.05, 0.05, 0.05, 0.05]
                }}
            """
 
        from volcenginesdkarkruntime import Ark
        client = Ark(
            api_key="318a3c6b-2626-4a4d-9f8b-81ce7c88da41"
        )

        # 请求1
        completion = client.chat.completions.create(
            model="doubao-1.5-lite-32k-250115",
            messages=[
                {"role": "system", "content": "你是一个模型助手"},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content
          
    def get_teacher_policy(self, agent_state, total_forward_count, timestep, n_agent, n_target, num_landmarks):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent_state = agent_state.squeeze(0).squeeze(0)
        
        A, D = agent_state.shape
        random_tensor = torch.distributions.Dirichlet(torch.ones(5, device=device))
        random_tensor = random_tensor.sample((1, A))                                 # Creates [1, A, 5] shaped tensor
        if timestep > self.__MAX_TEACH_STEP or total_forward_count > self.__MAX_FORWARD_COUNT:
            return random_tensor 
       
        n_agent, n_target, num_landmarks = int(n_agent), int(n_target), int(num_landmarks)
        output = self.get_teacher_policy_call(agent_state, n_agent, n_target, num_landmarks) 
        # print("output", output)
        if type(output) == str:
            try:
                data = self.match_pattern_json(output)
                if len(output) > 0:
                        policy_list = [data[f"agent{i + 1}"] for i in range(A)]
                        policy_tensor = torch.tensor(policy_list, dtype=torch.float32)  # shape (3, 5)
                        policy_tensor = policy_tensor.unsqueeze(0)  
                        policy_tensor = policy_tensor.to(device)
                        assert policy_tensor.shape == (1, n_agent, 5)
                        
                        return policy_tensor
            except:
                return random_tensor
                
        return random_tensor
    
    def match_pattern_json(self, text):
        import re, json
        # 使用正则表达式匹配大括号内的内容
        # re.DOTALL 使得 . 匹配任意字符，包括换行符
        matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        x = json.loads(matches[0])
        return json.loads(matches[0])
    
    def save2txt(self, data: str, file_name: str):
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(data)