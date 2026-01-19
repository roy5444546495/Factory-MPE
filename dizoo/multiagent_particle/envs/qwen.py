"""Qwen Agent."""
import re, json, ast
from typing import Dict
import dashscope
from http import HTTPStatus                        
                 
class QwenAgent:
    def __init__(
            self,
            api_key: str = "sk-4310960f79764d599d578513186b007b"         
        ):
        dashscope.api_key= api_key
        
    def match_pattern_json(self, text):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match is not None:
            try:
                output = json.loads(match.group(0))
            except:
                output = {}
        else:
            output = {}
        return output

    def match_pattern_jsonv2(self, s):
        dict_pattern = r'\{.*?\}'
        matches = re.findall(dict_pattern, s, re.DOTALL)
        if matches:
            try:
                for match in matches:
                    try:
                        return ast.literal_eval(match)
                    except (ValueError, SyntaxError):
                        continue
            except Exception as e:
                pass
        kv_pattern = r'(["\']?\w+["\']?)\s*[:=]\s*(["\']?.*?["\']?)(?=\s*[,}]|\s*$)'
        kv_pairs = re.findall(kv_pattern, s)
        if kv_pairs:
            result = {}
            for k, v in kv_pairs:
                k = k.strip('\'"')
                v = v.strip('\'"')
                result[k] = v
            return result
        
        return None

    def build_state_prompt(self, tensor, u_uav, u_target, num_landmarks):
        """
            tensor: torch.Tensor of shape [A, D]
            num_others: 其他智能体位置数量（根据实际环境配置）
            
            状态向量结构假设：
            [agent.state.p_pos] + [agent.state.p_vel] + other_pos(相对坐标) + other_vel + prey_pos + prey_vel + entity_pos
            
            返回：
            tuple: (智能体状态描述, 被巡逻机绝对位置描述)
        """
        descriptions = []
        prey_abs_positions = []  # 存储被巡逻机的绝对坐标
        entity_positions = []  # 存储环境实体的绝对坐标
        
        for agent_idx in range(tensor.shape[0]):
            agent_data = tensor[agent_idx]
            
            # 解析各部分数据
            p_pos = agent_data[0:2]  # 智能体绝对坐标
            p_vel = agent_data[2:4]
            other_relative_pos = agent_data[4: 4 + 2 * (u_uav + u_target - 1)].reshape(-1, 2)  # 其他单位的相对坐标

            # 构建智能体描述
            desc = f"智能体{agent_idx+1}:\n"
            desc += f"- 速度: ({p_vel[0]:.2f}, {p_vel[1]:.2f}) m/s\n"
            desc += f"- 当前位置(绝对坐标): ({p_pos[0]:.2f}, {p_pos[1]:.2f})\n"
            
            # 添加其他智能体位置（相对坐标）
            if len(other_relative_pos) > 0:
                desc += "- 观测到其他单位的相对位置:\n"
                for i, rel_pos in enumerate(other_relative_pos):
                    # 计算绝对坐标 = 自身位置 + 相对位置
                    abs_pos = (p_pos[0] + rel_pos[0], p_pos[1] + rel_pos[1])
                    
                    if i < u_uav - 1:
                        desc += f"  巡逻机队友: 相对位置({rel_pos[0]:.2f}, {rel_pos[1]:.2f}) \n"
                    else:
                        desc += f"  故障点: 相对位置({rel_pos[0]:.2f}, {rel_pos[1]:.2f}) \n"
                        
                        # 记录故障点的绝对坐标
                        if agent_idx == 0:
                            prey_abs_positions.append(f"故障点{len(prey_abs_positions)+1}: 绝对位置({abs_pos[0]:.2f}, {abs_pos[1]:.2f})")
            
            descriptions.append(desc.strip())
            
            # 记录landmarks的绝对坐标
            entity_pos = agent_data[-2 * num_landmarks:]
            if agent_idx == 0:
                for lm_idx in range(num_landmarks):
                    rel_lm_pos = entity_pos[2 * lm_idx: 2 * lm_idx + 2]
                    abs_lm_pos = (p_pos[0] + rel_lm_pos[0], p_pos[1] + rel_lm_pos[1])
                    entity_positions.append(f"障碍物{lm_idx + 1}: 绝对位置({abs_lm_pos[0]:.2f}, {abs_lm_pos[1]:.2f})")
        
        # 环境实体描述
        entity_desc = f"\n环境实体:\n" + "\n".join(entity_positions) if entity_positions else ""
        
        # 故障点绝对位置汇总描述
        prey_abs_desc = "\n故障点绝对位置汇总:\n" + "\n".join(prey_abs_positions) if prey_abs_positions else ""
        
        # 组合完整描述
        full_description = "\n\n".join(descriptions) + entity_desc
        
        return full_description, prey_abs_desc

    def build_region_prompt(self, agent_state, u_uav, u_target, num_landmarks, k = 2):
        """
            构建系统区域划分 prompt
        """
        # >>> obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + entity_pos)
        task_description = f"我们正在基于多智能体强化学习训练工厂中故障巡检的多UAV集群, 当前环境中有{u_uav}个UAV（由RL控制的智能体，目前感知范围内有{u_target}个故障点需要前往检测。"
        state_description, target_description = self.build_state_prompt(agent_state, u_uav, u_target, num_landmarks)
        output_description = f"""
            请结合任务的特点和当前系统的状态作出宏观决策，请预先将无人机集群划分为{k}个不同的子系统中，辅助决策，请输出一个Json字典。
            以下是一个合法的输出示例：
            {{
                "system_0": [0, 1, 2],
                "system_1": [3, 4, 5]
            }}

            - 其表示将0, 1, 2划分到子系统0，而无人机3, 4, 5划分到子系统1。
            - 输出时不需要作过多的解释。
        """
        
        prompt = f"""
                # 多智能体协作任务决策请求
                ## 任务描述
                    {task_description}
                ### 多智能体系统的UAV状态描述:
                    {state_description}
                ### 故障点状态描述:
                    {target_description}
                ## 输出要求
                    {output_description}
            """
        return prompt

    def get_region(self, agent_state, u_uav, u_target, num_landmarks, k):
        prompt = self.build_region_prompt(agent_state, u_uav, u_target, num_landmarks, k)
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
        
        # Get result
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages
        )
        if response.status_code == HTTPStatus.OK:
            text = response["output"]["choices"][0]["message"]["content"][0]["text"]
        else:
            text = 404
        # Transform into json
        if isinstance(text, str):
            try:
                return self.match_pattern_jsonv2(text)
            except:
                return 404