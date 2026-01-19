import os
import os.path as osp
import json
import shutil
import sys
import time
import tempfile
from importlib import import_module
from typing import Optional, Tuple, NoReturn
import yaml
from easydict import EasyDict

from ding.utils import deep_merge_dicts
from ding.envs import get_env_cls, get_env_manager_cls, BaseEnvManager
from ding.policy import get_policy_cls
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, Coordinator, \
    AdvancedReplayBuffer, get_parallel_commander_cls, get_parallel_collector_cls, get_buffer_cls, \
    get_serial_collector_cls, MetricSerialEvaluator, BattleInteractionSerialEvaluator
from ding.reward_model import get_reward_model_cls
from .utils import parallel_transform, parallel_transform_slurm, parallel_transform_k8s, save_config_formatted


class Config(object):
    r"""
    Overview:
        Base class for config.
    Interface:
        __init__, file_to_dict
    Property:
        cfg_dict
    """

    def __init__(
            self,
            cfg_dict: Optional[dict] = None,
            cfg_text: Optional[str] = None,
            filename: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init method. Create config including dict type config and text type config.
        Arguments:
            - cfg_dict (:obj:`Optional[dict]`): dict type config
            - cfg_text (:obj:`Optional[str]`): text type config
            - filename (:obj:`Optional[str]`): config file name
        """
        if cfg_dict is None:
            cfg_dict = {}
        if not isinstance(cfg_dict, dict):
            raise TypeError("invalid type for cfg_dict: {}".format(type(cfg_dict)))
        self._cfg_dict = cfg_dict
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = '.'
        self._text = text
        self._filename = filename

    @staticmethod
    def file_to_dict(filename: str) -> 'Config':  # noqa
        """
        Overview:
            Read config file and create config.
        Arguments:
            - filename (:obj:`Optional[str]`): config file name.
        Returns:
            - cfg_dict (:obj:`Config`): config class
        """
        cfg_dict, cfg_text = Config._file_to_dict(filename)
        return Config(cfg_dict, cfg_text, filename=filename)

    @staticmethod
    def _file_to_dict(filename: str) -> Tuple[dict, str]:
        """
        Overview:
            Read config file and convert the config file to dict type config and text type config.
        Arguments:
            - filename (:obj:`Optional[str]`): config file name.
        Returns:
            - cfg_dict (:obj:`Optional[dict]`): dict type config
            - cfg_text (:obj:`Optional[str]`): text type config
        """
        filename = osp.abspath(osp.expanduser(filename))
        # TODO check exist
        # TODO check suffix
        ext_name = osp.splitext(filename)[-1]
        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=ext_name)
            temp_config_name = osp.basename(temp_config_file.name)
            temp_config_file.close()
            shutil.copyfile(filename, temp_config_file.name)

            temp_module_name = osp.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)
            # TODO validate py syntax
            module = import_module(temp_module_name)
            cfg_dict = {k: v for k, v in module.__dict__.items() if not k.startswith('_')}
            del sys.modules[temp_module_name]
            sys.path.pop(0)

        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        return cfg_dict, cfg_text

    @property
    def cfg_dict(self) -> dict:
        return self._cfg_dict


def read_config_yaml(path: str) -> EasyDict:
    """
    Overview:
        read configuration from path
    Arguments:
        - path (:obj:`str`): Path of source yaml
    Returns:
        - (:obj:`EasyDict`): Config data from this file with dict type
    """
    with open(path, "r") as f:
        config_ = yaml.safe_load(f)

    return EasyDict(config_)


def save_config_yaml(config_: dict, path: str) -> NoReturn:
    """
    Overview:
        save configuration to path
    Arguments:
        - config (:obj:`dict`): Config dict
        - path (:obj:`str`): Path of target yaml
    """
    config_string = json.dumps(config_)
    with open(path, "w") as f:
        yaml.safe_dump(json.loads(config_string), f)


def save_config_py(config_: dict, path: str) -> NoReturn:
    """
    Overview:
        save configuration to python file
    Arguments:
        - config (:obj:`dict`): Config dict
        - path (:obj:`str`): Path of target yaml
    """
    # config_string = json.dumps(config_, indent=4)
    config_string = str(config_)
    from yapf.yapflib.yapf_api import FormatCode
    config_string, _ = FormatCode(config_string)
    config_string = config_string.replace('inf,', 'float("inf"),')
    with open(path, "w") as f:
        f.write('exp_config = ' + config_string)
    # Qtran
    config_['policy']['type'] = 'qtran' + '_command'
    print("hello pig")
    
    version1 = False
    if version1:
        print("WGNOWNG")
        # Fixed scale related parameters
        config_['policy']['collect']['collector']['rate'] = float(config_['env']['n_target'] / 3.0)
        config_['policy']['collect']['collector']['lower'] = 0
        config_['policy']['collect']['collector']['upper'] = 30
        config_['policy']['eval']['evaluator']['lower'] = 0
        config_['policy']['eval']['evaluator']['upper'] = 30
        config_['policy']['eval']['evaluator']['rate'] = float(config_['env']['n_target'] / 3.0)
        # Tune scale related parameters
        if config_['env']['n_target'] <= 5:                                     # [4, 5]
            config_['policy']['collect']['collector']['alpha'] = 1e2
            config_['policy']['collect']['collector']['beta'] = 1e6
            config_['policy']['collect']['collector']['scale'] = 1.43
            if not config_['policy']['experiment']:
                config_['policy']['collect']['collector']['scale'] *= 0.95      # [6, 8]
        elif config_['env']['n_target'] <= 8:
            config_['policy']['collect']['collector']['alpha'] = 1.7e6
            config_['policy']['collect']['collector']['beta'] = 1e6
            config_['policy']['collect']['collector']['scale'] = 1.35 
            if not config_['policy']['experiment']:
                config_['policy']['collect']['collector']['scale'] *= 0.9
        elif config_['env']['n_target'] <= 11:                                  # [9]
            config_['policy']['collect']['collector']['alpha'] = 4.2e6         
            config_['policy']['collect']['collector']['beta'] = 1e6
            config_['policy']['collect']['collector']['scale'] = 1.20
            if not config_['policy']['experiment']:
                config_['policy']['collect']['collector']['scale'] *= 0.85
        elif config_['env']['n_target'] <= 15:                                  # [12] （base的模式崩塌）
            config_['policy']['collect']['collector']['alpha'] = 4.8e6         
            config_['policy']['collect']['collector']['beta'] = 1e6
            config_['policy']['collect']['collector']['scale'] = 1.15
            if not config_['policy']['experiment']:
                config_['policy']['collect']['collector']['scale'] *= 0.85      # [15] （alg的模式崩塌）
        else:
            config_['policy']['collect']['collector']['alpha'] = 8e6         
            config_['policy']['collect']['collector']['beta'] = 1e6
            config_['policy']['collect']['collector']['scale'] = 1.0
            if not config_['policy']['experiment']:
                config_['policy']['collect']['collector']['scale'] *= 0.75
    else:
        def configure_scale_parameters(config_):
            """
            配置与任务规模相关的参数
            
            参数说明：
            - alpha/beta: 控制奖励曲线的衰减速度（越大衰减越慢）
            - scale: 奖励缩放因子
            - 随着n_target增大，alpha增大使奖励曲线更平缓，scale减小使整体奖励降低
            - experiment1/experiment2: 两个独立的优化措施，未启用时会导致性能衰减
            """
            n_target = config_['env']['n_target']
            n_obstacle = config_['env']['num_landmarks']
            experiment1 = config_['policy'].get('experiment1', False)
            experiment2 = config_['policy'].get('experiment2', False)
            
            # 固定参数配置
            base_rate = float(n_target / 3.0)
            config_['policy']['collect']['collector'].update({
                'rate': base_rate,
                'lower': 0,
                'upper': 30
            })
            config_['policy']['eval']['evaluator'].update({
                'rate': base_rate,
                'lower': 0,
                'upper': 30
            })
            
            obstacle_thresholds = [(0, 1.0), (3, 0.85), (6, 0.6), (10, 0.2), (float('inf'), 0.80)]
            obstacle_factor = next(factor for threshold, factor in obstacle_thresholds if n_obstacle <= threshold)
            # 根据任务规模定义参数配置表（包含默认配置）
            # (max_targets, alpha, beta, base_scale, exp1_decay, exp2_decay, exp2_decay_alpha)
            scale_configs = [
                (5,  1e2,   1e6, 1.43 * obstacle_factor, 0.9, 0.96),  # 小规模[4-5]: 
                (8,  1.0e6, 1e6, 1.35 * obstacle_factor, 0.8, 0.92),  # 中小规模[6-8]:
                (11, 2.5e6, 1e6, 1.20 * obstacle_factor, 0.675,  0.86),  # 中等规模[9-11]: 
                (15, 3.5e6, 1e6, 1.15 * obstacle_factor, 0.675,  0.0),  # 中大规模[12-15]: 
                (float('inf'), 4e6, 1e6, 1.0 * obstacle_factor, 0.0, 0.0),  # 大规模[>15]: 
            ]

            # 查找适配的配置
            for max_targets, alpha, beta, base_scale, exp1_decay, exp2_decay in scale_configs:
                if n_target <= max_targets:
                    # 根据优化措施调整scale（未启用优化时应用衰减）
                    scale = base_scale
                    if not experiment1:
                        scale *= exp1_decay
                    if not experiment2:
                        scale *= exp2_decay
                    # 非experiment2训练速度还要慢
                    if not experiment2:
                        alpha += 1.0e6
                        
                    collector_config = {
                        'alpha': alpha,
                        'beta': beta,
                        'scale': scale
                    }
                    break
            
            config_['policy']['collect']['collector'].update(collector_config)
            
            return config_
        config_ = configure_scale_parameters(config_)


    # Refresh about environment
    config_['env']['n_ugv'] = 3
    config_['env']['n_target'] = 3    
    config_['env']['num_landmarks'] = 1    
    config_['env']['num_catch'] = 2 
    config_['env']['reward_right_catch'] = 10  
    config_['env']['reward_wrong_catch'] = -2 
    config_['env']['collision_ratio'] = 2    

    # About experiment parameters
    if not config_['policy']['experiment1'] or not config_['policy']['experiment2']:
        pass
    else:
        config_['policy']['learn']['discount_factor'] = 0.85
        config_['policy']['other']['eps']['end'] = 0.2
        config_['policy']['model']['embedding_size'] = 256
        config_['policy']['model']['hidden_size_list'] = [384, 384]

    # About model cfg
    config_['policy']['model']['agent_num'] = 3 
    config_['policy']['model']['obs_shape'] = 2 + 2 + (3 + 3 - 1) * 2 + 2 * config_['env']['num_landmarks']
    config_['policy']['model']['global_obs_shape'] = (3 + 3) * 2 + 2 * config_['env']['num_landmarks'] + (3 + 3) * 2

def read_config_directly(path: str) -> dict:
    """
    Overview:
        Read configuration from a file path(now only suport python file) and directly return results.
    Arguments:
        - path (:obj:`str`): Path of configuration file
    Returns:
        - cfg (:obj:`Tuple[dict, dict]`): Configuration dict.
    """
    suffix = path.split('.')[-1]
    if suffix == 'py':
        return Config.file_to_dict(path).cfg_dict
    else:
        raise KeyError("invalid config file suffix: {}".format(suffix))


def read_config(path: str) -> Tuple[dict, dict]:
    """
    Overview:
        Read configuration from a file path(now only suport python file). And select some proper parts.
    Arguments:
        - path (:obj:`str`): Path of configuration file
    Returns:
        - cfg (:obj:`Tuple[dict, dict]`): A collection(tuple) of configuration dict, divided into `main_config` and \
            `create_cfg` two parts.
    """
    suffix = path.split('.')[-1]
    if suffix == 'py':
        cfg = Config.file_to_dict(path).cfg_dict
        assert "main_config" in cfg, "Please make sure a 'main_config' variable is declared in config python file!"
        assert "create_config" in cfg, "Please make sure a 'create_config' variable is declared in config python file!"
        return cfg['main_config'], cfg['create_config']
    else:
        raise KeyError("invalid config file suffix: {}".format(suffix))


def read_config_with_system(path: str) -> Tuple[dict, dict, dict]:
    """
    Overview:
        Read configuration from a file path(now only suport python file). And select some proper parts
    Arguments:
        - path (:obj:`str`): Path of configuration file
    Returns:
        - cfg (:obj:`Tuple[dict, dict]`): A collection(tuple) of configuration dict, divided into `main_config`, \
            `create_cfg` and `system_config` three parts.
    """
    suffix = path.split('.')[-1]
    if suffix == 'py':
        cfg = Config.file_to_dict(path).cfg_dict
        assert "main_config" in cfg, "Please make sure a 'main_config' variable is declared in config python file!"
        assert "create_config" in cfg, "Please make sure a 'create_config' variable is declared in config python file!"
        assert "system_config" in cfg, "Please make sure a 'system_config' variable is declared in config python file!"
        return cfg['main_config'], cfg['create_config'], cfg['system_config']
    else:
        raise KeyError("invalid config file suffix: {}".format(suffix))


def save_config(config_: dict, path: str, type_: str = 'py', save_formatted: bool = False) -> NoReturn:
    """
    Overview:
        save configuration to python file or yaml file
    Arguments:
        - config (:obj:`dict`): Config dict
        - path (:obj:`str`): Path of target yaml or target python file
        - type (:obj:`str`): If type is ``yaml`` , save configuration to yaml file. If type is ``py`` , save\
             configuration to python file.
        - save_formatted (:obj:`bool`): If save_formatted is true, save formatted config to path.\
            Formatted config can be read by serial_pipeline directly.
    """
    assert type_ in ['yaml', 'py'], type_
    if type_ == 'yaml':
        save_config_yaml(config_, path)
    elif type_ == 'py':
        save_config_py(config_, path)
        if save_formatted:
            formated_path = osp.join(osp.dirname(path), 'formatted_' + osp.basename(path))
            save_config_formatted(config_, formated_path)
            save_config_json(config_, formated_path.replace('.py', '.json'))
            
def save_config_json(data_dict, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)
    except Exception as e:
        pass

def compile_buffer_config(policy_cfg: EasyDict, user_cfg: EasyDict, buffer_cls: 'IBuffer') -> EasyDict:  # noqa

    def _compile_buffer_config(policy_buffer_cfg, user_buffer_cfg, buffer_cls):

        if buffer_cls is None:
            assert 'type' in policy_buffer_cfg, "please indicate buffer type in create_cfg"
            buffer_cls = get_buffer_cls(policy_buffer_cfg)
        buffer_cfg = deep_merge_dicts(buffer_cls.default_config(), policy_buffer_cfg)
        buffer_cfg = deep_merge_dicts(buffer_cfg, user_buffer_cfg)
        return buffer_cfg

    policy_multi_buffer = policy_cfg.other.replay_buffer.get('multi_buffer', False)
    user_multi_buffer = user_cfg.policy.get('other', {}).get('replay_buffer', {}).get('multi_buffer', False)
    assert not user_multi_buffer or user_multi_buffer == policy_multi_buffer, "For multi_buffer, \
        user_cfg({}) and policy_cfg({}) must be in accordance".format(user_multi_buffer, policy_multi_buffer)
    multi_buffer = policy_multi_buffer
    if not multi_buffer:
        policy_buffer_cfg = policy_cfg.other.replay_buffer
        user_buffer_cfg = user_cfg.policy.get('other', {}).get('replay_buffer', {})
        return _compile_buffer_config(policy_buffer_cfg, user_buffer_cfg, buffer_cls)
    else:
        return_cfg = EasyDict()
        for buffer_name in policy_cfg.other.replay_buffer:  # Only traverse keys in policy_cfg
            if buffer_name == 'multi_buffer':
                continue
            policy_buffer_cfg = policy_cfg.other.replay_buffer[buffer_name]
            user_buffer_cfg = user_cfg.policy.get('other', {}).get('replay_buffer', {}).get('buffer_name', {})
            return_cfg[buffer_name] = _compile_buffer_config(
                policy_buffer_cfg, user_buffer_cfg, buffer_cls[buffer_name]
            )
            return_cfg[buffer_name].name = buffer_name
        return return_cfg


def compile_collector_config(
        policy_cfg: EasyDict,
        user_cfg: EasyDict,
        collector_cls: 'ISerialCollector'  # noqa
) -> EasyDict:
    policy_collector_cfg = policy_cfg.collect.collector
    user_collector_cfg = user_cfg.policy.get('collect', {}).get('collector', {})
    # step1: get collector class
    # two cases: create cfg merged in policy_cfg, collector class, and class has higher priority
    if collector_cls is None:
        assert 'type' in policy_collector_cfg, "please indicate collector type in create_cfg"
        # use type to get collector_cls
        collector_cls = get_serial_collector_cls(policy_collector_cfg)
    # step2: policy collector cfg merge to collector cfg
    collector_cfg = deep_merge_dicts(collector_cls.default_config(), policy_collector_cfg)
    # step3: user collector cfg merge to the step2 config
    collector_cfg = deep_merge_dicts(collector_cfg, user_collector_cfg)

    return collector_cfg


policy_config_template = dict(
    model=dict(),
    learn=dict(learner=dict()),
    collect=dict(collector=dict()),
    eval=dict(evaluator=dict()),
    other=dict(replay_buffer=dict()),
)
policy_config_template = EasyDict(policy_config_template)
env_config_template = dict(manager=dict(), )
env_config_template = EasyDict(env_config_template)


def compile_config(
        cfg: EasyDict,
        env_manager: type = None,
        policy: type = None,
        learner: type = BaseLearner,
        collector: type = None,
        evaluator: type = InteractionSerialEvaluator,
        buffer: type = AdvancedReplayBuffer,
        env: type = None,
        reward_model: type = None,
        seed: int = 0,
        auto: bool = False,
        create_cfg: dict = None,
        save_cfg: bool = True,
        save_path: str = 'total_config.py',
) -> EasyDict:
    """
    Overview:
        Combine the input config information with other input information.
        Compile config to make it easy to be called by other programs
    Arguments:
        - cfg (:obj:`EasyDict`): Input config dict which is to be used in the following pipeline
        - env_manager (:obj:`type`): Env_manager class which is to be used in the following pipeline
        - policy (:obj:`type`): Policy class which is to be used in the following pipeline
        - learner (:obj:`type`): Input learner class, defaults to BaseLearner
        - collector (:obj:`type`): Input collector class, defaults to BaseSerialCollector
        - evaluator (:obj:`type`): Input evaluator class, defaults to InteractionSerialEvaluator
        - buffer (:obj:`type`): Input buffer class, defaults to BufferManager
        - env (:obj:`type`): Environment class which is to be used in the following pipeline
        - reward_model (:obj:`type`): Reward model class which aims to offer various and valuable reward
        - seed (:obj:`int`): Random number seed
        - auto (:obj:`bool`): Compile create_config dict or not
        - create_cfg (:obj:`dict`): Input create config dict
        - save_cfg (:obj:`bool`): Save config or not
        - save_path (:obj:`str`): Path of saving file
    Returns:
        - cfg (:obj:`EasyDict`): Config after compiling
    """
    if auto:
        assert create_cfg is not None
        # for compatibility
        if 'collector' not in create_cfg:
            create_cfg.collector = EasyDict(dict(type='sample'))
        if 'replay_buffer' not in create_cfg:
            create_cfg.replay_buffer = EasyDict(dict(type='advanced'))
        if env is None:
            env = get_env_cls(create_cfg.env)
        if env_manager is None:
            env_manager = get_env_manager_cls(create_cfg.env_manager)
        if policy is None:
            policy = get_policy_cls(create_cfg.policy)
        if 'default_config' in dir(env):
            env_config = env.default_config()
        else:
            env_config = EasyDict()  # env does not have default_config
        env_config = deep_merge_dicts(env_config_template, env_config)
        env_config.update(create_cfg.env)
        env_config.manager = deep_merge_dicts(env_manager.default_config(), env_config.manager)
        env_config.manager.update(create_cfg.env_manager)
        policy_config = policy.default_config()
        policy_config = deep_merge_dicts(policy_config_template, policy_config)
        policy_config.update(create_cfg.policy)
        policy_config.collect.collector.update(create_cfg.collector)
        policy_config.other.replay_buffer.update(create_cfg.replay_buffer)

        policy_config.other.commander = BaseSerialCommander.default_config()
        if 'reward_model' in create_cfg:
            reward_model = get_reward_model_cls(create_cfg.reward_model)
            reward_model_config = reward_model.default_config()
        else:
            reward_model_config = EasyDict()
    else:
        if 'default_config' in dir(env):
            env_config = env.default_config()
        else:
            env_config = EasyDict()  # env does not have default_config
        env_config = deep_merge_dicts(env_config_template, env_config)
        if env_manager is None:
            env_manager = BaseEnvManager  # for compatibility
        env_config.manager = deep_merge_dicts(env_manager.default_config(), env_config.manager)
        policy_config = policy.default_config()
        policy_config = deep_merge_dicts(policy_config_template, policy_config)
        if reward_model is None:
            reward_model_config = EasyDict()
        else:
            reward_model_config = reward_model.default_config()
    
    policy_config.learn.learner = deep_merge_dicts(
        learner.default_config(),
        policy_config.learn.learner,
    )
    if create_cfg is not None or collector is not None:
        policy_config.collect.collector = compile_collector_config(policy_config, cfg, collector)
    policy_config.eval.evaluator = deep_merge_dicts(
        evaluator.default_config(),
        policy_config.eval.evaluator,
    )
    if create_cfg is not None or buffer is not None:
        policy_config.other.replay_buffer = compile_buffer_config(policy_config, cfg, buffer)
    default_config = EasyDict({'env': env_config, 'policy': policy_config})
    if len(reward_model_config) > 0:
        default_config['reward_model'] = reward_model_config
    cfg = deep_merge_dicts(default_config, cfg)
    cfg.seed = seed
    # check important key in config
    if evaluator in [InteractionSerialEvaluator, BattleInteractionSerialEvaluator]:  # env interaction evaluation
        assert all([k in cfg.env for k in ['n_evaluator_episode', 'stop_value']]), cfg.env
        cfg.policy.eval.evaluator.stop_value = cfg.env.stop_value
        cfg.policy.eval.evaluator.n_episode = cfg.env.n_evaluator_episode
    if 'exp_name' not in cfg:
        cfg.exp_name = 'default_experiment'
    if save_cfg:
        if not os.path.exists(cfg.exp_name):
            try:
                os.mkdir(cfg.exp_name)
            except FileExistsError:
                pass
        save_path = os.path.join(cfg.exp_name, save_path)
        save_config(cfg, save_path, save_formatted=True)

    return cfg


def compile_config_parallel(
        cfg: EasyDict,
        create_cfg: EasyDict,
        system_cfg: EasyDict,
        seed: int = 0,
        save_cfg: bool = True,
        save_path: str = 'total_config.py',
        platform: str = 'local',
        coordinator_host: Optional[str] = None,
        learner_host: Optional[str] = None,
        collector_host: Optional[str] = None,
        coordinator_port: Optional[int] = None,
        learner_port: Optional[int] = None,
        collector_port: Optional[int] = None,
) -> EasyDict:
    """
    Overview:
        Combine the input parallel mode configuration information with other input information. Compile config\
             to make it easy to be called by other programs
    Arguments:
        - cfg (:obj:`EasyDict`): Input main config dict
        - create_cfg (:obj:`dict`): Input create config dict, including type parameters, such as environment type
        - system_cfg (:obj:`dict`): Input system config dict, including system parameters, such as file path,\
            communication mode, use multiple GPUs or not
        - seed (:obj:`int`): Random number seed
        - save_cfg (:obj:`bool`): Save config or not
        - save_path (:obj:`str`): Path of saving file
        - platform (:obj:`str`): Where to run the program, 'local' or 'slurm'
        - coordinator_host (:obj:`Optional[str]`): Input coordinator's host when platform is slurm
        - learner_host (:obj:`Optional[str]`): Input learner's host when platform is slurm
        - collector_host (:obj:`Optional[str]`): Input collector's host when platform is slurm
    Returns:
        - cfg (:obj:`EasyDict`): Config after compiling
    """
    # for compatibility
    if 'replay_buffer' not in create_cfg:
        create_cfg.replay_buffer = EasyDict(dict(type='advanced'))
    # env
    env = get_env_cls(create_cfg.env)
    if 'default_config' in dir(env):
        env_config = env.default_config()
    else:
        env_config = EasyDict()  # env does not have default_config
    env_config = deep_merge_dicts(env_config_template, env_config)
    env_config.update(create_cfg.env)

    env_manager = get_env_manager_cls(create_cfg.env_manager)
    env_config.manager = env_manager.default_config()
    env_config.manager.update(create_cfg.env_manager)

    # policy
    policy = get_policy_cls(create_cfg.policy)
    policy_config = policy.default_config()
    policy_config = deep_merge_dicts(policy_config_template, policy_config)
    cfg.policy.update(create_cfg.policy)

    collector = get_parallel_collector_cls(create_cfg.collector)
    policy_config.collect.collector = collector.default_config()
    policy_config.collect.collector.update(create_cfg.collector)
    policy_config.learn.learner = BaseLearner.default_config()
    policy_config.learn.learner.update(create_cfg.learner)
    commander = get_parallel_commander_cls(create_cfg.commander)
    policy_config.other.commander = commander.default_config()
    policy_config.other.commander.update(create_cfg.commander)
    policy_config.other.replay_buffer.update(create_cfg.replay_buffer)
    policy_config.other.replay_buffer = compile_buffer_config(policy_config, cfg, None)

    default_config = EasyDict({'env': env_config, 'policy': policy_config})
    cfg = deep_merge_dicts(default_config, cfg)

    cfg.policy.other.commander.path_policy = system_cfg.path_policy  # league may use 'path_policy'

    # system
    for k in ['comm_learner', 'comm_collector']:
        system_cfg[k] = create_cfg[k]
    if platform == 'local':
        cfg = parallel_transform(EasyDict({'main': cfg, 'system': system_cfg}))
    elif platform == 'slurm':
        cfg = parallel_transform_slurm(
            EasyDict({
                'main': cfg,
                'system': system_cfg
            }), coordinator_host, learner_host, collector_host
        )
    elif platform == 'k8s':
        cfg = parallel_transform_k8s(
            EasyDict({
                'main': cfg,
                'system': system_cfg
            }),
            coordinator_port=coordinator_port,
            learner_port=learner_port,
            collector_port=collector_port
        )
    else:
        raise KeyError("not support platform type: {}".format(platform))
    cfg.system.coordinator = deep_merge_dicts(Coordinator.default_config(), cfg.system.coordinator)
    # seed
    cfg.seed = seed

    if save_cfg:
        save_config(cfg, save_path)
    return cfg
