from copy import deepcopy
from ding.entry import serial_pipeline
from easydict import EasyDict
from random import randrange
import datetime
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
seed = randrange(20000, 30000)
n_uav = 3
n_target = 8
n_entity = n_uav + n_target
num_landmarks = 3

collector_env_num = 4
evaluator_env_num = 5
experiment = False
main_config = dict(
    exp_name=f"3vs8_{time}_base",
    env=dict(
        max_step=100,
        n_uav=n_uav,
        n_target=n_target,
        num_landmarks=num_landmarks,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        n_evaluator_episode=5,
        stop_value=100,
        experiment=experiment
    ),
    policy=dict(
        experiment=experiment,
        model=dict(
            agent_num=n_uav,
            obs_shape=2 + 2 + (n_entity - 1) * 2 + num_landmarks * 2,
            global_obs_shape=n_entity * 2 + num_landmarks * 2 + n_entity * 2,
            action_shape=5,
            hidden_size_list=[128],
            embedding_size=64,
            lstm_type='gru',
            dueling=False,
        ),
        learn=dict(
            update_per_collect=100,
            batch_size=32,
            learning_rate=0.0005,
            double_q=True,
            target_update_theta=0.001,
            discount_factor=0.99,
            td_weight=1,
            opt_weight=0.1,
            nopt_min_weight=0.0001,
        ),
        collect=dict(
            n_sample=600,
            unroll_len=16,
            env_num=collector_env_num,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.0,
                end=0.05,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=15000,
                # (int) The maximum reuse times of each data
                max_reuse=1e+9,
                max_staleness=1e+9,
            ),
        ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        import_names=['dizoo.multiagent_particle.envs.particle_env'],
        type='modified_predator_prey',
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)

modified_predator_prey_qtran_config = main_config
modified_predator_prey_qtran_create_config = create_config


def train(args):
    config = [main_config, create_config]
    args.seed = seed
    serial_pipeline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    

    train(args)