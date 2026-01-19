exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'step_timeout': 60,
            'auto_reset': True,
            'reset_timeout': 60,
            'retry_waiting_time': 0.1,
            'cfg_type': 'BaseEnvManagerDict',
            'type': 'base',
            'shared_memory': False
        },
        'import_names': ['dizoo.multiagent_particle.envs.particle_env'],
        'type': 'modified_predator_prey',
        'max_step': 100,
        'n_uav': 4,
        'n_target': 10,
        'num_landmarks': 5,
        'collector_env_num': 4,
        'evaluator_env_num': 5,
        'n_evaluator_episode': 5,
        'stop_value': 100
    },
    'policy': {
        'model': {
            'agent_num': 4,
            'obs_shape': 40,
            'global_obs_shape': 66,
            'action_shape': 5,
            'hidden_size_list': [128],
            'embedding_size': 64,
            'lstm_type': 'gru',
            'dueling': False
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'multi_gpu': False,
            'epoch_per_collect': 10,
            'batch_size': 32,
            'learning_rate': 0.0005,
            'value_weight': 0.5,
            'entropy_weight': 0.0,
            'clip_ratio': 0.2,
            'adv_norm': True,
            'value_norm': True,
            'ppo_param_init': True,
            'grad_clip_type': 'clip_norm',
            'grad_clip_value': 0.5,
            'ignore_done': False,
            'update_per_collect': 100,
            'double_q': True,
            'target_update_theta': 0.001,
            'discount_factor': 0.99,
            'td_weight': 1,
            'opt_weight': 0.1,
            'nopt_min_weight': 0.0001
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            },
            'unroll_len': 16,
            'discount_factor': 0.99,
            'gae_lambda': 0.95,
            'n_sample': 600,
            'env_num': 4
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1000,
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'stop_value': 100,
                'n_episode': 5
            },
            'env_num': 5
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 15000,
                'max_use': float("inf"),
                'max_staleness': 1000000000.0,
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict',
                'max_reuse': 1000000000.0
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            },
            'eps': {
                'type': 'exp',
                'start': 1.0,
                'end': 0.05,
                'decay': 100000
            }
        },
        'type': 'ppo_command',
        'cuda': False,
        'on_policy': True,
        'priority': False,
        'priority_IS_weight': False,
        'recompute_adv': True,
        'continuous': True,
        'multi_agent': False,
        'transition_with_policy_data': True,
        'cfg_type': 'PPOCommandModePolicyDict',
        'experiment': False
    },
    'exp_name': '4vs10_0724_alg',
    'seed': 0
}
