import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from einops import rearrange
import wandb
import time
from torchvision import transforms

from constants import FPS
from constants import PUPPET_GRIPPER_JOINT_OPEN
from general_utils.utils import load_data # data functions
from general_utils.utils import sample_box_pose, sample_insertion_pose # robot functions
from general_utils.utils import set_seed # helper functions
from act.training import train_bc
from visualize_episodes import save_videos
from general_utils.config_handler import create_config, create_task_config, make_policy
from act.evaluation import eval_bc

from detr.models.latent_model import Latent_Model_Transformer

from sim_env import BOX_POSE

import IPython
e = IPython.embed

WANDB_PROJECT = "eggs-machina"
WANDB_ENTITY = "alanbohannon-hypernour-llc"


def evaluate(config):
    ckpt_names = [f'policy_last.ckpt']
    results = []
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)
        # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
        results.append([ckpt_name, success_rate, avg_return])

    for ckpt_name, success_rate, avg_return in results:
        print(f'{ckpt_name}: {success_rate=} {avg_return=}')
    print()


def save_dataset_stats(ckpt_dir, stats):
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

def save_best_checkpoint(ckpt_dir, best_ckpt_info):
    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    task_config = create_task_config(is_sim, task_name)

    dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    config = create_config(task_config, args, policy_class, camera_names, ckpt_dir, task_name, is_sim)

    if is_eval:
        evaluate(config)
    else:
        expr_name = ckpt_dir.split('/')[-1]
        wandb.init(project=WANDB_PROJECT, reinit=True, entity=WANDB_ENTITY, name=expr_name)
        wandb.config.update(config)
        train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, args['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)

        save_dataset_stats(ckpt_dir, stats)

        best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)

        save_best_checkpoint(ckpt_dir, best_ckpt_info)
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')
    
    main(vars(parser.parse_args()))
