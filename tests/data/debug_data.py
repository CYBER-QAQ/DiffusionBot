import pickle as pkl
import numpy as np
from datetime import datetime as dt
import sys, os, time
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    inside_slurm,
    set_global_seed,
)
from lerobot.scripts.config_inference_policy import get_pretrained_policy_path

import yaml
import argparse
import json, sys, os
import logging
import threading
import time
from torch import Tensor, nn
from lerobot.common.logger import log_output_dir



class DiffusionPolicy:
    def __init__(self, load_path, ckpt="Unet",):
        self.policy = config_diffusion_policy_for_inference(load_path)
        # key for dataset -> key for policy
        self.key_mapping = {
            'observation.image.camera0': 'observation.images.top',
            'observation.image.camera3': 'observation.images.front',
            'observation.image.camera.zed_left': 'observation.images.zed_left',
            'observation.image.camera.zed_right': 'observation.images.zed_right',
            'observation.state':  'observation.state',
            'observation.last_action': 'observation.last_action',
            'observation.right_tac_dict': 'observation.tac'
        }
    def reset(self):
        pass
    
    def get_action(self, obs_dict):
        obs = {}
        # print(obs_dict.keys())
        #obs['observation.lastaction'] = np.zeros((0,))
        for k in obs_dict.keys():
            if k in self.key_mapping:
                if 'image' in k:
                    obs[self.key_mapping[k]] = torch.from_numpy(obs_dict[k]).to('cuda').permute(2,0,1).unsqueeze(0).float()
                elif 'state' in k:
                    obs[self.key_mapping[k]] = torch.from_numpy(obs_dict[k]).to('cuda').unsqueeze(0)
                else:
                    obs[self.key_mapping[k]] = torch.from_numpy(obs_dict[k]).to('cuda')
        obs['score'] = torch.ones((1,)).to('cuda') * 2.0
        # for k in obs.keys():
        #     print(k, obs[k].shape)
        now = time.time()
        #print([(k,v.shape) for k,v in obs.items()])
        act = self.policy.select_action(obs)
        #print("policy time:", time.time()-now)
        return act

class TrainedPolicy:
    def __init__(self, ckpt, load_path = ""):
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../train'))
        
        from lerobot.scripts.config_inference_policy import config_policy_for_inference
        self.policy = config_policy_for_inference(ckpt=ckpt)
        
        # key for dataset -> key for policy
        self.key_mapping = {
            'observation.image.camera0': 'observation.images.top',
            'observation.image.camera3': 'observation.images.front',
            'observation.image.camera.zed_left': 'observation.images.egol',
            'observation.image.camera.zed_right': 'observation.images.egor',
            'observation.state':  'observation.jointangles.arm',
            'observation.last_action': 'observation.lastaction',
            'observation.right_tac_dict': 'observation.tac'
        }
    
    def get_action(self, obs_dict):
        obs = {}
        #obs['observation.lastaction'] = np.zeros((0,))
        for k in obs_dict.keys():
            if k in self.key_mapping:
                obs[self.key_mapping[k]] = obs_dict[k]
        now = time.time()
        #print([(k,v.shape) for k,v in obs.items()])
        act = self.policy.act(obs)
        #print("policy time:", time.time()-now)
        return act


def config_diffusion_policy_for_inference(path):
    init_logging()
    hydra_cfg_path = None
    out_dir= None #args.out_dir
    config_overrides = [] #args.overrides
    pretrained_policy_path = get_pretrained_policy_path(
        os.path.join(os.path.dirname(__file__), path), revision=None)#args.revision)

    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if pretrained_policy_path is not None:
        # config is in pretrained_policy_path / "config.yaml" or pretrained_policy_path / "config.json"
        assert (pretrained_policy_path / "config.yaml").exists() or (pretrained_policy_path / "config.json").exists()
        if (pretrained_policy_path / "config.yaml").exists():
            hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), config_overrides)
        else:
            with open((pretrained_policy_path / "config.json"), "r") as json_file:
                data = json.load(json_file)
            with open((pretrained_policy_path / "config.yaml"), "w") as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False, allow_unicode=True)
            hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), config_overrides)
    else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)

    if hydra_cfg.eval.batch_size > hydra_cfg.eval.n_episodes:
        raise ValueError(
            "The eval batch size is greater than the number of eval episodes "
            f"({hydra_cfg.eval.batch_size} > {hydra_cfg.eval.n_episodes}). As a result, {hydra_cfg.eval.batch_size} "
            f"eval environments will be instantiated, but only {hydra_cfg.eval.n_episodes} will be used. "
            "This might significantly slow down evaluation. To fix this, you should update your command "
            f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={hydra_cfg.eval.batch_size}`), "
            f"or lower the batch size (e.g. `eval.batch_size={hydra_cfg.eval.n_episodes}`)."
        )

    if out_dir is None:
        out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    log_output_dir(out_dir)

    #print(hydra_cfg_path, hydra_cfg)
    #sys.exit(0)

    # make policy
    if hydra_cfg_path is None:
        print("hydra_config:", hydra_cfg)
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)

    assert isinstance(policy, nn.Module)
    policy.eval()
    return policy



# with open('PolePickPlace_H1-2_Inspire_20241119/episode_53.pkl', 'rb') as f:
#     data = pkl.load(f)
# print(data["observation.state"][30])
# sys.exit(0)

#with open('PolePickPlace_H1-2_Inspire_20241117/episode_0.observation.state.pkl', 'rb') as f:
#    data = pkl.load(f)
#print(data[100])
#sys.exit(0)

### check data
# actions = data['action']
# print(actions[:-1] - data['observation.last_action'][1:])
# for t, action in enumerate(actions[:]):
#     print(data['action'][t])#['observation.state'][t])
#     if t==0:
#         a0 = np.array(action)
#     elif t<30:
#         print("diff", np.sum(np.abs(a0-np.array(action))[0:14]))
#     else:
#         break

### check policy
from data_loader import EpisodeLoader
import torch.nn.functional as F
import torch


GRASP_BOTTLE_PATH = "../../outputs/train/2025-01-10/22-25-41_real_world_diffusion_default/checkpoints/last/pretrained_model"
GRASP_OBJECT_PATH = "../../outputs/train/2025-01-10/22-25-41_real_world_diffusion_default/checkpoints/last/pretrained_model"
GRASP_BOTTLE_MASK_PATH=""

policy = DiffusionPolicy(load_path=GRASP_BOTTLE_PATH)

# policy = TrainedPolicy('h1-2/PlayChess_resnet50_state_30_n0.0')
#policy = TrainedPolicy('h1-2/PlayChess_resnet50_state_30_n0.0')
#policy = TrainedPolicy('h1-2/GraspCoffeeStanding_resnet50_state_30_n0.0_v3')

#episode_data = EpisodeLoader("PourWater_H1-2_Inspire_20241129", i_episode=0) # training data
#episode_data = EpisodeLoader("GraspAnyObject_H1-2_Inspire_20241127", i_episode=0) # real test data
#episode_data = EpisodeLoader("GraspAnyBottle_H1-2_Inspire_20241123", i_episode=10)
#episode_data = EpisodeLoader("Minecraft_H1-2_Inspire_20241207", i_episode=10)
#episode_data = EpisodeLoader("CarryBasket_H1-2_Inspire_20241207", i_episode=8)
#episode_data = EpisodeLoader("PlaceBasket_H1-2_Inspire_20241207", i_episode=9)
#episode_data = EpisodeLoader("OpenBeer_H1-2_Inspire_20241214", i_episode=9)
#episode_data = EpisodeLoader("HandoverBasket_H1-2_Inspire_20241215", i_episode=8)
#episode_data = EpisodeLoader("PlaceCoffeeCup_H1-2_Inspire_20241218", i_episode=8)
#episode_data = EpisodeLoader("GraspCoffeeCup_H1-2_Inspire_20241218", i_episode=9)
#episode_data = EpisodeLoader("GraspCoffeeStanding_H1-2_Inspire_20241222", i_episode=287)
#episode_data = EpisodeLoader("PlaceCoffeeStanding_H1-2_Inspire_20241222", i_episode=288)
#episode_data = EpisodeLoader("PlaceCoffeeOnTable_H1-2_Inspire_20241223", i_episode=221)
#episode_data = EpisodeLoader("GraspCoffeeOnTable_H1-2_Inspire_20241223", i_episode=220)

# episode_data = EpisodeLoader("PlayChess_notac_H1-2_Inspire_20250101", i_episode=44)
episode_data = EpisodeLoader("grasp_bottle", i_episode=0)
print(len(episode_data))
MSE = 0
for t in range(len(episode_data)):
    now = time.time()
    data = episode_data.get_data(t%len(episode_data))
    #print(data['observation.last_action'])
    # if only use last action, uncomment:
    # data['observation.state'] = np.zeros((0,))
    action = policy.get_action(data)
    action = action.cpu().detach().numpy()
    #if t==0:
    #    a0 = np.array(action)
    #elif t<30:
    #   print("diff", np.sum(np.abs(a0-np.array(action))[0:14]))
    mse = F.mse_loss(torch.tensor(data["action"]), torch.tensor(action))
    l1 = F.l1_loss(torch.tensor(data["action"]), torch.tensor(action))
    print("action mse error:", mse, "L1:", l1)
    MSE += mse
    #print(data["action"], action)
    #sys.exit(0)
    print('time:', time.time()-now)

print(MSE/len(episode_data))
