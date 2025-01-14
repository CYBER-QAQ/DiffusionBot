#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py -p lerobot/diffusion_pusht eval.n_episodes=10
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.

```
python lerobot/scripts/eval.py \
    -p outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    eval.n_episodes=10
```

Note that in both examples, the repo/folder should contain at least `config.json`, `config.yaml` and
`model.safetensors`.

Note the formatting for providing the number of episodes. Generally, you may provide any number of arguments
with `qualified.parameter.name=value`. In this case, the parameter eval.n_episodes appears as `n_episodes`
nested under `eval` in the `config.yaml` found at
https://huggingface.co/lerobot/diffusion_pusht/tree/main.
"""

import yaml
import argparse
import json, sys, os
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from huggingface_hub import snapshot_download
# from huggingface_hub.utils._errors import RepositoryNotFoundError
# from huggingface_hub.utils._validators import HFValidationError
from torch import Tensor, nn
from tqdm import trange
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    inside_slurm,
    set_global_seed,
)


PATH = "../outputs/train/2025-01-02/15-18-45_real_world_diffusion_default"
PATH = "../outputs/train/2025-01-02/15-18-45_real_world_diffusion_default/checkpoints/last/pretrained_model"

def get_pretrained_policy_path(pretrained_policy_name_or_path, revision=None):
    # try:
        # pretrained_policy_path = Path(snapshot_download(pretrained_policy_name_or_path, revision=revision))
    # except (HFValidationError, RepositoryNotFoundError) as e:
    #     if isinstance(e, HFValidationError):
    #         error_message = (
    #             "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
    #         )
    #     else:
    #         error_message = (
    #             "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
    #         )

        # logging.warning(f"{error_message} Treating it as a local directory.")
    pretrained_policy_path = Path(pretrained_policy_name_or_path)
    if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
        raise ValueError(
            "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
            "repo ID, nor is it an existing local directory."
        )
    return pretrained_policy_path


def config_policy_for_inference(ckpt="act_2cam"):
    init_logging()
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument(
    #     "-p",
    #     "--pretrained-policy-name-or-path",
    #     help=(
    #         "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
    #         "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
    #         "(useful for debugging). This argument is mutually exclusive with `--config`."
    #     ),
    #     default="checkpoints",
    # )
    # group.add_argument(
    #     "--config",
    #     help=(
    #         "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
    #         "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
    #     ),
    # )
    # parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    # parser.add_argument(
    #     "--out-dir",
    #     help=(
    #         "Where to save the evaluation outputs. If not provided, outputs are saved in "
    #         "outputs/eval/{timestamp}_{env_name}_{policy_name}"
    #     ),
    # )
    # parser.add_argument(
    #     "overrides",
    #     nargs="*",
    #     help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    # )
    # args = parser.parse_args()
    #print(args)
    # hydra_cfg_path = "lerobot/lerobot/configs/default.yaml" # cfg path
    hydra_cfg_path = None
    out_dir= None #args.out_dir
    config_overrides = [] #args.overrides
    pretrained_policy_path = get_pretrained_policy_path(
        os.path.join(os.path.dirname(__file__), PATH, ckpt), revision=None)#args.revision)

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


if __name__ == "__main__":
    # policy = config_policy_for_inference(ckpt="output_ckpt")
    policy = config_policy_for_inference(ckpt="")
    
    
    '''
    'observation.images.zed_left' and 'observation.images.zed_right' are captured from RGB cameras, and 'observation.state' is captured from the environment state.
    obs_dict:
        observation.images.zed_left: (720, 1280, 3) numpy array
        observation.images.zed_right: (720, 1280, 3) numpy array
        observation.state: (28,) numpy array
        observation.last_action: (28,) numpy array
        score: (1,) numpy array
    returns:
        action: (28,) numpy array
    '''
    
    
    for t in range(1000):
        now = time.time()
        # observation = {
        #     'observation.images.zed_left':np.zeros((720, 1280, 3)),
        #     'observation.images.zed_right':np.zeros((720, 1280, 3)),
        #     'observation.state':np.zeros((28,)),
        #     'observation.last_action': np.zeros((28,)),
        #     'score': np.zeros((1,)),
        # }
        observation = {
            'observation.images.zed_left':torch.zeros((1, 3, 720, 1280)).to('cuda'),
            'observation.images.zed_right':torch.zeros((1, 3, 720, 1280)).to('cuda'),
            'observation.state':torch.zeros((1,28,)).to('cuda'),
            'observation.last_action': torch.zeros((28,)).to('cuda'),
            'score': torch.zeros((1,)).to('cuda'),
        }
        with torch.inference_mode():
            action = policy.select_action(observation)

        action = action.cpu()
        if t==0:
            a0 = np.array(action)
        elif t<32:
            print("diff", np.sum(np.abs(a0-np.array(action))[0:14]))
        else:
            break

        print(time.time()-now)

