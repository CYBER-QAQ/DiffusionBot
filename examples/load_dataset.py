"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Viewing a dataset's metadata and exploring its properties.
- Loading an existing dataset from the hub or a subset of it.
- Accessing frames by episode number.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

GraspAnythingData = LeRobotDataset(repo_id="GraspAnything",root="/home/zc/DiffusionPolicy/datasets/CvtGraspAnyObject_H1-2_Inspire_20241127/")
# GraspAnythingData = LeRobotDataset(repo_id="GraspAnything",root="/home/zc/.cache/huggingface/lerobot/lerobot/pusht/")

