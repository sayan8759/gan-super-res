# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
#from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cpu")
# Turning on when the image size does not change during training can speed up training
#cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
model_arch_name = "msrn_x2"
# Model in channels
in_channels = 3
# Model in channels
out_channels = 3
# Image magnification factor
upscale_factor = 2
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = f"{model_arch_name}"

if mode == "train":
    # Dataset address
    train_gt_images_dir = "./data/DIV2K/MSRN/train"

    test_gt_images_dir = "./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    gt_image_size = int(upscale_factor * 64)
    batch_size = 16
    num_workers = 4

    # Load the address of the pretrained model
    pretrained_model_weights_path = "./results/pretrained_models/MSRN_x2-DIV2K-e19a5cef.pth.tar"

    # Incremental training and migration training
    resume = ""

    # Total num epochs (1,000,000 iters)
    epochs = 400

    # loss function weights
    loss_weights = 1.0

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)
    model_eps = 1e-8

    # EMA parameter
    model_ema_decay = 0.99998

    # Dynamically adjust the learning rate policy (200,000 iters)
    lr_scheduler_step_size = epochs // 5
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 100
    valid_print_frequency = 1

if mode == "test":
    # Test data address
    test_gt_images_dir = "./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"./results/{exp_name}"

    model_weights_path = "./results/pretrained_models/MSRN_x2-DIV2K-e19a5cef.pth.tar"
