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
import os

import cv2
import torch
from natsort import natsorted

import imgproc
import model
import realesrgan_config
from image_quality_assessment import NIQE
from utils import load_state_dict


def main() -> None:
    # Initialize the super-resolution model
    g_model = model.__dict__[realesrgan_config.g_model_arch_name](in_channels=realesrgan_config.g_in_channels,
                                                                  out_channels=realesrgan_config.g_out_channels,
                                                                  channels=realesrgan_config.g_channels,
                                                                  growth_channels=realesrgan_config.g_growth_channels,
                                                                  num_rrdb=realesrgan_config.g_num_rrdb)
    g_model = g_model.to(device=realesrgan_config.device)
    print("Build Real_ESRGAN model successfully.")

    # Load the super-resolution model weights
    g_model = load_state_dict(g_model, realesrgan_config.g_model_weights_path)
    print(f"Load RealESRGAN model weights `{os.path.abspath(realesrgan_config.g_model_weights_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    if not os.path.exists(realesrgan_config.degradation_test_sr_images_dir):
        os.makedirs(realesrgan_config.degradation_test_sr_images_dir)

    # Start the verification mode of the model.
    g_model.eval()

    # Initialize the sharpness evaluation function
    niqe = NIQE(realesrgan_config.upscale_factor, realesrgan_config.niqe_model_path)

    # Set the sharpness evaluation function calculation device to the specified model
    niqe = niqe.to(device=realesrgan_config.device)

    # Initialize IQA metrics
    niqe_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(realesrgan_config.degradation_test_lr_images_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(realesrgan_config.degradation_test_lr_images_dir, file_names[index])
        sr_image_path = os.path.join(realesrgan_config.degradation_test_sr_images_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Read LR image
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED)

        # Convert BGR channel image format data to RGB channel image format data
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert RGB channel image format data to Tensor channel image format data
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False).unsqueeze_(0)

        # Transfer Tensor channel image format data to CUDA device
        lr_tensor = lr_tensor.to(device=realesrgan_config.device, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = g_model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        niqe_metrics += niqe(sr_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # NIQE range value is 0~100
    avg_niqe = 100 if niqe_metrics / total_files > 100 else niqe_metrics / total_files

    print(f"NIQE: {avg_niqe:4.2f} 100u")


if __name__ == "__main__":
    main()
