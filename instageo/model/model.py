# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Model Module."""

import os
import time
from pathlib import Path

import numpy as np
import requests  # type: ignore
import torch
import torch.nn as nn
import yaml  # type: ignore
from absl import logging

from instageo.model.Prithvi import ViTEncoder


def download_file(url: str, filename: str | Path, retries: int = 3) -> None:
    """Downloads a file from the given URL and saves it to a local file.

    Args:
        url (str): The URL from which to download the file.
        filename (str): The local path where the file will be saved.
        retries (int, optional): The number of times to retry the download
                                 in case of failure. Defaults to 3.

    Raises:
        Exception: If the download fails after the specified number of retries.

    Returns:
        None
    """
    if os.path.exists(filename):
        logging.info(f"File '{filename}' already exists. Skipping download.")
        return

    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                logging.info(f"Download successful on attempt {attempt + 1}")
                break
            else:
                logging.warning(
                    f"Attempt {attempt + 1} failed with status code {response.status_code}"  # noqa
                )
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < retries - 1:
            time.sleep(2)

    else:
        raise Exception("Failed to download the file after several attempts.")


class Norm2D(nn.Module):
    """A normalization layer for 2D inputs.

    This class implements a 2D normalization layer using Layer Normalization.
    It is designed to normalize 2D inputs (e.g., images or feature maps in a
    convolutional neural network).

    Attributes:
        ln (nn.LayerNorm): The layer normalization component.

    Args:
        embed_dim (int): The number of features of the input tensor (i.e., the number of
            channels in the case of images).

    Methods:
        forward: Applies normalization to the input tensor.
    """

    def __init__(self, embed_dim: int):
        """Initializes the Norm2D module.

        Args:
            embed_dim (int): The number of features of the input tensor.
        """
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the normalization process to the input tensor.

        Args:
            x (torch.Tensor): A 4D input tensor with shape
                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The normalized tensor, having the same shape as the input.
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


import math 
class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: dict, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states
    
import argparse
def dict_to_namespace(d):
    """Recursively convert dictionary to argparse.Namespace"""
    namespace = argparse.Namespace()
    for key, value in d.items():
        setattr(namespace, key, dict_to_namespace(value) if isinstance(value, dict) else value)
    return namespace

config_dict = {
     "hidden_sizes" : [32, 64, 160, 256],
     "decoder_hidden_size" : 256,
     "num_encoder_blocks": 4, 
     "num_labels" : 2,
     "classifier_dropout_prob" : 0.1
}
config = dict_to_namespace(config_dict)

class SegformerDecodeHead(nn.Module):
    def __init__(self, config):
        super().__init__()  # Fix here: remove config from super()
        self.config = config
        # Rest of the code remains unchanged
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)
        return logits

class PrithviSeg(nn.Module):
    """Prithvi Segmentation Model."""

    def __init__(
        self,
        temporal_step: int = 1,
        image_size: int = 224,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        head_type = None
    ) -> None:
        """Initialize the PrithviSeg model.

        This model is designed for image segmentation tasks on remote sensing data.
        It loads Prithvi configuration and weights and sets up a ViTEncoder backbone
        along with a segmentation head.

        Args:
            temporal_step (int): Size of temporal dimension.
            image_size (int): Size of input image.
            num_classes (int): Number of target classes.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
        """
        super().__init__()
        weights_dir = Path.home() / ".instageo" / "prithvi"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "Prithvi_EO_V1_100M.pt"
        cfg_path = weights_dir / "config.yaml"
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt?download=true",  # noqa
            weights_path,
        )
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/raw/main/config.yaml",  # noqa
            cfg_path,
        )
        checkpoint = torch.load(weights_path, map_location="cpu")
        with open(cfg_path) as f:
            model_config = yaml.safe_load(f)

        model_args = model_config["model_args"]

        model_args["num_frames"] = temporal_step
        model_args["img_size"] = image_size
        self.model_args = model_args
        self.temporal_step = temporal_step
        # instantiate model
        model = ViTEncoder(**model_args)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        filtered_checkpoint_state_dict = {
            key[len("encoder.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("encoder.")
        }
        filtered_checkpoint_state_dict["pos_embed"] = torch.zeros(
            1, (temporal_step * (image_size // 16) ** 2 + 1), 768
        )
        _ = model.load_state_dict(filtered_checkpoint_state_dict)

        self.prithvi_100M_backbone = model

        def upscaling_block(in_channels: int, out_channels: int) -> nn.Module:
            """Upscaling block.

            Args:
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.

            Returns:
                An upscaling block configured to upscale spatially.
            """
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(p=0.5)
            )

        embed_dims = [
                (model_args["embed_dim"] * model_args["num_frames"]) // (2**i)
                for i in range(5)
            ]
        if head_type is None:
            self.segmentation_head = nn.Sequential(
                *[upscaling_block(embed_dims[i], embed_dims[i + 1]) for i in range(4)],
                nn.Conv2d(
                    kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes
                ),
            )
        elif head_type == "segformer":
           
            self.segmentation_head = SegformerDecodeHead(config)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            img (torch.Tensor): The input tensor representing the image.

        Returns:
            torch.Tensor: Output tensor after image segmentation.
        """
        features = self.prithvi_100M_backbone(img)
        # drop cls token
        reshaped_features = features[:, 1:, :]
        feature_img_side_length = int(
            np.sqrt(reshaped_features.shape[1] // self.model_args["num_frames"])
        )
        reshaped_features = reshaped_features.permute(0, 2, 1).reshape(
            features.shape[0], -1, feature_img_side_length, feature_img_side_length
        )

        out = self.segmentation_head(reshaped_features)
        return out