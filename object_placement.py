import os
import math
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

sys.path.append("./stable_diffusion")

from edit_cli import CFGDenoiser, load_model_from_config
from stable_diffusion.ldm.util import instantiate_from_config


def get_data(data_path: str) -> pd.DataFrame:
    # Expected columns: ["bg_image_path", "object_to_add", "filename"]
    data = pd.read_csv(data_path)
    return data


def preprocess_images(images_paths, resolution):
    return [
        Image.open(img_path).convert("RGB").resize((resolution, resolution))
        for img_path in images_paths
    ]


def get_args(args):
    parser = ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/instruct_diffusion.yaml", type=str)
    parser.add_argument(
        "--ckpt", default="checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt", type=str
    )
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--cfg-text", default=5.0, type=float)
    parser.add_argument("--cfg-image", default=1.25, type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args(args)


def main():
    args = get_args(sys.argv[1:])

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed

    data = get_data(args.data_path)
    bg_images = preprocess_images(data["bg_image_path"], args.resolution)

    for bg_image, object_to_add, filename in zip(
        bg_images, data["object_to_add"].to_list(), data["filename"].to_list()
    ):
        width, height = bg_image.size
        edit = f"add a {object_to_add}"
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([edit])]
        bg_image = 2 * torch.tensor(np.array(bg_image)).float() / 255 - 1
        bg_image = rearrange(bg_image, "h w c -> 1 c h w").to(
            next(model.parameters()).device
        )
        cond["c_concat"] = [model.encode_first_stage(bg_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(args.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }

        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(
            model_wrap_cfg, z, sigmas, extra_args=extra_args
        )
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        print(x.shape)
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

        edited_image = ImageOps.fit(
            edited_image, (width, height), method=Image.Resampling.LANCZOS
        )
        edited_image.save(os.path.join(args.output_dir, f"{filename}.png"))


if __name__ == "__main__":
    main()
