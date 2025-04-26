import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import BertTokenizer, BertModel

from dataset import get_dataloader

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MODEL_NAME   = "CompVis/stable-diffusion-v1-4"
TEXT_ENCODER = "bert-base-uncased"
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU training of SD1.4 UNet (reinitialized, DataParallel)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated list of GPU ids to use, e.g. '0,1,2'",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="filtered",
        help="Which split/topic to train on (expects `{topic}_ids.pkl`)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size *per iteration* across all GPUs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution for training (square)",
    )
    return parser.parse_args()
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = len(args.gpus.split(","))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using GPUs {args.gpus} → DataParallel device_ids = {list(range(num_gpus))}")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae=nn.DataParallel(vae, device_ids=list(range(num_gpus)))
    vae.requires_grad_(False)
    vae.eval()
    pretrained_unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    config = pretrained_unet.config
    del pretrained_unet
    unet = UNet2DConditionModel.from_config(config)
    unet.to(device)
    unet = nn.DataParallel(unet, device_ids=list(range(num_gpus)))
    tokenizer    = BertTokenizer.from_pretrained(TEXT_ENCODER)
    text_encoder = BertModel.from_pretrained(TEXT_ENCODER).to(device)
    text_encoder=nn.DataParallel(text_encoder, device_ids=list(range(num_gpus)))
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
    ])
    train_loader = get_dataloader(
        topic=args.topic,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        transform=transform,
    )
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        unet.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            pixel_values = batch["image"].to(device)
            with torch.no_grad():
                latents = vae.module.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.module.config.scaling_factor
            caps = batch["caption"]
            txt = tokenizer(
                caps,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            input_ids      = txt.input_ids.to(device)
            attention_mask = txt.attention_mask.to(device)
            with torch.no_grad():
                encoder_states = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_states,
                return_dict=False
            )[0]
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(avg_loss=total_loss / (pbar.n + 1))
        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch} complete — average loss: {avg:.4f}")
    os.makedirs(f"ckpt/{epoch}", exist_ok=True)
    unet.module.save_pretrained(f"ckpt/{epoch}")
    print(f"Fine-tuned UNet saved to ckpt/{epoch}")
if __name__ == "__main__":
    main()
