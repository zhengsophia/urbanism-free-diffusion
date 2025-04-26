#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import BertTokenizer, BertModel

from dataset import SA1BDataset  # your Dataset class

MODEL_NAME   = "CompVis/stable-diffusion-v1-4"
TEXT_ENCODER = "bert-base-uncased"

def parse_args():
    p = argparse.ArgumentParser(description="DDP training of SD1.4 UNet")
    p.add_argument("--topic",       type=str,   default="filtered",
                   help="Which split/topic to train on (expects `{topic}_ids.pkl`)")
    p.add_argument("--batch-size",  type=int,   default=16,
                   help="Batch size per GPU")
    p.add_argument("--num-workers", type=int,   default=8,
                   help="DataLoader num_workers")
    p.add_argument("--epochs",      type=int,   default=100,
                   help="Number of epochs")
    p.add_argument("--lr",          type=float, default=5e-5,
                   help="Learning rate")
    p.add_argument("--resolution",  type=int,   default=512,
                   help="Image resolution (square)")
    return p.parse_args()

def main():
    args = parse_args()

    # ---- Initialize DDP ----
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Running on {world_size} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ---- VAE (frozen, no DDP) ----
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.eval()

    # ---- UNet (reinit, trainable, wrapped in DDP) ----
    base = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    config = base.config
    del base
    unet = UNet2DConditionModel.from_config(config).to(device)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # ---- BERT encoder (frozen, no DDP) ----
    tokenizer    = BertTokenizer.from_pretrained(TEXT_ENCODER)
    text_encoder = BertModel.from_pretrained(TEXT_ENCODER).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # ---- Noise scheduler ----
    scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    # ---- Dataset & DataLoader ----
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
    ])
    ids_pkl = Path(f"SamDataset/ids/{args.topic}_ids.pkl")
    dataset = SA1BDataset(ids_pkl=ids_pkl, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # ---- Training Loop ----
    checkpoint_root = Path("ckpt")
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        unet.train()
        total_loss = 0.0

        it = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}") if rank == 0 else loader
        for batch in it:
            imgs = batch["image"].to(device)
            with torch.no_grad():
                enc = vae.encode(imgs)
                latents = enc.latent_dist.sample() * vae.config.scaling_factor

            caps = batch["caption"]
            toks = tokenizer(
                caps,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            input_ids      = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
            with torch.no_grad():
                enc_states = text_encoder(
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
            noisy = scheduler.add_noise(latents, noise, timesteps)

            pred = unet(
                noisy,
                timesteps,
                encoder_hidden_states=enc_states,
                return_dict=False
            )[0]
            loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if rank == 0:
                it.set_postfix(avg_loss=total_loss / (it.n + 1))

        if rank == 0:
            avg = total_loss / len(loader)
            print(f"Epoch {epoch} complete — avg loss: {avg:.4f}")
            ckpt_dir = checkpoint_root / f"epoch_{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unet.module.save_pretrained(ckpt_dir)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
