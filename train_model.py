import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import BertTokenizer, BertModel
# from custom_datasets.sam_dataset import SamDataset
from sam import SamDataset
from tqdm.auto import tqdm

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
TEXT_ENCODER = "bert-base-uncased"
IMAGE_FOLDER = "data/sam_images"
CAPTION_FOLDER = "data/sam_captions"
IDS_TRAIN = "data/ids_train.pickle"
ID_DICT = "data/id_dict.pickle"
RESOLUTION = 512
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
OUTPUT_DIR = "./unet_ckpt"

def make_train_loader():
    def preprocess(ex):
        img = ex["image"].convert("RGB").resize((RESOLUTION, RESOLUTION))
        px = transforms.ToTensor()(img)
        txt = tokenizer(
            ex["text"][0],
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        return {
            "pixel_values":   px,
            "input_ids":      txt.input_ids.squeeze(0),
            "attention_mask": txt.attention_mask.squeeze(0)
        }
    ds = SamDataset(
        image_folder_path=IMAGE_FOLDER,
        caption_folder_path=CAPTION_FOLDER,
        id_file=IDS_TRAIN,
        id_dict_file=ID_DICT,
        transforms=preprocess,
        resolution=RESOLUTION
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    accelerator = Accelerator()
    device = accelerator.device

    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)

    tokenizer = BertTokenizer.from_pretrained(TEXT_ENCODER)
    text_encoder = BertModel.from_pretrained(TEXT_ENCODER).to(device)

    scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    train_loader = make_train_loader()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)

    for epoch in range(NUM_EPOCHS):
        unet.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            noisy = scheduler.add_noise(latents, noise, timesteps)
            enc = text_encoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            encoder_out = enc.last_hidden_state
            pred = unet(noisy, timesteps, encoder_out, return_dict=False)[0]
            loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
