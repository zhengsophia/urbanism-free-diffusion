import pickle
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SA1BDataset(Dataset):
    def __init__(
        self,
        ids_pkl: Path,
        transform: Optional[transforms.Compose] = None,
    ):
        self.ids = pickle.load(open(ids_pkl, "rb"))
        self.img_dir = Path("SamDataset/images/")
        self.cap_dir = Path("SamDataset/captions/")
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)
    def __getitem__(self,idx):
        sa_id = self.ids[idx]
        img_path = self.img_dir/f"{sa_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        cap_path = self.cap_dir/f"{sa_id}.txt"
        caption = cap_path.read_text(encoding="utf-8").strip()
        return {"id": sa_id, "image": image, "caption": caption}
        
class SA1BDatasetFinetune(SA1BDataset):
    def __init__(
        self,
        ids_pkl: Path,
        ids_pkl2: Path,
        transform: Optional[transforms.Compose] = None,
    ):
        self.ids = pickle.load(open(ids_pkl, "rb"))+pickle.load(open(ids_pkl2, "rb"))
        self.img_dir = Path("SamDataset/images/")
        self.cap_dir = Path("SamDataset/captions/")
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

def get_dataloader(
    topic: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:
    ids_pkl = Path(f"SamDataset/ids/{topic}_ids.pkl")
    ds = SA1BDataset(
        ids_pkl=ids_pkl,
        transform=transform,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
def get_finetuneloader(
    topic: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:
    ids_pkl = Path(f"SamDataset/ids/{topic}_ids.pkl")
    ds = SA1BDatasetFinetune(
        ids_pkl=Path(f"SamDataset/ids/filtered_ids.pkl")
        ids_pkl2=ids_pkl,
        transform=transform,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

if __name__ == "__main__":
    vehicle_loader = get_dataloader("vehicle", batch_size=16, num_workers=8)
    batch=next(iter(vehicle_loader))
    print(len(vehicle_loader))
    print("IDs:",batch["id"])
    print("Images:", batch["image"].shape)
    print("Captions:", batch["caption"])
