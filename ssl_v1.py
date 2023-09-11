import torch
import warnings
from utils import *
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms


warnings.filterwarnings("ignore")

data_dir = r"/mnt/d/work/detr crops/750_crops/"
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = ContrastiveDataset(data_dir, transform=transform)

base_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

criterion = ContrastiveLoss()
optimizer = optim.Adam(base_encoder.parameters(), lr=0.0001)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_encoder.to(device)
c = 0

for epoch in range(num_epochs):
    base_encoder.train()
    total_loss = 0.0

    i = 0
    for im1, im2, flag in tqdm(dataset):
        im1 = im1.unsqueeze(0).to(device)
        im2 = im2.unsqueeze(0).to(device)

        z1 = base_encoder(im1)
        z2 = base_encoder(im2)

        loss = criterion(z1, z2, flag)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        i += 1

        if i == 10000:
            break

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / len(dataset)}")

    model_save_path = f"/mnt/d/work/model_saves/base_encoder_{epoch}.pt"

    model_state = {
        "model": base_encoder.state_dict(),
        "architecture": base_encoder,
    }

    torch.save(model_state, model_save_path)

print("Skipped:", model_save_path)
