import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.functional import cosine_similarity


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2, flag=True):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        similarity_matrix = cosine_similarity(z1, z2, dim=-1)

        if flag:
            logits = similarity_matrix / self.temperature
        else:
            logits = -similarity_matrix / self.temperature  # Negative pairs

        return logits


def resize_to_nearest_multiple_of_14(image_path):
    img = Image.open(image_path)

    width, height = img.size

    new_width = max(round(width / 14) * 14, 14)
    new_height = max(round(height / 14) * 14, 14)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    return resized_img


class ContrastiveDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.folders = os.listdir(path)

    def __getitem__(self, index):
        rand_num = torch.rand(1)

        if rand_num > 0.5:
            folder = torch.randint(0, 5, (1,))
            folder = self.folders[folder]

            img1 = torch.randint(0, 750, (1,))
            img2 = torch.randint(0, 750, (1,))

            img1 = os.listdir(self.path + folder)[img1]
            img2 = os.listdir(self.path + folder)[img2]

            im1 = resize_to_nearest_multiple_of_14(
                self.path + str(folder) + "/" + str(img1)
            )
            im2 = resize_to_nearest_multiple_of_14(
                self.path + str(folder) + "/" + str(img2)
            )
            flag = True

        else:
            folder1 = torch.randint(0, 5, (1,))
            folder2 = torch.randint(0, 5, (1,))

            while folder1 == folder2:
                folder2 = torch.randint(0, 5, (1,))

            folder1 = self.folders[folder1]
            folder2 = self.folders[folder2]

            img1 = torch.randint(0, 750, (1,))
            img2 = torch.randint(0, 750, (1,))

            img1 = os.listdir(self.path + folder1)[img1]
            img2 = os.listdir(self.path + folder2)[img2]

            im1 = resize_to_nearest_multiple_of_14(
                self.path + str(folder1) + "/" + str(img1)
            )
            im2 = resize_to_nearest_multiple_of_14(
                self.path + str(folder2) + "/" + str(img2)
            )
            flag = False

        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        return torch.tensor(im1).float(), torch.tensor(im2).float(), flag

    def __len__(self):
        return 10000
