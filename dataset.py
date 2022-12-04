import os
from PIL import Image
import torch
import torchvision
import tqdm
import pandas as pd
import glob 

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((112, 112)),  # resize to 224*224
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # normalization
    ]
)
to_tensor = torchvision.transforms.ToTensor()
Image.MAX_IMAGE_PIXELS = None


class BreastDataset(torch.utils.data.Dataset):
    """Pytorch dataset api for loading patches and preprocessed clinical data of breast."""

    def __init__(self, csv_path, data_dir_path='./dataset', is_preloading=True):
        self.data_dir_path = data_dir_path
        self.is_preloading = is_preloading
        self.csv_data=pd.read_csv(csv_path)

        if self.is_preloading:
            self.bag_tensor_list = self.preload_bag_data()

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        label = int(self.csv_data.iloc[index]["N_category"])
        patient_id = self.csv_data.iloc[index]["ID"]
        patch_paths = glob.glob(os.path.join(self.data_dir_path, patient_id, "*.png"))

        data = {}
        if self.is_preloading:
            data["bag_tensor"] = self.bag_tensor_list[index]
        else:
            data["bag_tensor"] = self.load_bag_tensor(patch_paths)

        data["label"] = label
        data["patient_id"] = patient_id
        data["patch_paths"] = patch_paths

        return data

    def preload_bag_data(self):
        """Preload data into memory"""

        bag_tensor_list = []
        for item in tqdm.tqdm(self.json_data, ncols=120, desc="Preloading bag data"):
            patch_paths = [os.path.join(self.data_dir_path, p_path) for p_path in item["patch_paths"]]
            bag_tensor = self.load_bag_tensor(patch_paths)  # [N, C, H, W]
            bag_tensor_list.append(bag_tensor)

        return bag_tensor_list

    def load_bag_tensor(self, patch_paths):
        """Load a bag data as tensor with shape [N, C, H, W]"""

        patch_tensor_list = []
        for p_path in patch_paths:
            patch = Image.open(p_path).convert("RGB")
            patch_tensor = transform(patch)  # [C, H, W]
            patch_tensor = torch.unsqueeze(patch_tensor, dim=0)  # [1, C, H, W]
            patch_tensor_list.append(patch_tensor)

        bag_tensor = torch.cat(patch_tensor_list, dim=0)  # [N, C, H, W]

        return bag_tensor
