import os
from PIL import Image
import torch
import tqdm
import pandas as pd
import glob 
import cv2

class BreastDataset(torch.utils.data.Dataset):
    """Pytorch dataset api for loading patches and preprocessed clinical data of breast."""

    def __init__(self, data_df, data_dir_path='./dataset',transforms=None):
        self.data_dir_path = data_dir_path
        self.data_df=data_df
        self.transforms=transforms

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = int(self.data_df.iloc[index]["N_category"])
        patient_id = self.data_df.iloc[index]["ID"]
        patch_paths = glob.glob(os.path.join(self.data_dir_path, patient_id, "*.png"))

        data = {}

        data["bag_tensor"] = self.load_bag_tensor(patch_paths)

        data["label"] = label
        data["patient_id"] = patient_id
        data["patch_paths"] = patch_paths

        return data

    def load_bag_tensor(self, patch_paths):
        """Load a bag data as tensor with shape [N, C, H, W]"""

        patch_tensor_list = []
        for p_path in patch_paths:
            image = cv2.imread(p_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            patch_tensor = self.transforms(image=image)['image']  # [C, H, W]
            patch_tensor = torch.unsqueeze(patch_tensor, dim=0)  # [1, C, H, W]
            patch_tensor_list.append(patch_tensor)

        bag_tensor = torch.cat(patch_tensor_list, dim=0)  # [N, C, H, W]

        return bag_tensor


class BreastDatasetMulti(torch.utils.data.Dataset):
    """Pytorch dataset api for loading patches and preprocessed clinical data of breast."""
    def __init__(self, data_df, data_dir_path='./dataset',transforms=None):
        self.data_dir_path = data_dir_path
        self.data_df=data_df
        self.transforms=transforms

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = int(self.data_df.iloc[index]["N_category"])
        patient_id = self.data_df.iloc[index]["ID"]
        patch_paths = glob.glob(os.path.join(self.data_dir_path, patient_id, "*.png"))
        clinical_data = torch.Tensor(self.data_df.iloc[index][['나이', '암의 장경', 'KI-67_LI_percent', '암의 개수','NG', 'HG', 'HG_score_1', 'HG_score_2', 'HG_score_3', 'ER','PR','HER2','DCIS_or_LCIS_여부', '진단명', '암의 위치', 'T_category', 'BRCA_mutation']])

        data = {}
        data["bag_tensor"] = self.load_bag_tensor(patch_paths)

        data["label"] = label
        data["patient_id"] = patient_id
        data["patch_paths"] = patch_paths
        data["clinical"] = clinical_data

        return data

    def load_bag_tensor(self, patch_paths):
        """Load a bag data as tensor with shape [N, C, H, W]"""

        patch_tensor_list = []
        for p_path in patch_paths:
            image = cv2.imread(p_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            patch_tensor = self.transforms(image=image)['image']  # [C, H, W]
            patch_tensor = torch.unsqueeze(patch_tensor, dim=0)  # [1, C, H, W]
            patch_tensor_list.append(patch_tensor)

        bag_tensor = torch.cat(patch_tensor_list, dim=0)  # [N, C, H, W]

        return bag_tensor