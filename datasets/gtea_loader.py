import numpy as np
import torch
from torch.utils.data import Dataset

class GTEADataset(Dataset):
    def __init__(self, npz_path, transform=None):
        self.data = np.load(npz_path)
        self.images = self.data["images"]  # (N, 6, 224, 224, 3)
        self.gaze = self.data["gaze"]      # (N, 2)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # shape: (6, 224, 224, 3)
        sequence = self.images[idx]
        gaze_point = self.gaze[idx]  # (2,)

        # 画像の正規化・転置（BGR→RGB、HWC→CHW）
        tensor_seq = []
        for img in sequence:
            # numpy → float32 → [0,1] → CHW
            img = img.astype(np.float32) / 255.0
            img = img[:, :, ::-1].copy()  # BGR → RGB
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            tensor_seq.append(torch.tensor(img))

        # 結果のテンソル化
        tensor_seq = torch.stack(tensor_seq)  # shape: (6, 3, 224, 224)
        gaze_tensor = torch.tensor(gaze_point, dtype=torch.float32)

        return tensor_seq, gaze_tensor
