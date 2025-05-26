# === data_v3.py ===
import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

class PxPyVectorDataset(Dataset):
    def __init__(self, rootfile_path, img_size=(100, 100), px_range=(-5, 5), py_range=(-5, 5)):
        self.img_size = img_size
        self.px_min, self.px_max = px_range
        self.py_min, self.py_max = py_range

        file = uproot.open(rootfile_path)
        tree = file["T_Final_Hadron_Dis"]

        px_array = tree["px"].arrays(library="ak")["px"]
        py_array = tree["py"].arrays(library="ak")["py"]

        self.images = []

        for px_evt, py_evt in zip(px_array, py_array):
            px_evt = np.array(px_evt, dtype=np.float32)
            py_evt = np.array(py_evt, dtype=np.float32)

            mask = (
                (px_evt >= self.px_min) & (px_evt < self.px_max) &
                (py_evt >= self.py_min) & (py_evt < self.py_max)
            )
            px_evt = px_evt[mask]
            py_evt = py_evt[mask]

            # 粒子数直方图
            hist_count, xedges, yedges = np.histogram2d(
                px_evt, py_evt,
                bins=self.img_size,
                range=[[self.px_min, self.px_max], [self.py_min, self.py_max]]
            )

            # px 分量直方图（方向信息）
            hist_pxsum, _, _ = np.histogram2d(
                px_evt, py_evt,
                bins=self.img_size,
                range=[[self.px_min, self.px_max], [self.py_min, self.py_max]],
                weights=px_evt
            )

            image = np.stack([hist_count, hist_pxsum], axis=0).astype(np.float32)  # shape = (2, H, W)
            self.images.append((image, xedges, yedges))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, xedges, yedges = self.images[idx]
        return torch.from_numpy(img), xedges, yedges

def my_collate_fn(batch):
    images, xedges_list, yedges_list = zip(*batch)
    return torch.stack(images), xedges_list, yedges_list