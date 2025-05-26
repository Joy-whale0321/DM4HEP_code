import uproot
import numpy as np
import torch
from torch.utils.data import Dataset

class ParticleDataset(Dataset):
    def __init__(self, rootfile_path, tree_name="your_tree_name", max_events=100):
        file = uproot.open(rootfile_path)
        tree = file[tree_name]

        px_array = tree["px"].array()
        py_array = tree["py"].array()
        pz_array = tree["pz"].array()

        # 限制最多读取 max_events 个 event
        max_events = min(max_events, len(px_array))
        px_array = px_array[:max_events]
        py_array = py_array[:max_events]
        pz_array = pz_array[:max_events]

        self.particles = np.vstack([
            np.stack([px_array[i], py_array[i]], axis=-1)
            for i in range(max_events)
        ])

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, idx):
        return torch.tensor(self.particles[idx], dtype=torch.float32)
