# particles_dataset.py
import uproot
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset

class ParticlesPointDataset(Dataset):
    """
    拆解所有 event 中的粒子，每个样本是一个粒子坐标 [px, py]
    """

    def __init__(self, rootfile_path, treename="T_Final_Hadron_Dis", px_key="px", py_key="py"):
        file = uproot.open(rootfile_path)
        tree = file[treename]
        px_array = tree[px_key].arrays(library="ak")[px_key]
        py_array = tree[py_key].arrays(library="ak")[py_key]

        self.particles = []
        for px_evt, py_evt in itertools.islice(zip(px_array, py_array), 100):
            px = np.array(px_evt, dtype=np.float32)
            py = np.array(py_evt, dtype=np.float32)
            for x, y in zip(px, py):
                self.particles.append([x, y])  # 每个粒子是一个样本 [px, py]

        self.particles = torch.tensor(self.particles, dtype=torch.float32)  # shape: [total_particles, 2]

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, idx):
        return self.particles[idx]  # shape: [2]
