# Get each events particle pid pt eta phi from ROOT file, and convert them to point clouds 
# From root file to batch = [tensor(shape=(N_1, 3)), tensor(shape=(N_2, 3)), ...]

import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

class ParticlesPointDataset(Dataset):
    def __init__(self, rootfile_path):
        file = uproot.open(rootfile_path) # open the Data ROOT file
        tree = file["T_Final_Hadron_Dis"] # read the tree from the file

        pid_array = tree["pid"].arrays(library="ak")["pid"] # return a awkward array, eg.{[p1_1, p2_1, ..., pm_1], [], ... , [p1_n, p2_n, ..., pm_n]} n events, m particles for each event.
        pt_array = tree["pt"].arrays(library="ak")["pt"]
        eta_array = tree["eta"].arrays(library="ak")["eta"]
        phi_array = tree["phi"].arrays(library="ak")["phi"]

        px_array = tree["px"].arrays(library="ak")["px"]
        py_array = tree["py"].arrays(library="ak")["py"]
        pz_array = tree["pz"].arrays(library="ak")["pz"]

        self.particles = []
        self.point_clouds = []
        
        # loop events, get pid pt eta phi of every particle in each event
        for pt_evt, eta_evt, phi_evt, px_evt, py_evt, pz_evt in zip(pt_array, eta_array, phi_array, px_array, py_array, pz_array): 
            pt_evt = np.array(pt_evt, dtype=np.float32)
            eta_evt = np.array(eta_evt, dtype=np.float32)
            phi_evt = np.array(phi_evt, dtype=np.float32)
            px_evt = np.array(px_evt, dtype=np.float32)
            py_evt = np.array(py_evt, dtype=np.float32)
            pz_evt = np.array(pz_evt, dtype=np.float32)

            # for i in range(len(px_evt)):
            #     self.particles.append(np.array([px_evt[i], py_evt[i]], dtype=np.float32))  # 每个粒子一个样本

            # points = np.stack([pt_evt, eta_evt, phi_evt], axis=-1)  # shape: (N, 3)
            points = np.stack([px_evt, py_evt], axis=-1)  # shape: (N, 3)
            # points = np.stack([phi_evt], axis=-1)  # shape: (N, 1)
            self.point_clouds.append(points)

        # # Step 2: 计算全局 mean 和 std
        # all_points = np.concatenate(self.point_clouds, axis=0)  # shape: (total_particles, 3)
        # self.mean = np.mean(all_points, axis=0)  # shape: (3,)
        # self.std = np.std(all_points, axis=0)    # shape: (3,)
    
        # # Step 3: 对每个事件做标准化
        # self.point_clouds = [(pc - self.mean) / self.std for pc in self.point_clouds]
    
        # # Step 4: 保存 mean 和 std（用于 sampling 后反归一化）
        # np.save("mean.npy", self.mean)
        # np.save("std.npy", self.std)

    # tell pytorch how many samples in the dataset (how many events)
    def __len__(self):
        return len(self.point_clouds)

    # tell pytorch how to get the dataset of idx
    def __getitem__(self, idx):
        return self.point_clouds[idx]  # (N, 3) np.array
    
    # def __len__(self):
    #     return len(self.particles)

    # def __getitem__(self, idx):
    #     return torch.from_numpy(self.particles[idx])  # 单个粒子是 shape=(2,)
    

def my_collate_fn(batch):
    """
    batch 是一个 list, 包含 batch_size 个事件（每个事件是 shape=(N_i, 4) 的 NumPy array)
    我们把它变成 list[tensor]，每个 tensor 是 (N_i, 4)，方便送入模型。
    """
    return [torch.from_numpy(item) for item in batch]

# def my_collate_fn(batch):
#     return torch.stack(batch)
