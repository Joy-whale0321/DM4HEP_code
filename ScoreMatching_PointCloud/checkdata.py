import uproot
import awkward as ak
import numpy as np
from torch.utils.data import DataLoader
from JY_data import ParticlesPointDataset, my_collate_fn

# 加载数据
dataset = ParticlesPointDataset("/mnt/c/Users/12896/Desktop/GeneAI/DM4HEP/Dataset/AMPT_AuAu/GeDataset_fb07_1k.root")  # 替换成你的输入ROOT路径
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)

px_list = []
py_list = []

for batch in loader:
    # 每个 batch 是一个事件的 list（因为 batch_size=1）
    for event in batch:
        px = event[:, 0].numpy().tolist()
        py = event[:, 1].numpy().tolist()
        px_list.append(px)
        py_list.append(py)

# 转换成 awkward array
px_array = ak.Array(px_list)
py_array = ak.Array(py_list)
pt_array = np.sqrt(px_array ** 2 + py_array ** 2)

# 写入 ROOT 文件
with uproot.recreate("checkfile/InputPointCloud.root") as f:
    f["T_Input"] = {
        "px": px_array,
        "py": py_array,
        "pt": pt_array
    }

print("✅ 写入成功：InputPointCloud.root")
