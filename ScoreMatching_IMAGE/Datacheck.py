import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch


def save_example_events(dataset, out_dir="event_images", every_n=100, max_events=None):
    """
    从 dataset 中每隔 every_n 个 event 保存一个 px-py 分布图像为 PDF。

    参数:
    - dataset: 你的 PxPyImageDataset 对象
    - out_dir: 保存目录
    - every_n: 每隔多少个 event 保存一次
    - max_events: 最多处理多少个 event（默认遍历全体）
    """
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    print(f"Saving px-py images to {out_path} ...")

    total = len(dataset) if max_events is None else min(max_events, len(dataset))

    for i in range(total):
        if i % every_n == 0:
            H, xedges, yedges = dataset[i]
            H = H[0].numpy()

            plt.figure(figsize=(5, 5))
            plt.imshow(H.T, origin="lower", cmap="inferno", aspect='equal', vmin=-5, vmax=None)
            # plt.imshow(np.log1p(H.T), origin="lower", cmap="inferno", extent=[
            #     xedges[0], xedges[-1], yedges[0], yedges[-1]
            # ], aspect='equal')

            plt.colorbar(label="counts")
            plt.title(f"Event #{i}")
            plt.xlabel("px")
            plt.ylabel("py")

            pdf_path = out_path / f"event_{i:04d}.pdf"
            plt.savefig(pdf_path)
            plt.close()

    print("Finished saving example event images.")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def save_noisy_examples(dataset, get_sigma, device, out_file="noisy_examples.pdf"):
    """
    从 dataset 中取一个图像，展示不同时间 t 下的加噪图像，并保存为 PDF。

    参数：
    - dataset: PxPyImageDataset 对象
    - get_sigma: 函数，输入 t 返回对应 sigma
    - device: torch.device
    - out_file: 保存文件名（PDF）
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
    sample_img = next(iter(loader))[0]  # 拿到 image tensor tuple
    sample_img = torch.stack(sample_img).to(device)  # (1, 1, H, W)

    t_vals = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], device=device)
    fig, axs = plt.subplots(1, len(t_vals), figsize=(15, 3))

    for i, t in enumerate(t_vals):
        sigma = get_sigma(t).view(1, 1, 1, 1)
        eps = torch.randn_like(sample_img)
        x_t = sample_img + sigma * eps

        img_np = x_t[0, 0].cpu().numpy().T  # ✅ 注意：转置
        # axs[i].imshow(np.log1p(img_np), cmap="inferno", origin="lower", aspect="equal")
        axs[i].imshow(img_np, cmap="inferno", origin="lower", aspect="equal", vmin=0, vmax=None)
        axs[i].set_title(f"t={t.item():.1f}")
        axs[i].axis("off")

    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"Saved noisy examples to {out_file}")
