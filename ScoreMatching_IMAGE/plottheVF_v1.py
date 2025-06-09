import torch
import matplotlib.pyplot as plt

score_field = torch.load("score_vector_output.pt")  # shape: (B, 1, H, W)
print(score_field.shape)

# 可视化第一个图像的分数函数
plt.imshow(score_field[2, 0].numpy(), cmap='seismic', origin='lower')
plt.colorbar(label='score value')
plt.title("Predicted Score Field (1st image)")
plt.show()
