import torch
import matplotlib.pyplot as plt
import numpy as np

score = torch.load("score_vector_output.pt")[0, 0].numpy()
H, W = score.shape
Y, X = np.mgrid[0:H:5, 0:W:5]
U = score[::5, ::5]  # 假设你当前只有一个方向分量（伪装成 U）
V = np.zeros_like(U)  # 没有第二个分量就设为 0
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='red')
plt.title("Score Vector Field (Pseudo-1D)")
plt.show()
