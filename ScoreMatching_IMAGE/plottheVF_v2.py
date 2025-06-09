import torch
import matplotlib.pyplot as plt
import numpy as np

score = torch.load("score_vector_output.pt")  # shape: (B, 2, H, W)
score_np = score[0].numpy()  # (2, H, W)

U = score_np[0]
V = score_np[1]
H, W = U.shape
X, Y = np.meshgrid(np.arange(W), np.arange(H))

plt.figure(figsize=(6, 6))
plt.quiver(X, Y, U, V, scale=10, angles="xy")
plt.gca().invert_yaxis()
plt.title("Final Score Vector Field")
plt.savefig("final_score_vector.png")
plt.show()
