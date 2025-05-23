import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from JY_model_v4 import ScoreNet  # 用你已有的模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Generate toy data ---
N = 10000
mean = torch.tensor([0.0, 0.0])
cov = torch.tensor([[1.0, 0.5], [0.5, 1.5]])
L = torch.linalg.cholesky(cov)
x_data = torch.randn(N, 2) @ L.T + mean  # (N, 2)

# --- Hyperparameters ---
EPOCHS = 100
BATCH_SIZE = 128
SIGMA_MIN = 0.05
SIGMA_MAX = 0.5
lr = 1e-4

def get_sigma(t):
    return SIGMA_MIN * ((SIGMA_MAX / SIGMA_MIN) ** t)

model = ScoreNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

x_data = x_data.to(device)

# --- Training ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    perm = torch.randperm(N)

    for i in range(0, N, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        x = x_data[idx]  # (B, 2)

        t = torch.rand((x.shape[0], 1), device=device)
        sigma = get_sigma(t)  # (B, 1)

        eps = torch.randn_like(x)
        x_t = x + sigma * eps

        score_pred = model(x_t, t.squeeze())  # (B, 2)
        target = -eps / sigma

        loss = ((score_pred - target)**2).sum(dim=1).mean()
        # loss = ((score_pred - target)**2).sum(dim=1)
        # loss = (loss * sigma.detach().squeeze()**1.5).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] loss = {total_loss:.4f}")

# --- Optional: Visualize learned score field ---
model.eval()
with torch.no_grad():
    grid_x, grid_y = torch.meshgrid(torch.linspace(-3, 3, 20), torch.linspace(-3, 3, 20), indexing='ij')
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).to(device)
    t_val = torch.ones((grid.shape[0],), device=device) * 0.01  # use small t
    score = model(grid, t_val)  # (N_grid, 2)

    score = score.cpu().numpy()
    grid = grid.cpu().numpy()

    plt.quiver(grid[:, 0], grid[:, 1], score[:, 0], score[:, 1], angles='xy', scale_units='xy', scale=10)
    plt.title("Learned Score Field (t=0.01)")
    plt.axis('equal')
    plt.show()
