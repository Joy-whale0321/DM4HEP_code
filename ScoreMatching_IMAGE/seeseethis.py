import numpy as np
import matplotlib.pyplot as plt

def generate_example_image(size=60):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    img = np.exp(-R**2 / 0.1)  # 高斯核模拟集中分布
    return img.astype(np.float32)

base_image = generate_example_image()
noise_levels = [0.1, 0.5, 1.0, 3.0]
images = [base_image] + [base_image + sigma * np.random.randn(*base_image.shape) for sigma in noise_levels]
images.append(np.random.randn(*base_image.shape))  # 纯噪声图

titles = [
    "Original",
    "Noisy σ=0.1",
    "Noisy σ=0.5",
    "Noisy σ=1.0",
    "Noisy σ=3.0",
    "Pure Gaussian Noise"
]

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
for ax, img, title in zip(axs.flat, images, titles):
    im = ax.imshow(img, cmap="inferno", origin="lower")
    ax.set_title(title)
    ax.axis("off")

fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
plt.tight_layout()
plt.show()
