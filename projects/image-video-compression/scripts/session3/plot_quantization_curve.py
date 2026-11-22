from src.helpers import read_img
from src.quantization import midtread_quantization, midtread_dequantization
from src.metrics import psnr_of_frame
import matplotlib.pyplot as plt

original_image = read_img("./flowergarden_cif_frame1.tif")
print("Image shape:", original_image.shape)

bit_depths = range(1, 9)
psnr_values = []

for bit_depth in bit_depths:
    stepsize = 2 / (2 ** bit_depth)

    quantized_image = midtread_quantization(original_image, stepsize)
    reconstructed_image = midtread_dequantization(quantized_image, stepsize)

    psnr = psnr_of_frame(original_image, reconstructed_image)
    psnr_values.append(psnr)

    print(f"PSNR at {bit_depth} bits/pixel: {psnr:.2f} dB")

plt.figure()
plt.plot(bit_depths, psnr_values, marker="o")
plt.xlabel("Rate (bits per pixel)")
plt.ylabel("PSNR (dB)")
plt.title("Uniform Quantization: Rate–Distortion Curve")
plt.xticks(bit_depths)
plt.grid(True)

comparison_step = 2 / (2 ** 3)
levels = midtread_quantization(original_image, comparison_step)
reconstructed_3bit = midtread_dequantization(levels, comparison_step)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(original_image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original")

axes[1].imshow(reconstructed_3bit, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Reconstructed at 3 bits/pixel")

for ax in axes:
    ax.axis("off")

fig.tight_layout()

plt.show()
