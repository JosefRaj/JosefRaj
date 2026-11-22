from src.metrics import psnr_of_frame
from src.helpers import read_img, design_quantizer_matrix
from src.quantization import midtread_quantization, midtread_dequantization

import numpy as np
import matplotlib.pyplot as plt


def main():
    q = 1 / 32
    img = read_img("./flowergarden_cif_frame1.tif")

    quantized = midtread_quantization(img, q)
    dequantized_uniform = midtread_dequantization(quantized, q)
    
    psnr_uniform = psnr_of_frame(img, dequantized_uniform)
    
    #q0 = 1 / 16
    #q1 = 1 / 32
    q0 = 1 / 32   # Flower bed: fine
    q1 = 1 / 16   # Rest of image: coarse
    #q_matrix, num_q0, num_q1 = design_quantizer_matrix(q0=1/16, q1=1/32)
    q_matrix, num_q0, num_q1 = design_quantizer_matrix(q0=q0, q1=q1)
    
    quantized_non_uniform = midtread_quantization(img, q_matrix)
    dequantized_non_uniform = midtread_dequantization(quantized_non_uniform, q_matrix)
    
    psnr_non_uniform = psnr_of_frame(img, dequantized_non_uniform)
    
    print(f"PSNR for uniform quantization: {psnr_uniform:.2f} dB")
    print(f"PSNR for non-uniform quantization: {psnr_non_uniform:.2f} dB")
    
    print(f"Rate Uniform: {np.log2(2 / q) * img.shape[0] * img.shape[1]:.2f} bits")
    print(f"Rate Non-Uniform: {np.log2(2 / q0) * num_q0 + np.log2(2 / q1) * num_q1:.2f} bits")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [img, dequantized_uniform, dequantized_non_uniform]
    #titles = ["Original", "Uniform: 40.84 dB", "Matrix-based: 37.97 dB"]
    titles = [
    "Original",
    f"Uniform: {psnr_uniform:.2f} dB",
    f"Matrix-based: {psnr_non_uniform:.2f} dB"
]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()