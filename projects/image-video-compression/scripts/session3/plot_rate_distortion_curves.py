from src.metrics import psnr_of_frame
from src.helpers import read_img, save_img, display_image, update_image
import matplotlib.pyplot as plt
import os 


def main():

    # TODO: Load the original image
    original_image = read_img("./flowergarden_cif_frame1.tif")

    # TODO: Calculate the number of pixels in the image
    num_pixels = original_image.shape[0] * original_image.shape[1]

    rates = []
    distortion = []

    # Prepare plots
    ##########################################################################
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    ax[0].set_title("Original Image")

    ax[1].set_title("Rate-Distortion Curve")
    ax[1].set_xlabel("Rate (bits per pixel)")
    ax[1].set_ylabel("Distortion (PSNR in dB)")
    ax[1].grid()
    ax[1].set_xlim(0, 7)
    ax[1].set_ylim(15, 60)

    im = display_image(original_image, interactive=True, ax=ax[0])
    plt.waitforbuttonpress() 
    ###########################################################################

    for quality in range(100, -1, -1):
        # TODO: Save the distorted image, calculate its filesize, load it and compute the PSNR
        save_img(f"./flowergarden_cif_frame1.jpeg", original_image, quality=quality)
        compressed_filesize = os.path.getsize(f"./flowergarden_cif_frame1.jpeg")
        compressed_img = read_img(f"./flowergarden_cif_frame1.jpeg", grayscale=True)
        
        rates.append(compressed_filesize * 8 / num_pixels)
        distortion.append(psnr_of_frame(original_image, compressed_img))
        
        if quality % 10 == 0:
            ax[0].set_title(f"Compressed Image (Quality={quality})")
            update_image(im, compressed_img)
            ax[1].plot(rates, distortion)
            ax[1].scatter(rates[-1], distortion[-1], color='red')
            plt.waitforbuttonpress()




if __name__ == "__main__":
    main()
