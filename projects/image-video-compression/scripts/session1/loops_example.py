
import numpy as np
import matplotlib.pyplot as plt

from src.helpers import display_image, read_img

def main():
    
    img = read_img('flowergarden_cif_frame1.tif')
    print(img.shape)
    
    n = 4
    for i in range(n):
        for j in range(n):
            rows = (np.arange(0, 288 / n) + (i) * 288 / n).astype(int)
            cols = (np.arange(0, 352 / n) + (j) * 352 / n).astype(int)
            display_image(img[np.ix_(rows, cols)], interactive=True)
            plt.waitforbuttonpress()      
    
    
if __name__ == "__main__":
    main()

   