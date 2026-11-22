from src.helpers import read_img, display_image, diag_gradient
from src.utils import blockwise_processing
import matplotlib.pyplot as plt

img = read_img("flowergarden_cif_frame1.tif")

result = diag_gradient(img)
display_image(result)
plt.show()

result_block =blockwise_processing(img, [50,100], diag_gradient)

display_image(result_block)
plt.show()