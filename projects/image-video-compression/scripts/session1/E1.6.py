from src.helpers import display_image, update_image, yuv_read_one_frame
import matplotlib.pyplot as plt

filename = "flowergarden_short_cif.yuv"
width = 352
height = 288

for i in range(5):
    Y, U, V = yuv_read_one_frame(
        file_name=filename,
        num_frame=i,
        height=height,
        width=width
    )

    if i == 0:
        img_handle = display_image(Y, interactive=True)
    else:
        update_image(img_handle, Y)

    plt.pause(0.5)

plt.ioff()
plt.show()

