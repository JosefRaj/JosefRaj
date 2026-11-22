''''
Helper functions provided for various tasks of the course.
'''
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np
from bitstring import BitStream
import os


def read_img(file_path: str, grayscale: bool = False) -> np.ndarray:
    """
    Load an image from the specified file path and convert it to a numpy array with float values in the range [0, 1].
    
    :param file_path: Path to the image file.
    :type file_path: str
    :param grayscale: Whether to convert the image to grayscale.
    :type grayscale: bool
    :return: Image as a numpy array with float values in [0, 1].
    :rtype: np.ndarray
    """
    if os.path.splitext(file_path)[1].lower() in (".tif", ".tiff"):
        import tifffile
        img = tifffile.imread(file_path)
    else:
        img = plt.imread(file_path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if grayscale and img.ndim == 3 and img.shape[2] >= 3:
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    return img


def save_img(file_path: str, img: np.ndarray, quality: int = 95):
    """
    Save a numpy array as an image file with the specified quality.
    
    :param file_path: Path to save the image file.
    :type file_path: str
    :param img: Image data as a numpy array with float values in [0, 1].
    :type img: np.ndarray
    :param quality: Quality of the saved image (1-100), where higher is better. Default is 95.
    :type quality: int
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if len(img.shape) == 2:
        plt.imsave(file_path, img, pil_kwargs={'quality': quality}, cmap='gray')
    else:
        plt.imsave(file_path, img, pil_kwargs={'quality': quality})


def display_image(img: np.ndarray, interactive: bool = False, ax=None) -> AxesImage:
    """
    Display an image using matplotlib.
    
    :param img: ndarray representing the image, either grayscale or RGB
    :type img: np.ndarray
    :param interactive: Whether to display the image in interactive mode. Set to True if you plan to update the image later.
    :type interactive: bool
    :return: The AxesImage object representing the displayed image.
    :rtype: AxesImage
    """
    if img.ndim == 1:
        img = img[None, :]
    
    if img.ndim == 2:
        img = img[:, :, None].repeat(3, axis=2)
        
    if interactive:
        plt.ion()

    if ax is None:
        ax = plt.gca()
    im = ax.imshow(img)
    ax.axis('off')
    plt.show()
    return im


def update_image(im: AxesImage, img: np.ndarray):
    """
    Update a matplotlib image with new data.
    
    :param im: The AxesImage object to update.
    :type im: AxesImage
    :param img: The new image data as an ndarray, either grayscale or RGB.  
    :type img: np.ndarray
    """
    if img.ndim == 1:
        img = img[None, :]
    
    if img.ndim == 2:
        img = img[:, :, None].repeat(3, axis=2)

    im.set_data(img)
    plt.draw()


def yuv_read_one_frame(file_name: str, num_frame: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read one frame from a YUV420 formatted file.
    
    :param file_name: Path to the YUV file.
    :param num_frame: Frame number to read (0-indexed).
    :param height: Height of the frame.
    :param width: Width of the frame.
    :return: A tuple containing the Y, U, and V components as ndarrays with type np.float32.
    :rtype: tuple[ndarray[_AnyShape, dtype[np.float32]], ndarray[_AnyShape, dtype[np.float32]], ndarray[_AnyShape, dtype[np.float32]]]
    """
    y_size = height * width
    uv_size = (height // 2) * (width // 2)
    
    with open(file_name, 'rb') as file:
        file.seek(num_frame * (y_size + 2 * uv_size))

        y = np.frombuffer(file.read(y_size), dtype=np.uint8)
        if y.size < y_size:
            raise ValueError(f"Frame {num_frame} is out of range in file {file_name}.")
        y = y.reshape((height, width))
        
        u = np.frombuffer(file.read(uv_size), dtype=np.uint8)
        if u.size < uv_size:
            raise ValueError(f"Frame {num_frame} is out of range in file {file_name}.")
        u = u.reshape((height // 2, width // 2))
        
        v = np.frombuffer(file.read(uv_size), dtype=np.uint8)
        if v.size < uv_size:
            raise ValueError(f"Frame {num_frame} is out of range in file {file_name}.")
        v = v.reshape((height // 2, width // 2))

    y, u, v = y.astype(np.float32) / 255.0, u.astype(np.float32) / 255.0, v.astype(np.float32) / 255.0

    return y, u, v


def yuv_write_one_frame(file_name: str, num_frame: int, y: np.ndarray, u: np.ndarray|None=None, v: np.ndarray|None=None):
    """
    Write one frame to a YUV420 formatted file.
    
    :param file_name: Path to the YUV file.
    :param num_frame: Frame number to write (0-indexed).
    :param y: Y component as a 2D ndarray with type uint8.
    :param u: U component as a 2D ndarray with type uint8.
    :param v: V component as a 2D ndarray with type uint8.
    """
    height, width = y.shape
    uv_height, uv_width = height // 2, width // 2

    if u is None:
        u = np.ones((uv_height, uv_width), dtype=np.float32) * 128 / 255  # Default to mid-gray for U
    if v is None:
        v = np.ones((uv_height, uv_width), dtype=np.float32) * 128 / 255  # Default to mid-gray for V
        
    
    # Clip the values to ensure they are in the range [0, 1] before converting to uint8
    # Convert the values to uint8 and scale to the range [0, 255]
    if y.dtype == np.uint8:
        y = np.clip(y, 0, 255)
    elif y.dtype == np.float32 or y.dtype == np.float64:
        y = np.clip(y, 0, 1)
        y = (y * 255).astype(np.uint8)
    if u.dtype == np.uint8:
        u = np.clip(u, 0, 255)
    elif u.dtype == np.float32 or u.dtype == np.float64:
        u = np.clip(u, 0, 1)
        u = (u * 255).astype(np.uint8)
    if v.dtype == np.uint8:
        v = np.clip(v, 0, 255)
    elif v.dtype == np.float32 or v.dtype == np.float64:
        v = np.clip(v, 0, 1)
        v = (v * 255).astype(np.uint8)

    mode = 'w+b' if not os.path.exists(file_name) else 'a+b'
    with open(file_name, mode) as file:
        file.seek(num_frame * (height * width + 2 * uv_height * uv_width))
        file.write(y.tobytes())
        file.write(u.tobytes())
        file.write(v.tobytes())


def diag_gradient(img: np.ndarray) -> np.ndarray:
    """
    Apply a diagonal gradient to the 2D grayscale input image.
    
    :param img: Input image as a 2D ndarray.
    :type img: np.ndarray
    :return: Image after applying the diagonal gradient.
    :rtype: np.ndarray
    """
    h, w = img.shape[-2:]
    gradient = np.linspace(0, 1, num=h).reshape(h, 1) + np.linspace(0, 1, num=w).reshape(1, w)
    gradient = gradient / gradient.max()
    return (img * gradient).astype(img.dtype)

    
def show_huffman_table(huffman_table: dict):
    """
    Print the Huffman table in a readable format.
    
    :param huffman_table: A dictionary mapping each unique value to its corresponding Huffman code.
    :type huffman_table: dict
    """
    print("\nHuffman Table:")
    print(f"Symbol\t\tCode Word")
    for symbol, code in huffman_table.items():
        print(f"{np.array(symbol)}\t{code}")

        
def show_huffman_tree(huffman_tree: list):
    """
    Print the Huffman tree in a readable format.
    
    :param huffman_tree: A list of tuples representing the Huffman tree, where each tuple contains a frequency and a list of symbols.
    :type huffman_tree: list
    """
    print("\nHuffman Tree:")
    print(f"Frequency\tSymbols")
    for freq, symbols in huffman_tree:
        print(f"{freq:0.7f}\t{[np.array(sym) for sym in symbols]}")

        
def init_bitstream() -> BitStream:
    """
    Initialize an empty bitstream as a bytearray.
    
    :return: An empty bytearray to be used as a bitstream.
    :rtype: bytearray
    """
    return BitStream()


def bitstream_append(bitstream: BitStream, bits: str):
    """
    Append a string of bits to the given bitstream.
    
    :param bitstream: The BitStream object to which the bits will be appended.
    :type bitstream: BitStream
    :param bits: A string representing the bits to append (e.g., '1010').
    :type bits: str
    """
    bitstream.append(f"0b{bits}")

    
def bitstream_read(bitstream: BitStream, num_bits: int) -> str:
    """
    Read a specified number of bits from the bitstream and return them as a string.
    
    :param bitstream: The BitStream object from which to read the bits.
    :type bitstream: BitStream
    :param num_bits: The number of bits to read from the bitstream.
    :type num_bits: int
    :return: A string representing the bits read from the bitstream (e.g., '1010').
    :rtype: str
    """
    if bitstream.pos + num_bits > bitstream.len:
        return ""
    
    bits = bitstream.read(num_bits).bin
    return bits


def bitstream_to_ndarray(bitstream: BitStream) -> np.ndarray:
    """
    Convert a BitStream object to a packed numpy array.
    
    :param bitstream: The BitStream object to convert.
    :type bitstream: BitStream
    :return: A numpy array containing the bits from the bitstream.
    :rtype: np.ndarray
    """
    return np.packbits(np.array(list(map(int, bitstream.bin)), dtype=np.uint8))


def ndarray_to_bitstream(arr: np.ndarray) -> BitStream:
    """
    Convert a numpy array of bits (0s and 1s) to a BitStream object.
    
    :param arr: A numpy array containing bits (0s and 1s).
    :type arr: np.ndarray
    :return: A BitStream object representing the bits in the input array.
    :rtype: BitStream
    """
    
    arr = np.unpackbits(arr).flatten()
    
    if not np.all(np.isin(arr, [0, 1])):
        raise ValueError("Input array must contain only 0s and 1s.")
    
    bit_string = ''.join(map(str, arr.flatten()))
    return BitStream(f"0b{bit_string}")


def shape_pdf(x: np.ndarray) -> np.ndarray:
    """
    Create a pdf with the same shape as x, but with only the top 90% of values.
    :param x: Input
    :type x: np.ndarray
    :return: PDF with the same shape as x, but with only the top 90% of values. 
    :rtype: np.ndarray
    """
    
    y = np.where(np.abs(x) >= 0.9 * np.max(np.abs(x)), x, 0)
    y = y / np.abs(y).sum() * np.abs(x).sum()
    return y

    
def design_quantizer_matrix(q0: float, q1: float) -> tuple[np.ndarray, int, int]:
    """
    Design a quantizer matrix for the flowerbed img based on the given parameters q0 and q1.
    
    :param q0: The first parameter for designing the quantizer matrix.
    :type q0: float
    :param q1: The second parameter for designing the quantizer matrix.
    :type q1: float
    :return: A 4x4 quantizer matrix designed based on the input parameters.
    :rtype: np.ndarray
    """
    segm = np.load('./flower_segm.npy')
    
    quantizer_matrix = np.where(segm == 0, q0, q1)

    num_q0 = np.sum(segm == 0)
    num_q1 = np.sum(segm == 1)
    
    return quantizer_matrix, num_q0, num_q1

    
def create_mv_test_matrices():
    """"
    Create two 8x8 matrices for testing motion estimation algorithms.
    The first matrix (y0) represents the reference image, and the second matrix (y1) represents the current image with some motion applied to it.
    :return: A tuple containing the reference image (y0) and the current image (y1) as 8x8 numpy arrays.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    y0 = np.array([[99 , 106 , 136 , 142 , 131 , 120 , 124 , 126 ],
            [107 , 118 , 126 , 129 , 131 , 132 , 135 , 128 ],
            [117 , 136 , 132 , 122 , 119 , 125 , 132 , 134 ],
            [122 , 133 , 136 , 132 , 125 , 125 , 121 , 117 ],
            [133 , 125 , 124 , 125 , 131 , 130 , 118 , 89 ],
            [138 , 134 , 113 , 88 , 69 , 63 , 91 , 130 ],
            [69 , 76 , 93 , 115 , 125 , 132 , 188 , 248 ],
            [97 , 113 , 169 , 228 , 255 , 255 , 230 , 208 ]])
    
    y1 = np.array([[111 , 138 , 139 , 129 , 121 , 122 , 127 , 130 ],
            [122 , 127 , 129 , 134 , 135 , 134 , 127 , 120 ],
            [135 , 130 , 124 , 122 , 128 , 135 , 136 , 130 ],
            [134 , 136 , 133 , 123 , 123 , 123 , 119 , 115 ],
            [124 , 125 , 128 , 130 , 130 , 113 , 76 , 45 ],
            [127 , 105 , 78 , 66 , 72 , 108 , 139 , 138 ],
            [84 , 104 , 123 , 125 , 145 , 215 , 255 , 254 ],
            [132 , 201 , 247 , 255 , 251 , 217 , 206 , 211 ]])

    return y0, y1


def create_synthetic_images(height: int, width:int, dx:int, dy:int, foreground_size: tuple[int, int] = (32, 32), solid_background: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Create two synthetic images for testing motion estimation algorithms. 
    The images contain the same noise like background with a noise like foreground object that is shifted by (dx, dy) pixels in the current image compared to the reference image.
    
    :param height: The height of the images.
    :type height: int
    :param width: The width of the images.
    :type width: int
    :param dx: The horizontal shift applied to create the current image from the reference image.
    :type dx: int
    :param dy: The vertical shift applied to create the current image from the reference image.
    :type dy: int
    :param foreground_size: The size of the foreground object.
    :type foreground_size: tuple[int, int]
    :return: A tuple containing the reference image and the current image as numpy arrays.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    
    current = np.random.rand(height, width) * 0.3 if not solid_background else np.zeros((height, width))
    reference = current.copy()

    foreground = np.random.rand(*foreground_size) * 0.5 + 0.5
    #foreground = diag_gradient(foreground)
    
    start_y = height // 2 - foreground_size[0] // 2
    start_x = width // 2 - foreground_size[1] // 2 
    
    current[start_y:start_y + foreground_size[0], start_x:start_x + foreground_size[1]] = foreground
    reference[start_y + dy:start_y + dy + foreground_size[0], start_x + dx:start_x + dx + foreground_size[1]] = foreground

    return current, reference


def plot_motion_vectors(motion_vectors: np.ndarray, height: int, width: int, blocksize: int):
    """
    Plot the motion vectors on a grid corresponding to the blocks of the image.
    
    :param motion_vectors: A 2D array of shape (num_blocks_y, num_blocks_x, 2) containing the motion vectors (dy, dx) for each block.
    :type motion_vectors: np.ndarray
    :param blocksize: The size of each block (assumed to be square).
    :type blocksize: int
    :param search_range: The range of the search area used for motion estimation.
    :type search_range: int
    """
    num_blocks_y, num_blocks_x = height // blocksize, width // blocksize
    motion_vectors = motion_vectors.reshape(num_blocks_y, num_blocks_x, 2)
    plt.figure(figsize=(10, 10))
    
    # Create a grid of block centers
    X, Y = np.meshgrid(np.arange(num_blocks_x) * blocksize + blocksize // 2, 
                       np.arange(num_blocks_y) * blocksize + blocksize // 2)
    
    U = motion_vectors[:, :, 1]  # dx
    V = motion_vectors[:, :, 0]  # dy
    
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1.0, color='r')
    
    # Plot the grid lines for blocks
    for x in range(0, width, blocksize):
        plt.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in range(0, height, blocksize):
        plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)
        
    plt.ylim(height, 0)
    plt.xlim(0, width)
    plt.ylabel("Y-axis")
    plt.xlabel("X-axis")

    plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    
    plt.show()
    

def plot_sad_map(sad_map: np.ndarray, search_range: int):
    """
    Plot the SAD map for a block in motion estimation.
    
    :param sad_map: A 2D array containing the SAD values for each candidate motion vector.
    :type sad_map: np.ndarray
    :param search_range: The range of the search area used for motion estimation.
    :type search_range: int
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    extent = (-search_range - 0.5, search_range + 0.5, search_range + 0.5, -search_range - 0.5)
    
    im = ax.imshow(sad_map, extent=extent, origin='upper', cmap='magma')
    ax.figure.colorbar(im, ax=ax, label='SAD Value', shrink=0.8)

    for i in range(sad_map.shape[0]):
        for j in range(sad_map.shape[1]):
            sad = sad_map[i, j]
            ax.text(j - search_range, i - search_range, f"{sad:.1f}", ha='center', va='center', color='white' if sad < np.percentile(sad_map, 50) else 'black', fontsize=11)
    ax.set_title('SAD Map')
    ax.set_xlabel('dx')
    ax.set_ylabel('dy')

    plt.show()

    
def get_quantization_matrix(qp: int, transform_blocksize: tuple) -> np.ndarray:
    """
    Get the quantization matrix for a given quantization parameter and transform block size.
    
    :param qp: The quantization parameter.
    :param transform_blocksize: The size of the transform block.
    :return: The quantization matrix.
    """
    qm = np.ones(transform_blocksize) * qp * transform_blocksize[0] / (2 * 256)
    qm[0, 0] = transform_blocksize[0] / 255 
    return qm
    

def huffman_table_dict_to_bitstream(huffman_table_dict: dict, blocksize: int = 8) -> BitStream:
    """
    Convert a Huffman table from dictionary format to a bitstream for storage in the encoded file. Can be decoded later using the `huffman_table_dict_from_bitstream` function.
    
    The bitstream is structured as follows:
    - 4 bits for the number of bits used to encode the run values (run_length_bits)
    - 4 bits for the number of bits used to encode the level values (level_bits)
    - 4 bits for the level offset (level_offset)
    - 5 bits for the number of bits to encode code lengths (code_length_bits)
    For each symbol in the Huffman table:
    - run_length_bits bits for the run value
    - level_bits bits for the level value
    - code_length_bits bits for the length of the Huffman code
    - The actual Huffman code bits (variable length)
    
    :param huffman_table_dict: A dictionary mapping each unique value to its corresponding Huffman code.
    :type huffman_table_dict: dict
    :return: A bitstream representation of the Huffman table that can be used in the encoded file.
    :rtype: BitStream
    """

    bitstream = init_bitstream()
    symbols = list(huffman_table_dict.keys())
    run_lengths = [symbol[0] for symbol in symbols]
    run_levels = [symbol[1] for symbol in symbols]
    code_words = list(huffman_table_dict.values())
    max_length_code = max(len(code) for code in code_words)
    
    level_range = max(run_levels) - min(run_levels) + 1
    offset = min(run_levels) 
    
    bits_to_encode_run = np.ceil(np.log2(blocksize ** 2 + 1))
    bits_to_encode_level = np.ceil(np.log2(level_range))
    bits_to_encode_code_length = np.ceil(np.log2(max_length_code + 2))
    
    bitstream_append(bitstream, np.binary_repr(int(bits_to_encode_run), width=4))
    bitstream_append(bitstream, np.binary_repr(int(bits_to_encode_level), width=5))
    bitstream_append(bitstream, np.binary_repr(int(offset), width=11))
    bitstream_append(bitstream, np.binary_repr(int(bits_to_encode_code_length), width=5))
    
    for run, level, code in zip(run_lengths, run_levels, code_words):
        bitstream_append(bitstream, np.binary_repr(int(run), width=int(bits_to_encode_run)))
        bitstream_append(bitstream, np.binary_repr(int(level - offset), width=int(bits_to_encode_level)))
        bitstream_append(bitstream, np.binary_repr(len(code), width=int(bits_to_encode_code_length)))
        bitstream_append(bitstream, code)
    bitstream.pos = 0  # Reset the position to the start of the bitstream for reading later
    #print(f"Encoded Huffman table length: {len(bitstream)} bits/ {len(bitstream) / 8 / 1000} kilobytes")
    return bitstream


def huffman_table_bitstream_to_dict(bitstream: BitStream) -> dict:
    """
    Convert a Huffman table from a bitstream back to dictionary format. This is the inverse of the `huffman_table_dict_to_bitstream` function.
    
    :param bitstream: A bitstream representation of the Huffman table.
    :type bitstream: BitStream
    :return: A dictionary mapping each unique value to its corresponding Huffman code.
    :rtype: dict
    """
    bits_to_encode_run = bitstream.read(4).int
    bits_to_encode_level = bitstream.read(5).int
    offset = bitstream.read(9).int
    bits_to_encode_code_length = bitstream.read(5).int

    huffman_table_dict = {}
    
    while bitstream.pos < bitstream.len:
        run = bitstream.read(bits_to_encode_run).int
        level = bitstream.read(bits_to_encode_level).uint + offset
        code_length = bitstream.read(bits_to_encode_code_length).uint
        code = bitstream.read(code_length).bin
        huffman_table_dict[(np.int64(run), np.int64(level))] = code
    
    return huffman_table_dict


def mv_huffman_table_dict_to_bitstream(huffman_table_dict: dict) -> BitStream:
    """
    Convert a Huffman table from dictionary format to a bitstream for storage in the encoded file. Can be decoded later using the `huffman_table_dict_from_bitstream` function.
    
    The bitstream is structured as follows:
    - 4 bits for the number of bits used to encode the run values (run_length_bits)
    - 4 bits for the number of bits used to encode the level values (level_bits)
    - 4 bits for the level offset (level_offset)
    - 5 bits for the number of bits to encode code lengths (code_length_bits)
    For each symbol in the Huffman table:
    - run_length_bits bits for the run value
    - level_bits bits for the level value
    - code_length_bits bits for the length of the Huffman code
    - The actual Huffman code bits (variable length)
    
    :param huffman_table_dict: A dictionary mapping each unique value to its corresponding Huffman code.
    :type huffman_table_dict: dict
    :return: A bitstream representation of the Huffman table that can be used in the encoded file.
    :rtype: BitStream
    """

    bitstream = init_bitstream()
    symbols = list(huffman_table_dict.keys())
    dys = [symbol[0] for symbol in symbols]
    dxs = [symbol[1] for symbol in symbols]
    code_words = list(huffman_table_dict.values())
    max_length_code = max(len(code) for code in code_words)

    subpixel_precision = False

    # If any dys or dx are floats, double all of them to store them as integers
    if any(isinstance(dy, float) for dy in dys) or any(isinstance(dx, float) for dx in dxs):
        subpixel_precision = True
        dys = [dy * 2 for dy in dys]
        dxs = [dx * 2 for dx in dxs]

    max_dy = max(dys)
    max_dx = max(dxs)
    min_dy = min(dys)
    min_dx = min(dxs)
    
    bits_to_encode_dy = np.ceil(np.log2(max_dy - min_dy + 1))
    bits_to_encode_dx = np.ceil(np.log2(max_dx - min_dx + 1))
    bits_to_encode_code_length = np.ceil(np.log2(max_length_code + 2))
    
    # Write subpixel precision flag
    bitstream_append(bitstream, np.binary_repr(int(subpixel_precision), width=1))
    bitstream_append(bitstream, np.binary_repr(int(bits_to_encode_dy), width=4))
    bitstream_append(bitstream, np.binary_repr(int(bits_to_encode_dx), width=4))
    bitstream_append(bitstream, np.binary_repr(int(bits_to_encode_code_length), width=5))
    
    for dy, dx, code in zip(dys, dxs, code_words):
        bitstream_append(bitstream, np.binary_repr(int(dy), width=int(bits_to_encode_dy)))
        bitstream_append(bitstream, np.binary_repr(int(dx), width=int(bits_to_encode_dx)))
        bitstream_append(bitstream, np.binary_repr(len(code), width=int(bits_to_encode_code_length)))
        bitstream_append(bitstream, code)
    bitstream.pos = 0  # Reset the position to the start of the bitstream for reading later
    #print(f"Encoded Huffman table length: {len(bitstream)} bits/ {len(bitstream) / 8 / 1000} kilobytes")
    return bitstream


def mv_huffman_table_bitstream_to_dict(bitstream: BitStream) -> dict:
    """
    Convert a Huffman table from a bitstream back to dictionary format. This is the inverse of the `huffman_table_dict_to_bitstream` function.
    
    :param bitstream: A bitstream representation of the Huffman table.
    :type bitstream: BitStream
    :return: A dictionary mapping each unique value to its corresponding Huffman code.
    :rtype: dict
    """
    subpixel_precision = bitstream.read(1).bool
    bits_to_encode_dy = bitstream.read(4).int
    bits_to_encode_dx = bitstream.read(4).int
    bits_to_encode_code_length = bitstream.read(5).int

    huffman_table_dict = {}
    
    while bitstream.pos < bitstream.len:
        dy = bitstream.read(bits_to_encode_dy).int
        dx = bitstream.read(bits_to_encode_dx).int
        code_length = bitstream.read(bits_to_encode_code_length).uint
        code = bitstream.read(code_length).bin
        if subpixel_precision:
            dy = dy / 2
            dx = dx / 2
            huffman_table_dict[(np.float64(dy), np.float64(dx))] = code
        else:
            huffman_table_dict[((np.int64(dy)), np.int64(dx))] = code
    
    return huffman_table_dict