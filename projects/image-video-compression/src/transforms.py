import numpy as np
from scipy.fft import dct, idct, dctn, idctn
from .utils import blockwise_processing


def dct_2d(block: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Discrete Cosine Transform (DCT) of a given block using the formula.
    
    :param block: 2D numpy array representing the block.
    :return: 2D numpy array representing the DCT of the block.
    """
    return dctn(block, type=2, axes=(-2, -1), norm="ortho")


def idct_2d(block: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Inverse Discrete Cosine Transform (IDCT) of a given block using the formula.
    
    :param block: 2D numpy array representing the DCT coefficients of the block.
    :return: 2D numpy array representing the reconstructed block.
    """
    return idctn(block, type=2, axes=(-2, -1), norm="ortho")


def blockbased_dct(input_image: np.ndarray, blocksize: int) -> np.ndarray:
    """
    Apply block-based DCT to the input image using the specified block size.
    
    :param input_image: 2D numpy array representing the input image.
    :param blocksize: Size of the blocks [height, width].
    :return: 2D numpy array representing the DCT of the input image.
    """
    height, width = input_image.shape

    padded_image = np.pad(
        input_image,
        ((0, -height % blocksize), (0, -width % blocksize)),
        mode="constant",
    )

    return blockwise_processing(
        padded_image,
        [blocksize, blocksize],
        dct_2d,
    )



def blockbased_idct(input_image: np.ndarray, blocksize: int) -> np.ndarray:
    """
    Apply block-based IDCT to the input image using the specified block size.
    
    :param input_image: 2D numpy array representing the DCT coefficients of the input image.
    :param blocksize: Size of the blocks [height, width].
    :return: 2D numpy array representing the reconstructed image from the DCT coefficients.
    """
    return blockwise_processing(
        input_image,
        [blocksize, blocksize],
        idct_2d,
    )