import numpy as np


# Session 3

def mse_of_frame(original_image: np.ndarray, distorted_image: np.ndarray) -> np.floating | float:
    """
    Calculate the Mean Squared Error (MSE) between the original and distorted images.
    
    :param original_image: The original image.
    :type original_image: np.ndarray
    :param distorted_image: The distorted image.
    :type distorted_image: np.ndarray
    :return: The MSE value.
    :rtype: float
    """
    difference = original_image.astype(float) - distorted_image.astype(float)
    squared_error = difference ** 2
    return np.mean(squared_error)


def psnr_of_frame(original_image: np.ndarray, distorted_image: np.ndarray) -> np.floating | float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original and distorted images.
    The function expects the pixel values to be in the range [0, 1].
    
    :param original_image: The original image.
    :type original_image: np.ndarray
    :param distorted_image: The distorted image.
    :type distorted_image: np.ndarray
    :return: The PSNR value in decibels (dB).
    :rtype: float
    """
    
    mse = mse_of_frame(original_image, distorted_image)

    if mse == 0:
        return float("inf")

    return 10 * np.log10(1.0 / mse)


# Optional Part

def mad_of_frame(original_image: np.ndarray, distorted_image: np.ndarray) -> np.floating | float:
    """
    Calculate the Mean Absolute Deviation (MAD) between the original and distorted images.
    
    :param original_image: The original image.
    :type original_image: np.ndarray
    :param distorted_image: The distorted image.
    :type distorted_image: np.ndarray
    :return: The MAD value.
    :rtype: float
    """
    return 0.0
    
    
def snr_of_frame(original_image: np.ndarray, distorted_image: np.ndarray) -> np.floating | float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) between the original and distorted images.
    The function expects the pixel values to be in the range [0, 1].
    
    :param original_image: The original image.
    :type original_image: np.ndarray
    :param distorted_image: The distorted image.
    :type distorted_image: np.ndarray
    :return: The SNR value in decibels (dB).
    :rtype: float
    """
  
    return 0.0


### Session 4

def calculate_sad(block1: np.ndarray, block2: np.ndarray) -> np.floating | np.ndarray | float:
    """
    Calculate the Sum of Absolute Differences (SAD) between two blocks.
    
    :param block1: The first block.
    :type block1: np.ndarray
    :param block2: The second block.
    :type block2: np.ndarray
    :return: The SAD value.
    :rtype: float
    """
    difference = block1.astype(float) - block2.astype(float)
    absolute_difference = np.abs(difference)
    return np.sum(absolute_difference)