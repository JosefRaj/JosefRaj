import numpy as np
from .utils import blockwise_processing

def midtread_quantization(input, stepsize) -> np.ndarray:
    """
    Quantize the input image using a simple uniform quantization scheme.
    
    :param input: Input image as a numpy array with float values in [0, 1].
    :type input: np.ndarray
    :param stepsize: Step size for quantization. Can be a single float for uniform quantization across all pixels, or a numpy array of the same shape as input for pixel-wise quantization.
    :type stepsize: float | np.ndarray
    :return: Quantization levels as a numpy array with integer values.
    :rtype: np.ndarray
    """
    # TODO: Implement the midtread quantization algorithm
    return np.round(input / stepsize)
    

def midtread_dequantization(quantized, stepsize) -> np.ndarray:
    """
    Dequantize the quantized image using the same step size.
    
    :param quantized: Quantized image as a numpy array with integer values.
    :type quantized: np.ndarray
    :param stepsize: Step size used for quantization. Can be a single float for uniform quantization across all pixels, or a numpy array of the same shape as quantized for pixel-wise quantization.
    :type stepsize: float | np.ndarray
    :return: Dequantized image as a numpy array with float values in [0, 1].
    :rtype: np.ndarray
    """
    # TODO: Implement the midtread dequantization algorithm
    return quantized * stepsize


class Quantizer:
    def __init__(self, stepsize: float | np.ndarray, quantization_type: str = 'midtread', blocksize=None):
        """
        Initialize the Quantizer with the specified step size and quantization type.

        :param stepsize: Step size for quantization.
        :type stepsize: float | np.ndarray
        :param quantization_type: Type of quantization ('uniform' or 'non-uniform').
        :type quantization_type: str
        """
        self.stepsize = stepsize
        self.blocksize = blocksize

        if quantization_type == "midtread":
            self.quantization_function = midtread_quantization
            self.dequantization_function = midtread_dequantization
        else:
            raise ValueError(
                f"Unsupported quantization type: {quantization_type}"
            )

    def quantize(self, input: np.ndarray) -> np.ndarray:
        """
        Quantize the input image using the specified quantization scheme.

        :param input: Input image as a numpy array with float values in [0, 1].
        :type input: np.ndarray
        :return: Quantized image as a numpy array with integer values.
        :rtype: np.ndarray
        """
        if self.blocksize is None:
            return self.quantization_function(input, self.stepsize)
        else:
            return blockwise_processing(
                input,
                self.blocksize,
                lambda block: self.quantization_function(block, self.stepsize)
            )

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """
        Dequantize the quantized image using the same quantization scheme.

        :param quantized: Quantized image as a numpy array with integer values.
        :type quantized: np.ndarray
        :return: Dequantized image as a numpy array with float values in [0, 1].
        :rtype: np.ndarray
        """
        if self.blocksize is None:
            return self.dequantization_function(quantized, self.stepsize)
        else:
            return blockwise_processing(
                quantized,
                self.blocksize,
                lambda block: self.dequantization_function(block, self.stepsize)
            )
        