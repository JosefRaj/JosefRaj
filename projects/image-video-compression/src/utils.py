import numpy as np

def my_sin(t: float, omega: float, phi: float = 0) -> float:

    '''
    This function returns the sine value at given omega and t.
    
    :param t: time
    :type t: float
    
    :param omega: frequency
    :type omega: float

    :param phi: phase (default = 0)
    :type omega: float

    :return: sin value of omega*t + phi
    :rtype: float
    '''

    return np.sin(omega*t + phi)

# Session 1
def blockwise_processing(input_image: np.ndarray, blocksize: list[int], func) -> np.ndarray:
    """
    Apply a function to non-overlapping blocks of the input matrix. Pads the input with zeros to make its dimensions multiples of block_size, then processes each block with the provided function.
    Assembles the processed blocks back into a single output matrix of the same size as the input.
    
    :param input_mat: 2D numpy array to be processed.
    :param block_size: Size of the blocks [height, width].
    :param func: Function to apply to each block. It should take a 2D numpy array as input and return a 2D numpy array of the same shape.
    :return: Processed 2D numpy array.
    """

    
    h, w = input_image.shape

    # Calculate padded dimensions
    padded_h = h + (-h % blocksize[0]) # TODO: calculate the image height that we need to pad to 
    padded_w = w + (-w % blocksize[1]) # TODO: calculate the image width that we need to pad to


    # Create padded input matrix
    padded_input = np.zeros((padded_h, padded_w), dtype=input_image.dtype)
    # TODO: copy input_mat into the top-left corner of padded_input
    padded_input [:h, :w] = input_image

    # Hint: use array slicing
    
    # Process all blocks at once in a Pythonic way by reshaping the padded input into blocks 
    # and applying the function the whole tensor
    # Reshape the padded input into blocks
    blocked = padded_input.reshape(padded_h // blocksize[0], blocksize[0], padded_w // blocksize[1], blocksize[1])
    blocked = blocked.transpose(0, 2, 1, 3)  # Rearrange axes to have blocks in the first two dimensions
    
    # Apply the function to each block
    blocked = func(blocked)
    # Reshape back to the original padded image shape
    output_image = blocked.transpose(0, 2, 1, 3).reshape(padded_h, padded_w)
            
    # TODO: Remove padding and return the output matrix of the original size
            
    return output_image[:h, :w]


