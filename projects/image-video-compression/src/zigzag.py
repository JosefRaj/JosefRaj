import numpy as np


def generate_zigzag_permutation_matrix(width: int, height: int) -> np.ndarray:
    """
    Generate a zigzag permutation matrix for a given width and height.
    
    :param width: Width of the matrix.
    :param height: Height of the matrix.
    :return: 2D numpy array representing the zigzag permutation matrix.
    """
    permutation_matrix = np.empty((height, width), dtype=int)
    index = 0

    for diagonal in range(height + width - 1):
        first_row = max(0, diagonal - width + 1)
        last_row = min(height - 1, diagonal)

        if diagonal % 2 == 0:
            rows = range(last_row, first_row - 1, -1)
        else:
            rows = range(first_row, last_row + 1)

        for row in rows:
            column = diagonal - row
            permutation_matrix[row, column] = index
            index += 1

    return permutation_matrix


class ZigZagScanner:
    def __init__(self, width: int, height: int):
        """
        Class to perform zigzag scanning on blocks of a given width and height.
        """
        self.width = width
        self.height = height

        self.permutation_matrix = generate_zigzag_permutation_matrix(
            width, height
        )
        self.scan_order = np.argsort(self.permutation_matrix.ravel())

    
    def encode_block(self, block: np.ndarray) -> np.ndarray:
        """
        Scan the input block in zigzag order and return a 1D array of the scanned values.
        
        :param block: 2D numpy array representing the block to be scanned.
        :return: 1D numpy array of the scanned values in zigzag order.
        """
        flat_block = block.reshape(
            block.shape[:-2] + (self.height * self.width,)
        )

        return flat_block[..., self.scan_order]


    def decode_block(self, zigzag_scanned: np.ndarray) -> np.ndarray:
        """
        Reconstruct the original block from the scanned values in zigzag order.
        
        :param zigzag_scanned: 1D numpy array of the scanned values in zigzag order.
        :return: 2D numpy array representing the reconstructed block.
        """
        return zigzag_scanned[..., self.permutation_matrix]


    def encode_image(self, input_image: np.ndarray) -> np.ndarray:
        """u
        Apply zigzag scanning to the entire image by processing it in blocks of the specified width and height.
        
        :param image: 2D numpy array representing the input image.
        :return: 2D numpy array representing the zigzag scanned image.
        """
        image_height, image_width = input_image.shape

        block_rows = image_height // self.height
        block_columns = image_width // self.width

        blocks = input_image.reshape(
            block_rows,
            self.height,
            block_columns,
            self.width,
        ).transpose(0, 2, 1, 3)

        scanned_blocks = self.encode_block(blocks)

        return scanned_blocks.reshape(-1, self.height * self.width)
        
    def decode_image(self, zigzag_scanned: np.ndarray, shape: tuple) -> np.ndarray:
        """
        Reconstruct the original image from the zigzag scanned values 
        by processing it in blocks of the specified width and height.
        
        :param zigzag_scanned: 2D numpy array representing the zigzag scanned image.
        :param shape: Tuple representing the shape of the reconstructed image.
        :return: 2D numpy array representing the reconstructed image.
        """
        image_height, image_width = shape

        block_rows = image_height // self.height
        block_columns = image_width // self.width

        blocks = self.decode_block(zigzag_scanned)

        blocks = blocks.reshape(
            block_rows,
            block_columns,
            self.height,
            self.width,
        )

        return blocks.transpose(0, 2, 1, 3).reshape(shape)