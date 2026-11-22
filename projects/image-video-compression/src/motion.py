import numpy as np

from .metrics import calculate_sad

def get_motion_vector_for_block_optimal(padded_input: np.ndarray, padded_previous_image, blockstart_x, blockstart_y, blocksize, search_range) -> tuple[np.ndarray, float]:
    """
    Calculate the optimal motion vector for a block in the current frame by searching within a specified range in the previous frame.
    
    :param padded_input: The padded current image.
    :type padded_input: np.ndarray
    :param padded_previous_image: The padded previous image.
    :type padded_previous_image: np.ndarray
    :param blockstart_x: The x-coordinate of the top-left corner of the block in the current image.
    :type blockstart_x: int
    :param blockstart_y: The y-coordinate of the top-left corner of the block in the current image.
    :type blockstart_y: int
    :param blocksize: The size of the block (assumed to be square).
    :type blocksize: int
    :param search_range: The range (in pixels) to search for matching blocks in the previous image.
    :type search_range: int
    :return: The optimal motion vector as a numpy array [dy, dx].
    :rtype: np.ndarray
    """
    current_block = padded_input[
        blockstart_y : blockstart_y + blocksize,
        blockstart_x : blockstart_x + blocksize
    ]

    best_sad = float("inf")
    best_vector = np.array([0, 0])

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            reference_y = blockstart_y + dy
            reference_x = blockstart_x + dx

            reference_block = padded_previous_image[
                reference_y : reference_y + blocksize,
                reference_x : reference_x + blocksize
            ]

            sad = calculate_sad(current_block, reference_block)

            if sad < best_sad:
                best_sad = sad
                best_vector = np.array([dy, dx])
                
    return best_vector, best_sad  # Placeholder for motion vector and SAD value
    
class MotionCompensator:
    def __init__(self, blocksize: int, search_range: int):
        """
        Initialize the MotionCompensator with the specified block size and search range.
        
        :param blocksize: The size of the blocks (assumed to be square).
        :type blocksize: int
        :param search_range: The range (in pixels) to search for matching blocks in the previous image.
        :type search_range: int
        :param estimation_method: The method to use for motion vector estimation.
        :type estimation_method: str
        :param intra_decision: Whether to use intra-block cost for motion estimation.
        :type intra_decision: bool
        """
        self.blocksize = blocksize
        self.search_range = search_range
        self.estimate_for_block = get_motion_vector_for_block_optimal

    def estimate_motion(self, current_image: np.ndarray, previous_image: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Estimate the motion vectors for each block in the current image based on the previous image.
        
        :param current_image: The current image as a numpy array with float values in [0, 1].
        :type current_image: np.ndarray
        :param previous_image: The previous image as a numpy array with float values in [0, 1].
        :type previous_image: np.ndarray
        :return: An array of motion vectors for each block in the current image.
        :rtype: np.ndarray
        """
        padded_current = np.pad(
            current_image, self.search_range, mode="edge"
        )
        padded_previous = np.pad(
            previous_image, self.search_range, mode="edge"
        )

        height, width = current_image.shape
        num_block_rows = height // self.blocksize
        num_block_cols = width // self.blocksize

        motion_vectors = np.zeros(
            (num_block_rows, num_block_cols, 2),
            dtype=int,
        )
                
        for block_row in range(num_block_rows):
            for block_col in range(num_block_cols):
                y = block_row * self.blocksize + self.search_range
                x = block_col * self.blocksize + self.search_range

                vector, _ = self.estimate_for_block(
                    padded_current,
                    padded_previous,
                    blockstart_x=x,
                    blockstart_y=y,
                    blocksize=self.blocksize,
                    search_range=self.search_range,
                )

                motion_vectors[block_row, block_col] = vector

        return motion_vectors.reshape(-1, 2)
    
    def compensate_motion(self, previous_image: np.ndarray, motion_vectors: np.ndarray) -> np.ndarray:
        """
        Compensate the motion in the previous image using the estimated motion vectors to create a predicted current image.
        
        :param previous_image: The previous image as a numpy array with float values in [0, 1].
        :type previous_image: np.ndarray
        :param motion_vectors: An array of motion vectors for each block in the current image.
        :type motion_vectors: np.ndarray
        :return: The predicted current image after motion compensation.
        :rtype: np.ndarray
        """
        padded_previous = np.pad(
            previous_image, self.search_range, mode="edge"
        )
        predicted_image = np.zeros_like(previous_image)

        height, width = previous_image.shape
        num_block_rows = height // self.blocksize
        num_block_cols = width // self.blocksize

        for block_row in range(num_block_rows):
            for block_col in range(num_block_cols):
                y = block_row * self.blocksize
                x = block_col * self.blocksize

                index = block_row * num_block_cols + block_col
                dy, dx = motion_vectors[index]

                ref_y = y + self.search_range + dy
                ref_x = x + self.search_range + dx

                predicted_image[
                    y:y + self.blocksize,
                    x:x + self.blocksize,
                ] = padded_previous[
                    ref_y:ref_y + self.blocksize,
                    ref_x:ref_x + self.blocksize,
                ]        

        return predicted_image  # Placeholder for the predicted current image