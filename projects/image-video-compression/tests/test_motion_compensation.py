import unittest
import numpy as np

from src.helpers import yuv_read_one_frame, display_image
from src.motion import MotionCompensator

class TestMotionCompensation(unittest.TestCase):
    def test_motion_estimation(self):
        # Create a simple test case with a known motion vector
        current_image = np.zeros((16, 16))
        previous_image = np.zeros((16, 16))
        
        # Create a block in the current image
        current_image[4:8, 4:8] = 1.0
        
        # Create a matching block in the previous image with a known offset
        previous_image[6:10, 6:10] = 1.0
        
        # Initialize the MotionCompensator
        motion_compensator = MotionCompensator(blocksize=4, search_range=4)
        
        # Estimate motion vectors
        motion_vectors = motion_compensator.estimate_motion(current_image, previous_image)
        
        # Check if the estimated motion vector is as expected (dy=2, dx=2)
        expected_motion_vector = np.array([2, 2])
        self.assertTrue(np.array_equal(motion_vectors[5, :], expected_motion_vector), f"Expected: {expected_motion_vector}, Got: {motion_vectors[5, :]}")
        
    def test_motion_compensation(self):
        # Create a simple test case with a known motion vector
        previous_image = np.zeros((16, 16))
        previous_image[6:10, 6:10] = 1.0
        
        # Known motion vector (dy=2, dx=2) for block starting at (4, 4), block 6
        motion_vectors = np.zeros((16, 2), dtype=int)
        motion_vectors[5, :] = np.array([2, 2])  # Block at (4, 4) moves to (6, 6)
        motion_vectors[6, :] = np.array([-2, -2])  # Move to any empty block for testing
        motion_vectors[9, :] = np.array([-2, -2])  # Move to any empty block for testing
        motion_vectors[10, :] = np.array([2, 2])  # Move to any empty block for testing
        
        
        # Initialize the MotionCompensator
        motion_compensator = MotionCompensator(blocksize=4, search_range=4)
        
        # Compensate motion
        predicted_image = motion_compensator.compensate_motion(previous_image, motion_vectors)
        
        # Check if the predicted image matches the expected result
        expected_predicted_image = np.zeros((16, 16))
        expected_predicted_image[4:8, 4:8] = 1.0
        
        self.assertTrue(np.array_equal(predicted_image, expected_predicted_image), f"Expected:\n{expected_predicted_image}\nGot:\n{predicted_image}")
        
    def test_motion_compensation_image(self):
        # load two frames and compensate with known motion vectors
        frame1 = yuv_read_one_frame('flowergarden_short_cif.yuv', 0, 288, 352)[0]
        frame2 = yuv_read_one_frame('flowergarden_short_cif.yuv', 1, 288, 352)[0]
        motion_vectors = np.load('motion_vectors_flower.npy')

        # Initialize the MotionCompensator
        motion_compensator = MotionCompensator(blocksize=8, search_range=16)

        compensated_frame = motion_compensator.compensate_motion(frame1, motion_vectors)
        
        psnr = 10 * np.log10(1 / np.mean((compensated_frame - frame2) ** 2))
        self.assertGreater(psnr, 26, f"PSNR is too low: {psnr} dB, expected > 30 dB")