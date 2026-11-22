import unittest
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio


class TestMetrics(unittest.TestCase):
    def test_mse_of_frame(self):
        from src.metrics import mse_of_frame
        
        np.random.seed(0)  # For reproducibility

        original_image = np.random.rand(10, 10)
        distorted_image = np.random.rand(10, 10)

        expected_mse = mean_squared_error(original_image, distorted_image)
        calculated_mse = mse_of_frame(original_image, distorted_image)

        self.assertEqual(calculated_mse, expected_mse)
        
    def test_psnr_of_frame_infinite(self):
        from src.metrics import psnr_of_frame
        
        original_image = np.array([[0, 0], [0, 0]])
        distorted_image = np.array([[0, 0], [0, 0]])

        expected_psnr = np.inf
        calculated_psnr = psnr_of_frame(original_image, distorted_image)

        self.assertEqual(calculated_psnr, expected_psnr)
        
    def test_psnr_of_frame(self):
        from src.metrics import psnr_of_frame
        
        np.random.seed(0)  # For reproducibility

        original_image = np.random.rand(10, 10)
        distorted_image = np.random.rand(10, 10)

        expected_psnr = peak_signal_noise_ratio(original_image, distorted_image, data_range=1.0)
        calculated_psnr = psnr_of_frame(original_image, distorted_image)

        self.assertEqual(calculated_psnr, expected_psnr)
        
        
if __name__ == '__main__':
    unittest.main()