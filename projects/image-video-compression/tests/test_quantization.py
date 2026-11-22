import unittest
import numpy as np

from src.quantization import Quantizer


class TestMidtreadQuantization(unittest.TestCase):
    
    quantizer_matrix = np.array([[1, 1, 1], 
                                 [0.1, 0.1, 0.1], 
                                 [10, 10, 10], 
                                 [10.8, 10.8, 10.8]])

    def test_midtread_quantization_scalar(self):
        quantizer = Quantizer(stepsize=0.1, quantization_type='midtread')
        
        input = np.array([[0.049, 0.151], [0.251, 0.351]])
        expected_quantized = np.array([[0, 2], [3, 4]])
        
        quantized = quantizer.quantize(input)
        
        self.assertTrue(np.array_equal(quantized, expected_quantized), f"Expected:\n{expected_quantized}\nGot:\n{quantized}")
        
    def test_midtread_dequantization_scalar(self):
        quantizer = Quantizer(stepsize=0.1, quantization_type='midtread')
        
        quantized = np.array([[0, 2], [3, 4]])
        expected_dequantized = np.array([[0.0, 0.2], [0.3, 0.4]])
        
        dequantized = quantizer.dequantize(quantized)
        
        self.assertTrue(np.allclose(dequantized, expected_dequantized), f"Expected:\n{expected_dequantized}\nGot:\n{dequantized}")
        
    def test_midtread_quantization_matrix(self):
        quantizer = Quantizer(stepsize=self.quantizer_matrix, quantization_type='midtread')

        input_image = np.array([[0.0, 0.4, 0.6], 
                                [-0.4499,-0.4501, 100], 
                                [4, 6, -4], 
                                [5.3, 5.5, -5.5]])

        expected_quantized =  np.array([[0, 0, 1], 
                                    [-4, -5, 1000], 
                                    [0, 1, 0], 
                                    [0, 1, -1]])
    

        quantized = quantizer.quantize(input_image)
        self.assertTrue(np.array_equal(quantized, expected_quantized), f"Expected:\n{expected_quantized}\nGot:\n{quantized}")

    def test_midtread_dequantization_matrix(self):
        quantizer = Quantizer(stepsize=self.quantizer_matrix, quantization_type='midtread')

        quantized_image = np.array([[0, 0, 1], 
                                    [-4, -5, 1000], 
                                    [0, 1, 0], 
                                    [0, 1, -1]])

        expected_dequantized = np.array([[0.0, 0.0, 1.0], 
                                        [-0.4, -0.5, 100], 
                                        [0.0, 10.0, 0.0], 
                                        [0.0, 10.8, -10.8]])

        dequantized = quantizer.dequantize(quantized_image)
        self.assertTrue(np.allclose(dequantized, expected_dequantized), f"Expected:\n{expected_dequantized}\nGot:\n{dequantized}")
        
class TestDeadzoneQuantization(unittest.TestCase):    

    quantizer_matrix = np.array([[1, 1, 1], 
                                 [0.1, 0.1, 0.1], 
                                 [10, 10, 10], 
                                 [10.8, 10.8, 10.8]])

    def test_deadzone_quantization_scalar(self):
        try:
            quantizer = Quantizer(stepsize=0.1, quantization_type='deadzone')
        except NotImplementedError as e:
            return self.skipTest("Deadzone quantization not implemented yet")
        
        input = np.array([[0.05, 0.151], [0.251, 0.351]])
        expected_quantized = np.array([[0, 1], [2, 3]])
        
        quantized = quantizer.quantize(input)
        
        self.assertTrue(np.array_equal(quantized, expected_quantized), f"Expected:\n{expected_quantized}\nGot:\n{quantized}")

    def test_deadzone_dequantization_scalar(self):
        try:
            quantizer = Quantizer(stepsize=0.1, quantization_type='deadzone')
        except NotImplementedError as e:
            return self.skipTest("Deadzone quantization not implemented yet")
        
        quantized = np.array([[0, 1], [2, 3]])
        expected_dequantized = np.array([[0.00, 0.15], [0.25, 0.35]])
        
        dequantized = quantizer.dequantize(quantized)
        
        self.assertTrue(np.allclose(dequantized, expected_dequantized), f"Expected:\n{expected_dequantized}\nGot:\n{dequantized}")
        
    def test_deadzone_quantization_matrix(self):
        try:
            quantizer = Quantizer(stepsize=self.quantizer_matrix, quantization_type='deadzone')
        except NotImplementedError as e:
            return self.skipTest("Deadzone quantization not implemented yet")
        
        input_image = np.array([[0.0, 0.4, 0.6], 
                                [-0.45,-0.45, 100], 
                                [4, 6, -4], 
                                [5.3, 5.5, -5.5]])
        expected_quantized =  np.array([[0, 0, 0], 
                                    [-4, -4, 1000], 
                                    [0, 0, 0], 
                                    [0, 0, 0]])
        
        quantized = quantizer.quantize(input_image)
        
        self.assertTrue(np.array_equal(quantized, expected_quantized), f"Expected:\n{expected_quantized}\nGot:\n{quantized}")
        
    def test_deadzone_dequantization_matrix(self):
        try:
            quantizer = Quantizer(stepsize=self.quantizer_matrix, quantization_type='deadzone')
        except NotImplementedError as e:
            return self.skipTest("Deadzone quantization not implemented yet")
        quantized_image = np.array([[0, 0, 0], 
                                    [-4, -4, 1000], 
                                    [0, 1, 0], 
                                    [0, 0, -1]])
        expected_dequantized = np.array([[0.0, 0.0, 0.0], 
                                        [-0.45, -0.45, 100.05], 
                                        [0.0, 15.0, 0.0], 
                                        [0.0, 0, -16.2]])
        dequantized = quantizer.dequantize(quantized_image)
        self.assertTrue(np.allclose(dequantized, expected_dequantized), f"Expected:\n{expected_dequantized}\nGot:\n{dequantized}")