from unittest import TestCase
from src.runlevel_coding import runlevel_encoding, runlevel_decoding, runlevel_decoding_blockbased, runlevel_encoding_blockbased
import numpy as np



class TestRunLevelCoding(TestCase):
    def test_encoding(self):
        input_array = np.array([0, 0, 3, 0, 0, 0, 5, 0, 2])
        expected_output = np.array([[2, 3], [3, 5], [1, 2]])
        
        encoded = runlevel_encoding(input_array)
        
        self.assertTrue(np.array_equal(encoded, expected_output), f"Run-level encoding failed.\nExpected: {expected_output}\nActual: {encoded}")
        
    def test_decoding(self):
        encoded_array = np.array([[2, 3], [3, 5], [1, 2]])
        output_length = 9
        expected_output = np.array([0, 0, 3, 0, 0, 0, 5, 0, 2])
        
        decoded = runlevel_decoding(encoded_array, output_length)
        
        self.assertTrue(np.array_equal(decoded, expected_output), f"Run-level decoding failed.\nExpected: {expected_output}\nActual: {decoded}")
        

class TestRunLevelCodingBlockBased(TestCase):
    def test_encoding(self):
        input_array = np.array([[0, 1, 0, 0, 2, 0],
                                [3, 0, 4, 0, 5, 6]])
        expected_output = np.array([[1, 1], [2, 2], [-1, -1], [0, 3], [1, 4], [1, 5], [0, 6], [-1, -1]])
        
        encoded = runlevel_encoding_blockbased(input_array)
        
        self.assertTrue(np.array_equal(encoded, expected_output), f"Block-based run-level encoding failed.\nExpected: {expected_output}\nActual: {encoded}")
        
    def test_decoding(self):
        encoded_array = np.array([[1, 1], [2, 2], [-1, -1], [0, 3], [1, 4], [1, 5], [0, 6], [-1, -1]])
        output_length = 6
        expected_output = np.array([[0, 1, 0, 0, 2, 0],
                                     [3, 0, 4, 0, 5, 6]])
        
        decoded = runlevel_decoding_blockbased(encoded_array, output_length)
        
        self.assertTrue(np.array_equal(decoded, expected_output), f"Block-based run-level decoding failed.\nExpected: {expected_output}\nActual: {decoded}")