from unittest import TestCase
from src.zigzag import generate_zigzag_permutation_matrix, ZigZagScanner
import numpy as np
from parameterized import parameterized


class TestZigZagPermutationMatrix(TestCase):
    @parameterized.expand([
        (2, 2, np.array([[0, 1],
                         [2, 3]])),
        (4, 4, np.array([[0, 1, 5, 6],
                         [2, 4, 7, 12],
                         [3, 8, 11, 13],
                         [9, 10, 14, 15]])),
        (8, 8, np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                         [2, 4, 7, 13, 16, 26, 29, 42],
                         [3, 8, 12, 17, 25, 30, 41, 43],
                         [9, 11, 18, 24, 31, 40, 44, 53],
                         [10, 19, 23, 32, 39, 45, 52, 54],
                         [20, 22, 33, 38, 46, 51, 55, 60],
                         [21, 34, 37, 47, 50, 56, 59, 61],
                         [35, 36, 48, 49, 57, 58, 62, 63]])),
        (3, 7, np.array([[0, 1, 5],
                         [2, 4, 6],
                         [3, 7, 11],
                         [8, 10, 12],
                         [9, 13, 17],
                         [14, 16, 18],
                         [15, 19, 20]])),
        (6, 4, np.array([[0, 1, 5, 6, 13, 14],
                         [2, 4, 7, 12, 15, 20],
                         [3, 8, 11, 16, 19, 21],
                         [9, 10, 17, 18, 22, 23]]))
    ])
    def test_matrix_basic(self, width, height, expected_permutation):
        permutation_matrix = generate_zigzag_permutation_matrix(width, height)
        
        self.assertTrue(np.array_equal(permutation_matrix, expected_permutation), "Generated zigzag permutation matrix does not match the expected matrix." +
                            f"\nExpected:\n{expected_permutation}\nActual:\n{permutation_matrix}")
        

class TestZigZagScanner(TestCase):
    @parameterized.expand([
        (2, 2),
        (4, 4),
        (8, 8),
        (3, 7),
        (6, 4)
    ])
    def test_encode(self, width, height):
        scanner = ZigZagScanner(width, height)
        
        permutation_matrix = generate_zigzag_permutation_matrix(width, height)
        
        encoded = scanner.encode_block(permutation_matrix)
        
        self.assertTrue(np.array_equal(encoded, np.arange(width * height)), f"Zigzag encoding failed for width={width}, height={height}.\n" + 
                        f"Expected: {np.arange(width * height)}\nActual: {encoded}")

    @parameterized.expand([
        (2, 2),
        (4, 4),
        (8, 8),
        (3, 7),
        (6, 4)
    ])
    def test_decode(self, width, height):
        scanner = ZigZagScanner(width, height)
        
        permutation_matrix = generate_zigzag_permutation_matrix(width, height)
        
        scanned_values = np.arange(width * height)
        decoded = scanner.decode_block(scanned_values)
        
        self.assertTrue(np.array_equal(decoded, permutation_matrix), f"Zigzag decoding failed for width={width}, height={height}.\n" + 
                        f"Expected:\n{permutation_matrix}\nActual:\n{decoded}")
        
    @parameterized.expand([
        (2, 2),
        (4, 4),
        (8, 8),
        (3, 7),
        (6, 4)
    ])
    def test_encode_decode(self, width, height):
        scanner = ZigZagScanner(width, height)
        
        # Create a test block with values from 0 to width*height-1
        block = np.arange(width * height).reshape((height, width))
        
        encoded = scanner.encode_block(block)
        decoded = scanner.decode_block(encoded)
        
        self.assertTrue(np.array_equal(block, decoded), f"Zigzag encode-decode failed for width={width}, height={height}.\n" + 
                        f"Original Block:\n{block}\nDecoded Block:\n{decoded}")
        

class TestZigZagScannerImage(TestCase):
    @parameterized.expand([
        (2, 2),
        (4, 4),
        (8, 8),
        (3, 7),
        (6, 4)
    ])
    def test_encode(self, width, height):
        scanner = ZigZagScanner(width, height)
        
        # Create a test image with values from 0 to 15
        image = generate_zigzag_permutation_matrix(width, height)
        image = np.tile(image, (4, 4))  # Create a larger image by tiling the original image
        
        encoded_image = scanner.encode_image(image)
        
        expected_encoded = np.arange(width * height)
        expected_encoded = np.tile(expected_encoded, (16, 1))  # Create a larger expected encoded image by tiling
        
        self.assertTrue(np.array_equal(encoded_image, expected_encoded), f"Zigzag image encoding failed.\n" +
                        f"Expected:\n{expected_encoded}\nActual:\n{encoded_image}")
    
    @parameterized.expand([
        (2, 2),
        (4, 4),
        (8, 8),
        (3, 7),
        (6, 4)
    ])
    def test_decode(self, width, height):
        scanner = ZigZagScanner(width, height)
        
        # Create a test image with values from 0 to 15
        expected_image = generate_zigzag_permutation_matrix(width, height)
        expected_image = np.tile(expected_image, (4, 4))  # Create a larger image by tiling the original image
        
        encoded_image = np.arange(width * height)
        encoded_image = np.tile(encoded_image, (16, 1))  # Create a larger encoded image by tiling
        
        decoded_image = scanner.decode_image(encoded_image, expected_image.shape)
        
        self.assertTrue(np.array_equal(expected_image, decoded_image), f"Zigzag image decode failed.\n" +
                        f"Original Image:\n{expected_image}\nDecoded Image:\n{decoded_image}")

    def test_encode_decode(self):
        scanner = ZigZagScanner(8, 8)
        
        # Create a test image with values from 0 to 15
        original_image = np.random.rand(288, 352)
        
        encoded_image = scanner.encode_image(original_image)
        decoded_image = scanner.decode_image(encoded_image, original_image.shape)
        
        self.assertTrue(np.array_equal(original_image, decoded_image), f"Zigzag image encode-decode failed.\n" +
                        f"Original Image:\n{original_image}\nDecoded Image:\n{decoded_image}")