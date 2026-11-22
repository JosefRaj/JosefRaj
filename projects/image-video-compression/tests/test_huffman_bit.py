from unittest import TestCase
import numpy as np

from src.huffman import create_huffman_table_from_signal
from src.helpers import huffman_table_dict_to_bitstream, huffman_table_bitstream_to_dict


class TestHuffmanBit(TestCase):
    def test_huffman_table_bitstream_conversion(self):
        # Sample Huffman table for testing
        original_huffman_table = {
            (np.int64(0), np.int64(1)): '00',
            (np.int64(0), np.int64(2)): '01',
            (np.int64(1), np.int64(1)): '10',
            (np.int64(1), np.int64(2)): '11'
        }

        # Convert the Huffman table to a bitstream
        bitstream = huffman_table_dict_to_bitstream(original_huffman_table)

        # Convert the bitstream back to a Huffman table
        reconstructed_huffman_table = huffman_table_bitstream_to_dict(bitstream)

        # Assert that the original and reconstructed tables are the same
        for key in original_huffman_table:
            self.assertIn(key, reconstructed_huffman_table, f"Key {key} missing in reconstructed Huffman table.")
            self.assertEqual(original_huffman_table[key], reconstructed_huffman_table[key], f"Code for key {key} does not match.\nOriginal: {original_huffman_table[key]}\nReconstructed: {reconstructed_huffman_table[key]}")
        
    def test_huffman_table_bitstream_conversion_with_signal(self):
        # Sample signal for testing
        signal = np.array([(0, 1), (0, 2), (1, 1), (1, 2), (0, 1), (1, 2)], dtype=np.int64)

        # Create a Huffman table from the signal
        original_huffman_table = create_huffman_table_from_signal(signal)

        # Convert the Huffman table to a bitstream
        bitstream = huffman_table_dict_to_bitstream(original_huffman_table)

        # Convert the bitstream back to a Huffman table
        reconstructed_huffman_table = huffman_table_bitstream_to_dict(bitstream)

        # Assert that the original and reconstructed tables are the same
        for key in original_huffman_table:
            self.assertIn(key, reconstructed_huffman_table, f"Key {key} missing in reconstructed Huffman table.")
            self.assertEqual(original_huffman_table[key], reconstructed_huffman_table[key], f"Code for key {key} does not match.\nOriginal: {original_huffman_table[key]}\nReconstructed: {reconstructed_huffman_table[key]}")
        