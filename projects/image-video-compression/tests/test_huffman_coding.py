import unittest
import numpy as np

from src.huffman import create_huffman_table_from_signal, create_huffman_table_from_probability


class TestHuffmanCoding(unittest.TestCase):
    def test_create_huffman_table_from_signal(self):
        input_signal = np.array([(0, 0), (0, 0), (0, 1), (0, 1), (0, 1), (0, 1), (2, 2)])
        expected_table = {
            (0, 0): '10',
            (0, 1): '0',
            (2, 2): '11'
        }
        
        huffman_table = create_huffman_table_from_signal(input_signal)
        
        self.assertEqual(set(huffman_table.keys()), set(expected_table.keys()))
        for symbol in expected_table.keys():
            self.assertIn(symbol, huffman_table)
            self.assertIsInstance(huffman_table[symbol], str)
            self.assertEqual(huffman_table[symbol], expected_table[symbol])
            
    def test_create_huffman_table_from_probability(self):
        #frequencies = {
        #    'A': 0.01,
        #    'F': 0.02,
        #    'E': 0.02,
        #    'G': 0.05,
        #    'C': 0.05,
        #    'B': 0.15,
        #    'D': 0.2,
        #    'LZ': 0.5,
        #}
        #expected_table = {
        #    'LZ': '0',
        #    'D': '11',
        #    'B': '100',
        #    'C': '1011',
        #    'G': '10100',
        #    'E': '101011',
        #    'F': '1010100',
        #    'A': '1010101',
        #}
        frequencies = [
                    ('A', 0.27),
                    ('B', 0.26),
                    ('C', 0.24),
                    ('D', 0.1),
                    ('E', 0.08),
                    ('F', 0.05),
        ]
        expected_table = {
            'A': '00',
            'B': '01',
            'C': '10',
            'D': '111',
            'E': '1100',
            'F': '1101'
        }
        huffman_table = create_huffman_table_from_probability(frequencies)
        
        self.assertEqual(set(huffman_table.keys()), set(expected_table.keys()))
        for symbol in expected_table.keys():
            self.assertIn(symbol, huffman_table)
            self.assertIsInstance(huffman_table[symbol], str)
            self.assertEqual(huffman_table[symbol], expected_table[symbol], f"Symbol: {symbol}, Expected: {expected_table[symbol]}, Got: {huffman_table[symbol]}")

            
if __name__ == '__main__':
    unittest.main()