import unittest
import numpy as np

from src.EntropyCoder import EntropyCoder


class TestEntropyCoder(unittest.TestCase):
    def test_entropy_coder(self):
        signal = np.array([(0, 0), (0, 0), (0, 1), (0, 1), (0, 1), (0, 1), (2, 2)])
        coder = EntropyCoder(signal=signal)
        bitstream = coder.encode_to_bitstream(signal)
        bitstream.pos = 0
        decoded_signal = coder.decode_from_bitstream(bitstream, num_symbols=len(signal))
        
        self.assertTrue(np.array_equal(signal, decoded_signal), f"Original signal: {signal}, Decoded signal: {decoded_signal}")

if __name__ == '__main__':
    unittest.main()