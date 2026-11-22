import numpy as np
from bitstring import BitStream

from .huffman import create_huffman_table_from_signal
from .helpers import init_bitstream, bitstream_append, bitstream_read


class EntropyCoder:
    def __init__(self, signal: np.ndarray | None = None, huffman_table: dict | None = None):
        self.huffman_table = None 

        if huffman_table is not None:
            self.huffman_table = huffman_table
        elif signal is not None:
            self.huffman_table = create_huffman_table_from_signal(signal)    

    def encode_to_bitstream(self, signal: np.ndarray, bitstream: BitStream | None = None) -> BitStream:

        if self.huffman_table is None:
            raise ValueError("Huffman table is not initialized.")

        if bitstream is None:
            bitstream = init_bitstream()

        for symbol in signal:
            symbol = tuple(symbol)

            if symbol not in self.huffman_table:
                raise ValueError(f"Symbol {symbol} is not in the Huffman table.")

            bitstream_append(bitstream, self.huffman_table[symbol])

        return bitstream
        
    
    def decode_from_bitstream(self, bitstream: BitStream, num_symbols: int = -1, huffman_table=None) -> np.ndarray:
        
        if huffman_table is None:
            huffman_table = self.huffman_table

        if huffman_table is None:
            raise ValueError("Huffman table is not initialized.")

        if num_symbols is None or num_symbols < 0:
            num_symbols = bitstream.len

        reverse_huffman_table = {}

        for symbol, codeword in huffman_table.items():
            reverse_huffman_table[codeword] = symbol

        min_code_length = float("inf")
        max_code_length = 0

        for codeword in reverse_huffman_table:
            code_length = len(codeword)

            min_code_length = min(min_code_length, code_length)
            max_code_length = max(max_code_length, code_length)
    
        decoded_symbols = []

        for _ in range(num_symbols):
            matched_code = None
            current_bits = bitstream_read(bitstream, min_code_length)

            if current_bits == "":
                break

            while len(current_bits) <= max_code_length:

                if current_bits in reverse_huffman_table:
                    matched_code = reverse_huffman_table[current_bits]
                    break
                
                if len(current_bits) == max_code_length:
                    break

                next_bit = bitstream_read(bitstream, 1)

                if next_bit == "":
                    print("End of bitstream reached before a complete codeword.")
                    return np.array(decoded_symbols)

                current_bits += next_bit

            if matched_code is None:
                raise ValueError(
                    f"No Huffman codeword matches the bits: {current_bits}"
                )

            decoded_symbols.append(matched_code)

        return np.array(decoded_symbols)