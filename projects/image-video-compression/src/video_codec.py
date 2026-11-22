
import numpy as np
from pickle import dump, load


from .helpers import yuv_read_one_frame, yuv_write_one_frame, get_quantization_matrix, huffman_table_dict_to_bitstream, huffman_table_bitstream_to_dict, mv_huffman_table_bitstream_to_dict, mv_huffman_table_dict_to_bitstream
from .transforms import blockbased_dct, blockbased_idct
from .motion import MotionCompensator
from .quantization import Quantizer
from .zigzag import ZigZagScanner, differential_dc_encoding, differential_dc_decoding
from .runlevel_coding import runlevel_encoding_blockbased, runlevel_decoding_blockbased
from .EntropyCoder import EntropyCoder
from .metrics import mse_of_frame


class EncodeIntra:
    def __init__(self, width: int, height: int, transform_blocksize: int = 8, qp: int = 1):
        pass
        
    def __call__(self, input_yuv_file: str, coded_file: str, number_of_frames: int):
        """
        Encode the input image using intra coding.
        
        :param input_yuv_file: Path to the input file.
        :param width: Width of the input image.
        :param height: Height of the input image.
        :param coded_file: Path where the encoded image will be saved.
        """
        pass
    
            
class DecodeIntra:
    def __call__(self, coded_file: str, decoded_yuv_file: str):
        """
        Decode the encoded image using intra coding.
        
        :param coded_file: Path to the encoded file.
        :param decoded_yuv_file: Path where the decoded image will be saved.
        """
        pass


class Encode:
    def __init__(self, width: int, height: int, transform_blocksize: int = 8, qp: int = 1, me_blocksize = 8, me_searchrange = 16):
        pass

    def __call__(self, input_yuv_file: str, coded_file: str, number_of_frames: int):
        """
        Encode the input image using intra coding.
        
        :param input_yuv_file: Path to the input file.
        :param width: Width of the input image.
        :param height: Height of the input image.
        :param coded_file: Path where the encoded image will be saved.
        """
        pass   
            
class Decode:
    def __call__(self, coded_file: str, decoded_yuv_file: str):
        """
        Decode the encoded image using intra coding.
        
        :param coded_file: Path to the encoded file.
        :param decoded_yuv_file: Path where the decoded image will be saved.
        """
        pass