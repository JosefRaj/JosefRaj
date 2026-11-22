from src.intra_codec import EncodeIntra, DecodeIntra
from src.helpers import yuv_read_one_frame
from src.metrics import psnr_of_frame

import matplotlib.pyplot as plt
import os

def main():
    original_image = yuv_read_one_frame("flowergarden_short_cif.yuv", 0, 288, 352)[0]
    
    rates = []
    psnrs = []
    
    qp_values = [6, 8, 10, 15, 20, 35]
    
    os.makedirs("encoded_files", exist_ok=True)
    # Example usage of EncodeIntra and DecodeIntra
    for qp in qp_values:
        encode_intra = EncodeIntra(width=352, height=288, qp=qp)
        encode_intra("flowergarden_short_cif.yuv", f"encoded_files/flowergarden_short_cif_frame0_encoded_qp_{qp}.pkl")
        
        # Calculate rate (in bits per pixel)
        rate = os.path.getsize(f"encoded_files/flowergarden_short_cif_frame0_encoded_qp_{qp}.pkl") * 8 / (352 * 288)  # bits per pixel
        print(f"Bytes written for QP {qp}: {os.path.getsize(f'encoded_files/flowergarden_short_cif_frame0_encoded_qp_{qp}.pkl') / 1000} KB")
        rates.append(rate)
        
        decode_intra = DecodeIntra()
        decode_intra(coded_file=f"encoded_files/flowergarden_short_cif_frame0_encoded_qp_{qp}.pkl", decoded_yuv_file=f"encoded_files/flowergarden_short_cif_frame0_decoded_qp_{qp}.yuv")

        decoded_image = yuv_read_one_frame(f"encoded_files/flowergarden_short_cif_frame0_decoded_qp_{qp}.yuv", 0, 288, 352)[0]
        psnr = psnr_of_frame(original_image, decoded_image)
        psnrs.append(psnr)
    
    # Plot Rate-Distortion Curve
    plt.figure(figsize=(10, 6))
    plt.plot(rates, psnrs, marker='o')
    plt.title('Rate-Distortion Curve')
    plt.xlabel('Rate (bits per pixel)') 
    plt.ylabel('PSNR (dB)')
    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    main()