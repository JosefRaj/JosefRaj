import numpy as np 
import matplotlib.pyplot as plt 
import os
from time import time
import pandas as pd
from argparse import ArgumentParser

from src.helpers import yuv_read_one_frame
from src.video_codec import EncodeIntra, DecodeIntra
from src.metrics import mse_of_frame

def main():
    parser = ArgumentParser()
    parser.add_argument('--sequence', type=str, default='flowergarden_short_cif')
    parser.add_argument('--blocksize', type=int, default=8)
    args = parser.parse_args()

    print(f"Evaluating intra coding of multiple frames for {args.sequence}.yuv \nwith blocksize {args.blocksize}")
    
    rates = []
    psnrs = []
    enc_times = []
    dec_times = []
    
    qp_values = [6, 8, 10, 15, 20, 35]
    
    os.makedirs("encoded_files", exist_ok=True)
    
    # Delete all auxiliary files
    for qp in qp_values:
        try: 
            os.remove(f"encoded_files/{args.sequence}_encoded_intra_qp_{qp}.pkl")
        except:
            pass
        try:
            os.remove(f"encoded_files/{args.sequence}_decoded_qp_{qp}.yuv")
        except:
            pass

    for qp in qp_values:
        framemse = []
        encode = EncodeIntra(width=352, height=288, qp=qp)
        decode = DecodeIntra()

        start_time = time()
        encode(f"{args.sequence}.yuv", f"encoded_files/{args.sequence}_encoded_intra_qp_{qp}.pkl", 5)
        enc_times.append(time() - start_time)

        start_time = time()
        decode(coded_file=f"encoded_files/{args.sequence}_encoded_intra_qp_{qp}.pkl", decoded_yuv_file=f"encoded_files/{args.sequence}_decoded_qp_{qp}.yuv")
        dec_times.append(time() - start_time)
        
        # Calculate rate (in bits per pixel)
        rate = os.path.getsize(f"encoded_files/{args.sequence}_encoded_intra_qp_{qp}.pkl") * 8 / (352 * 288 * 5)  # bits per pixel
        print(f"Bytes written for QP {qp}: {os.path.getsize(f'encoded_files/{args.sequence}_encoded_intra_qp_{qp}.pkl') / 1000} KB")
        rates.append(rate)
        framemse = []
        for i in range(5):
            decoded_frame = yuv_read_one_frame(f"encoded_files/{args.sequence}_decoded_qp_{qp}.yuv", i, 288, 352)[0]
            original_frame = yuv_read_one_frame(f"{args.sequence}.yuv", i, 288, 352)[0]
            framemse.append(mse_of_frame(original_frame, decoded_frame))
        psnrs.append(10 * np.log10(1. / np.mean(framemse)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(rates, psnrs, marker='o')
    ax[0].set_title('Rate-Distortion Curve')
    ax[0].set_xlabel('Rate (bits per pixel)') 
    ax[0].set_ylabel('PSNR (dB)')
    ax[0].grid()

    ax[1].plot(qp_values, enc_times, marker='o', label='Encoding Time')
    ax[1].plot(qp_values, dec_times, marker='o', label='Decoding Time')
    ax[1].set_title('Encoding and Decoding Time vs QP')
    ax[1].set_xlabel('QP')
    ax[1].set_ylabel('Time (s)')
    ax[1].legend()
    ax[1].grid()

    plt.show()

    # Save stat file
    stats = {
        "qp_values": qp_values,
        "rates": rates,
        "psnrs": psnrs,
        "enc_times": enc_times,
        "dec_times": dec_times,
    }

    os.makedirs("codec_stats", exist_ok=True)
    stats = pd.DataFrame(stats)
    stats.to_csv(f"codec_stats/{args.sequence}_bs_{args.blocksize}_stats_intra.csv", index=False)
    fig.savefig(f"codec_stats/{args.sequence}_bs_{args.blocksize}_stats_intra.png", dpi=300)


if __name__ == '__main__':
    main()