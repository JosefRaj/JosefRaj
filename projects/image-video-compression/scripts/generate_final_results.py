
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
import pandas as pd
from argparse import ArgumentParser

from src.helpers import yuv_read_one_frame
from src.final import get_final_encoder, get_final_decoder
from src.metrics import mse_of_frame

def main():
    parser = ArgumentParser()
    parser.add_argument('--sequence', type=str, nargs='+', default=['flowergarden_short_cif'])
    parser.add_argument('--output_dir', type=str, default='.')

    args = parser.parse_args()

    print(f"Evaluating coding of multiple frames for {'.yuv '.join(args.sequence)}.yuv for the final submission.")
    
    qp_values = [6, 8, 10, 15, 20, 35]

    # define output directories
    os.makedirs(f"{args.output_dir}/encoded_files", exist_ok=True)
    os.makedirs(f"{args.output_dir}/codec_stats", exist_ok=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title('Rate-Distortion Curve')
    ax[0].set_xlabel('Rate (bits per pixel)') 
    ax[0].set_ylabel('PSNR (dB)')
    ax[0].grid()
    ax[1].set_title('Encoding and Decoding Time vs QP')
    ax[1].set_xlabel('QP')
    ax[1].set_ylabel('Time (s)')
    ax[1].invert_xaxis()
    ax[1].grid()
    colors = plt.color_sequences['tab10']
    
    for idx, sequence in enumerate(args.sequence):
        print(f"Now testing {sequence}.yuv")
        rates = []
        psnrs = []
        enc_times = []
        dec_times = []
        # Delete all auxiliary files
        for qp in qp_values:
            try: 
                os.remove(f"{args.output_dir}/encoded_files/{sequence}_encoded_qp_{qp}_final.pkl")
            except:
                pass
            try:
                os.remove(f"{args.output_dir}/encoded_files/{sequence}_decoded_qp_{qp}_final.yuv")
            except:
                pass

        for qp in qp_values:
            framemse = []
            encode = get_final_encoder(width=352, height=288, qp=qp)
            decode = get_final_decoder()

            start_time = time()
            encode(f"{sequence}.yuv", f"{args.output_dir}/encoded_files/{sequence}_encoded_qp_{qp}_final.pkl", 5)
            enc_times.append(time() - start_time)

            start_time = time()
            decode(coded_file=f"{args.output_dir}/encoded_files/{sequence}_encoded_qp_{qp}_final.pkl", decoded_yuv_file=f"{args.output_dir}/encoded_files/{sequence}_decoded_qp_{qp}_final.yuv")
            dec_times.append(time() - start_time)
            
            # Calculate rate (in bits per pixel)
            rate = os.path.getsize(f"{args.output_dir}/encoded_files/{sequence}_encoded_qp_{qp}_final.pkl") * 8 / (352 * 288 * 5)  # bits per pixel
            print(f"Bytes written for QP {qp}: {os.path.getsize(f'{args.output_dir}/encoded_files/{sequence}_encoded_qp_{qp}_final.pkl') / 1000} KB\n")
            rates.append(rate)

            framemse = [0 for _ in range(5)]
            for i in range(5):
                decoded_frame = yuv_read_one_frame(f"{args.output_dir}/encoded_files/{sequence}_decoded_qp_{qp}_final.yuv", i, 288, 352)[0]
                original_frame = yuv_read_one_frame(f"{sequence}.yuv", i, 288, 352)[0]
                framemse[i] = mse_of_frame(original_frame, decoded_frame)
            psnrs.append(10 * np.log10(1. / np.mean(framemse)))

        ax[0].plot(rates, psnrs, marker='o', label=f'{sequence}', color=colors[idx])

        ax[1].plot(qp_values, enc_times, marker='o', label=f'{sequence} Encoding Time', color=colors[idx])
        ax[1].plot(qp_values, dec_times, marker='x', linestyle='--', label=f'{sequence} Decoding Time', color=colors[idx])

        # Save stat file
        stats = {
            "qp_values": qp_values,
            "rates": rates,
            "psnrs": psnrs,
            "enc_times": enc_times,
            "dec_times": dec_times,
        }

        stats = pd.DataFrame(stats)
        stats.to_csv(f"{args.output_dir}/codec_stats/final_stats_{sequence}.csv", index=False)
   
    ax[0].legend()
    # Set legend outside to the right of plot
    ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    plt.tight_layout()
    plt.show()

    fig.savefig(f"{args.output_dir}/codec_stats/final_graph_{'_'.join(args.sequence)}.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()