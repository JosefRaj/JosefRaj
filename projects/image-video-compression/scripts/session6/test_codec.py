import numpy as np 
import matplotlib.pyplot as plt 
import os
from time import time
import pandas as pd
from argparse import ArgumentParser

from src.helpers import yuv_read_one_frame
from src.video_codec import Encode, Decode
from src.metrics import mse_of_frame

def main():
    parser = ArgumentParser()
    parser.add_argument('--sequence', type=str, nargs='+', default=['flowergarden_short_cif'])
    parser.add_argument('--transform_blocksize', type=int, default=8)
    parser.add_argument('--me_blocksize', type=int, default=16)  
    parser.add_argument('--me_searchrange', type=int, default=8)

    args = parser.parse_args()

    print(f"Evaluating coding of multiple frames for {'.yuv '.join(args.sequence)}.yuv \nwith transform blocksize {args.transform_blocksize}\nwith motion blocksize {args.me_blocksize}\nwith motion search range {args.me_searchrange}\n")
    

    
    qp_values = [6, 8, 10, 15, 20, 35]
    
    os.makedirs("encoded_files", exist_ok=True)
    
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
                os.remove(f"encoded_files/{sequence}_encoded_qp_{qp}.pkl")
            except:
                pass
            try:
                os.remove(f"encoded_files/{sequence}_decoded_qp_{qp}.yuv")
            except:
                pass

        for qp in qp_values:
            framemse = []
            encode = Encode(width=352, height=288, qp=qp, transform_blocksize=args.transform_blocksize, me_blocksize=args.me_blocksize, me_searchrange=args.me_searchrange)
            decode = Decode()

            start_time = time()
            encode(f"{sequence}.yuv", f"encoded_files/{sequence}_encoded_qp_{qp}.pkl", 5)
            enc_times.append(time() - start_time)

            start_time = time()
            decode(coded_file=f"encoded_files/{sequence}_encoded_qp_{qp}.pkl", decoded_yuv_file=f"encoded_files/{sequence}_decoded_qp_{qp}.yuv")
            dec_times.append(time() - start_time)
            
            # Calculate rate (in bits per pixel)
            rate = os.path.getsize(f"encoded_files/{sequence}_encoded_qp_{qp}.pkl") * 8 / (352 * 288 * 5)  # bits per pixel
            print(f"Bytes written for QP {qp}: {os.path.getsize(f'encoded_files/{sequence}_encoded_qp_{qp}.pkl') / 1000} KB\n")
            rates.append(rate)
            framemse = []
            for i in range(5):
                decoded_frame = yuv_read_one_frame(f"encoded_files/{sequence}_decoded_qp_{qp}.yuv", i, 288, 352)[0]
                original_frame = yuv_read_one_frame(f"{sequence}.yuv", i, 288, 352)[0]
                framemse.append(mse_of_frame(original_frame, decoded_frame))
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

        os.makedirs("codec_stats", exist_ok=True)
        stats = pd.DataFrame(stats)
        stats.to_csv(f"codec_stats/{sequence}_bs_{args.transform_blocksize}_mebs_{args.me_blocksize}_sr_{args.me_searchrange}_stats.csv", index=False)
   
    ax[0].legend()
    # Set legend outside to the right of plot
    ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    plt.tight_layout()
    plt.show()

    fig.savefig(f"codec_stats/{'_'.join(args.sequence)}_bs_{args.transform_blocksize}_mebs_{args.me_blocksize}_sr_{args.me_searchrange}_stats.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()