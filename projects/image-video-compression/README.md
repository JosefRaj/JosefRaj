# Image and Video Compression Laboratory Project

**Status: in progress.** This repository snapshot documents ongoing work for the Image and Video Compression laboratory at FAU Erlangen-Nürnberg. It contains working components, course scaffolding and unfinished parts; it is not a complete video codec.

## Work in the current snapshot

- Blockwise image processing and midtread quantization/dequantization using NumPy.
- Huffman table construction, DCT/IDCT transforms, zigzag scanning, and motion estimation/compensation code.
- Image-distortion metrics including MSE and PSNR, plus scripts intended for later rate–distortion experiments.
- Unit-test files and laboratory scripts covering individual components.

The project includes laboratory-provided scaffolding and helper code. The snapshot should not be interpreted as a claim that every included file was independently authored from scratch.

## Known incomplete components

- The intra-image and video encoder/decoder classes in `src/intra_codec.py` and `src/video_codec.py`, and the final encoder/decoder factories in `src/final.py`, are placeholders.
- Run-level coding currently returns empty results for the provided tests, and dead-zone quantization is not supported by the current `Quantizer`.
- End-to-end coding, bitrate and rate–distortion results have not been verified.

## Environment

The project targets Python 3.10. If conda is installed, create the environment with:

```bash
conda env create -f environment.yml
conda activate privc
```

From this project directory, run tests with:

```bash
python -m unittest discover -s tests -v
```

Some tests and scripts require original laboratory media such as YUV sequences or reference data. The course notebooks, media assets, generated files and local editor settings are not included in this snapshot.

## Verification of this snapshot

On 22 November 2026, Python syntax compilation succeeded for `src`, `scripts` and `tests`. In a separate Python 3.12 environment with NumPy, all four midtread quantization tests passed; four dead-zone quantization tests errored because that mode is not supported; and all four run-level coding tests failed because those functions return empty results. The full suite was not run because several dependencies were unavailable in the verification environment.

## Next steps

Complete and validate the coding chain and run-level operations, then measure rate–distortion and reconstruction quality once the final encoder and decoder are implemented.
