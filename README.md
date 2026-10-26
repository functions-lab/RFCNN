# RF-CNN: Radio-Frequency Convolutional Neural Networks

Code for **[Radio-Frequency Convolutional Neural Networks](https://arxiv.org/abs/2609.19279)** (Gao, Ma, Chen, Englund, Chen, 2026).

A frequency mixer multiplies two signals in time, which is a correlation in frequency, i.e. the operation of a convolutional layer. RF-CNN maps the input **X** and the weights **W** onto frequency tones (*tone mapping*), so that a passive *computing mixer* performs a whole multi-channel convolution in one pass. Models are trained digitally and then run through RF-CNN without retraining.

This README covers the **simulation** mode (the default): an ideal mixer with AWGN, implemented in PyTorch. Hardware experiments on USRPs are enabled with `--exp`.

## Setup

```bash
pip install torch torchvision numpy scipy matplotlib tqdm numba
pip install "torchmetrics[image]"   # GAN / FID only
```

Run all scripts from the `RFCNN/` directory. Data is read from `../Data/` and results are written to `../Result/<DATA>_size<SIZE>_<token>/`.

```
../Data/
├── MNIST/      mnist_train.csv, mnist_test.csv
├── FMNIST/     fashion-mnist_train.csv, fashion-mnist_test.csv
├── SVHN/       train_32x32.mat, test_32x32.mat
├── CIFAR-10/   cifar-10-batches-py/
├── DeepSig/    data_<m>_<snr>.mat   (RadioML 2018.01A split per modulation m=1..24 and SNR; variable `wave`)
└── CelebA/     auto-downloaded by torchvision
```

Images are not resampled, so `--size` must match the native resolution: 28 for MNIST/FMNIST, 32 for SVHN/CIFAR-10, and 1024 for DeepSig. CelebA is resized to `--size`.

## Usage

**1. Train digitally**

```bash
python main_classify_digital.py --data CIFAR-10 --size 32 --param VGG11 --tokenOut VGG11-best
python main_classify_digital.py --data SVHN     --size 32 --param VGG11 --tokenOut VGG11-best
python main_classify_digital.py --data DeepSig  --size 1024 --model ConvNet1D --param VGG-DeepSig-new --tokenOut VGG-v2-best

python main_gan_digital.py --data MNIST  --size 28  --param GRAY --tokenOut pixel-GRAY
python main_gan_digital.py --data CelebA --size 128 --param High --tokenOut pixel-High
```

**2. Run RF-CNN inference in simulation** (default `--snr 20`)

```bash
python main_classify_analog.py --data CIFAR-10 --size 32 --param VGG11 --tokenIn VGG11-best --snr 20
python main_gan_analog.py      --data MNIST --size 28 --param GRAY --batch 10 --tokenIn pixel-GRAY
```

**3. Compute the FID (GAN only)**

```bash
python main_gan_evaluate.py --data MNIST --size 28 --tokenOut pixel-GRAY-sim
```

Each analog run also performs a full-precision pass for reference. It reports accuracy, confusion matrices, normalized RMSE, resolution bit and ENOB (classification), or generated image grids (GAN).

## Models

| Preset | Paper model |
| --- | --- |
| `VGG-DeepSig-new` + `ConvNet1D` | 6-layer DeepSig classifier (Fig. 3a) |
| `VGG11` | 9-layer SVHN / CIFAR-10 classifier (Fig. 3b) |
| `GRAY` (`--size 28`) | 4-layer MNIST / FMNIST InfoGAN generator (Fig. 4a) |
| `High` (`--size 128`) | 7-layer CelebA generator (Fig. 4b) |
| `RGB` (`--size 64`) | extra 64×64 CelebA variant |

## How the simulation works

- **Tone mapping** ([Core/interface_torch.py](Core/interface_torch.py), `ConvAcc` / `FCAcc`): each input channel is unrolled row by row onto tones, with zero guard tones between rows. Channels occupy adjacent sub-bands, so the mixer also sums over input channels. Zadoff-Chu phases reduce the waveform PAPR.
- **Mixer** ([Tx2Rx_sim_torch.py](Core/Tx2Rx_mixer/Tx2Rx_sim_torch.py)): the two OFDM waveforms are multiplied (`s_X · conj(s_W)`, down-conversion), low-pass filtered, given AWGN at `--snr`, and decimated. The output tones are then decoded by FFT, with phase correction from the preamble and postamble.
- **Network** ([analog.py](analog.py)): all layers are complex-valued. `AnalogZoom` matches each layer's output scale to full precision, and `AnalogInfer` chains the layers. `engineList` is set per dataset in `main_classify_analog.py` and per preset in `main_gan_analog.py`. A layer whose entry is `None` runs digitally (e.g. DeepSig `Conv1`, as in the paper). In simulation the numeric `engineList` values are unused; they are per-layer backoffs for hardware.
- **Link parameters** are in [Core/GetParam.py](Core/GetParam.py): `waveAmp`, padding, `sampleRate`, carriers, `upsample`, `shrinkMax`.

## Notes

- Results are cached per batch in `Details/`, and the default `--tokenOut` (`<tokenIn>-sim`) does not include the SNR. **Use a different `--tokenOut` for each `--snr`**, or delete the old folder.
- Simulation is slow for large models; use a GPU and `--batch 1` for CelebA.
- The DeepSig loader uses all 24 modulation classes, while the paper merges them into 10.

## Citation

```bibtex
@article{gao2026radio,
  title   = {Radio-frequency convolutional neural networks},
  author  = {Gao, Zhihui and Ma, Shi-Yuan and Chen, Yiran and Englund, Dirk and Chen, Tingjun},
  journal = {arXiv preprint arXiv:2609.19279},
  year    = {2026},
}
```
