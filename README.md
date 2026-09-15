# COMP3710 — Lab Demonstration 2: Pattern Recognition

Repository for the Lab 2 demo (Pattern Analysis, UQ) — dimensionality reduction, classification, and deep learning pipelines using TensorFlow/PyTorch, following the lab sheet by Shekhar "Shakes" Chandra.

The repo is organised by part; each part folder contains the source code and its corresponding output(s).

## Repository structure

```
comp3710_demo2/
├── part1/                      # Discrete Fourier Transform
│   ├── part1.py
│   └── part1_dft_plot.png
├── part2/                      # Eigenfaces (PCA + Random Forest)
│   ├── part2.py
│   ├── eigenfaces.png
│   └── compactness_plot.png
├── part3/                      # CNNs & the DAWNBench challenge
│   ├── part3_1.py              # basic CNN classifier (LFW)
│   ├── part3_2.py              # DAWNBench: ResNet-18 on CIFAR-10
│   └── dawnbench_resnet18.pth
├── part4/                      # Recognition task: OASIS brain MRI
│   ├── vae/
│   │   ├── part4.py
│   │   └── vae_brain_manifold.png
│   ├── unet/
│   │   ├── part4_unet.py
│   │   └── unet_segmentation_result.png
│   └── gan/
│       ├── part4_gan.py
│       ├── gan_loss_plot.png
│       └── gan_generated_brains.png
└── README.md
```

## Part 1 — Discrete Fourier Transform 

Reconstructs a square wave from its odd-harmonic Fourier series, then compares three ways of computing the DFT of the resulting signal:

- **Naive DFT** — direct O(N²) double-loop implementation (CPU)
- **Vectorized DFT** — same O(N²) computation expressed as a matrix–vector product, run on GPU/CPU tensors
- **Built-in FFT** — `torch.fft.fft`, O(N log N)

**Run:**
```bash
python part1/part1.py
```
**Output:** `part1_dft_plot.png` — input square wave (with visible Gibbs-phenomenon ripple at the discontinuities) and its magnitude spectrum, showing energy only at odd harmonics.

**Result:** FFT < vectorized (GPU) < naive (CPU) in execution time, consistent with algorithmic complexity (O(N log N) vs O(N²)) and the parallelism available to each implementation.

## Part 2 — Eigenfaces 

PCA (via SVD) of the Labeled Faces in the Wild (LFW) dataset, followed by a Random Forest classifier on the projected "face space" features.

**Run:**
```bash
python part2/part2.py
```
**Outputs:**
- `eigenfaces.png` — top eigenfaces (principal component directions reshaped as images)
- `compactness_plot.png` — cumulative explained variance vs. number of components

**Result:** 150 principal components retain ~95% of total variance; classification performance/report printed to console.

## Part 3 — CNNs & DAWNBench 

**3.1 — CNN classifier:** two 3×3 convolution layers (32 filters each) + dense layers, trained on LFW with Adam and (sparse) categorical cross-entropy.

**3.2 — DAWNBench challenge:** modified ResNet-18 (3×3 stride-1 stem, no initial maxpool — adapted for CIFAR-10's 32×32 inputs) trained with data augmentation, a OneCycleLR schedule, and mixed-precision (AMP) for fast convergence.

**Run:**
```bash
python part3/part3_1.py
python part3/part3_2.py
```

**Targets:**
- \>90% test accuracy, training in reasonable time on Rangpur
- Runs inference + one training epoch live on Rangpur during the demo
- Stretch: ~94% accuracy in ≈360s (V100-equivalent benchmark)

**Result:** _fill in final test accuracy and total training time achieved._

## Part 4 — Recognition: OASIS Brain MRI

All three recognition tasks attempted on the preprocessed OASIS MRI dataset.

### Task 1 — Variational Autoencoder (VAE)
Convolutional encoder → (μ, log σ²) → reparameterization trick → convolutional decoder, trained with a BCE reconstruction loss + KL-divergence regularizer (ELBO).

```bash
python part4/vae/part4.py
```
**Output:** `vae_brain_manifold.png` — 64 brain slices decoded from random latent samples.

### Task 2 — UNet segmentation
Encoder–decoder with skip connections, segmenting each MRI slice into 4 classes (background, CSF, grey matter, white matter). Trained with cross-entropy; evaluated with the Dice similarity coefficient (DSC).

```bash
python part4/unet/part4_unet.py
```
**Output:** `unet_segmentation_result.png` — original MRI, ground-truth mask, and predicted mask side by side.

**Result:** _fill in per-class DSC (target: >0.9 for all labels, including background)._

### Task 3 — Generative Adversarial Network (GAN)
DCGAN-style generator/discriminator pair trained adversarially to generate realistic brain MRI slices from 100-dim latent noise.

```bash
python part4/gan/part4_gan.py
```
**Outputs:** `gan_loss_plot.png` (generator/discriminator loss curves), `gan_generated_brains.png` (generated sample grid).

## Environment

- Python 3.x, PyTorch (CUDA/MPS-aware device selection)
- Additional: `torchvision`, `numpy`, `matplotlib`, `scikit-learn`, `Pillow`
- Part 3.2 and Part 4 were trained on the **Rangpur HPC cluster** (UQ), using the OASIS dataset at `/home/groups/comp3710/OASIS/`

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

## Author

Sulaim Shahood — COMP3710 (Pattern Analysis), University of Queensland
