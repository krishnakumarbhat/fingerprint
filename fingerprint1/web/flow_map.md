# Fingerprint Verification — Complete Flow Map (HLD + LLD)

This document provides a detailed high-level and low-level design walkthrough for the 12 files that make up the biometric fingerprint verification pipeline.

---

## STAGE 1 — Config Dataclasses
**File:** `src/p01_config.py`

### HLD
Global configuration storage using frozen Python dataclasses. Encapsulates dataset directories, neural network dimension layers, training hyperparameters, and experimental sweep grids.

### LLD
| Component | Detail |
|-----------|--------|
| **SplitConfig** | Defines dataset splits (Train=0.70, Val=0.15, Test=0.15) and global randomization seed=42. |
| **RuntimeConfig** | Specifies input size (96x103), embedding size (128-D), batch size (48), and GPU hardware pin memory configuration. |
| **TrainingConfig** | Configures learning rates (1e-3, 3e-4), triplet margins (0.75, 1.0), and early stopping patience (2). |
| **ExperimentGrid** | Sets up the parametric sweeps across 5 preprocess levels, 5 augmentation options, and 2 architectures. |
| **AppConfig** | Root settings manager resolving relative paths (e.g. `archive/SOCOFing`) and validating directory existence. |

---

## STAGE 2 — BMP Decoder
**File:** `src/p02_bmp_reader.py`

### HLD
Custom low-level BMP image reader implementing binary struct parsing. Bypasses external image processing libraries to operate purely on standard library primitives.

### LLD
| Component | Detail |
|-----------|--------|
| **Header Parser** | Unpacks BMP header bytes to extract pixel offset, DIB size, width, height, planes, bpp, and compression details. |
| **Row Stride Calc** | Handles 4-byte alignment padding. Formula: `row_stride = ((width * bpp + 31) // 32) * 4`. |
| **8-bit Decoder** | Reads palette color table, computes grayscale average of each color entry, and maps indices to grayscale pixels. |
| **Direct Decoder** | Decodes 24/32-bit pixel matrices. Converts color channels using NTSC formula: `0.299*R + 0.587*G + 0.114*B`. |
| **Output Format** | Scales pixels to `[0.0, 1.0]` and wraps them in a PyTorch tensor of shape `(1, H, W)` float32. |

---

## STAGE 3 — Indexing & Parsing
**File:** `src/p03_dataset_index.py`

### HLD
Iterates through raw dataset folders, parses file path metadata, and catalogs all available prints into a structured index.

### LLD
| Component | Detail |
|-----------|--------|
| **Regex Metadata** | Parses filename: `subject_id` (int), `gender` (M/F), `hand` (Left/Right), `finger` (thumb/index/middle/ring/little). |
| **Altered Flags** | If path contains `'/Altered/'` -> source="altered", difficulty = parent folder (Easy/Med/Hard), else source="real". |
| **Data Record** | Encapsulates metadata in `FingerprintRecord` dataclass containing subject, gender, hand, finger, path, alteration details. |
| **Dataset Indexer** | Walks dataset path, filters files, instantiates records, and returns `list[FingerprintRecord]`. |
| **Identity Key** | Generates a unique finger identifier key: `'{subject_id}__{gender}_{hand}_{finger}'` to track unique fingerprints. |

---

## STAGE 4 — Identity-Disjoint Split
**File:** `src/p04_phase_one_split.py`

### HLD
Splits the indexed dataset into Train (70%), Val (15%), and Test (15%) partitions at the identity level to prevent data leakage.

### LLD
| Component | Detail |
|-----------|--------|
| **Stratified Split** | Groups by identity key, shuffles identities with seed 42, and distributes them (70/15/15 ratio). |
| **Leakage Policy** | Guarantees all altered versions (Easy/Medium/Hard) of a subject are grouped in the same split as the real scan. |
| **Split CSV** | Saves split mapping to `artifacts/splits/identity_split.csv` containing (split, identity_key, file_name, source, difficulty). |
| **Identity Sampler** | `IdentityBatchSampler` groups `B/4` identities, sampling 4 prints each to guarantee genuine pairs in every batch. |
| **DataLoader** | Creates PyTorch DataLoaders with pinned memory, multi-worker prefetching, and custom collation wrappers. |

---

## STAGE 5 — Image Preprocessing
**File:** `src/p05_phase_two_preprocessing.py`

### HLD
Enhances ridge clarity and removes background noise. Computes 5 distinct preprocessed variants of every image.

### LLD
| Component | Detail |
|-----------|--------|
| **Segmentation** | Divides image into 8x8 blocks, computes local variance, and masks out background blocks (threshold = `0.35 * max`). |
| **Normalization** | Adjusts segmented foreground pixels dynamically to target mean = 0.5 and standard deviation = 0.22. |
| **Orientation Field** | Computes Sobel horizontal/vertical gradients, estimates local ridge angles for Gabor filters. |
| **Gabor Filter Bank** | Applies a bank of 6 oriented Gabor kernels (kernel size=11, σ=3, λ=5.5, γ=0.6) to enhance ridge lines. |
| **Skeletonization** | Applies adaptive binarization followed by Zhang-Suen thinning to reduce ridge lines to 1-pixel width. |

---

## STAGE 6 — Data Augmentation
**File:** `src/p06_phase_three_augmentation.py`

### HLD
Applies stochastic geometric and structural transforms during training to improve spatial alignment and rotation robustness.

### LLD
| Component | Detail |
|-----------|--------|
| **Affine Rotation** | Applies random rotation up to ±45° using bilinear grid sampling in PyTorch spatial grid generator. |
| **Elastic Distortion** | Generates 8x8 random displacement field, upsamples/smoothes it (alpha=6.0), and warps the ridge grid. |
| **Obliteration** | Overlays random black rectangular patches (10% to 22% of short side) to simulate scars, cuts, or dirt. |
| **Combined Transform** | Sequentially applies rotation -> elastic deformation -> obliteration (found to be too destructive). |
| **Aug Wrapper** | Wraps dataset loading to dynamically apply selected augmentation variant on-the-fly during training. |

---

## STAGE 7 — Model Architectures
**File:** `src/p07_phase_four_models.py`

### HLD
Two deep neural network feature extractors that map 103x96 input tensors to normalized 128-dimensional embeddings.

### LLD
| Component | Detail |
|-----------|--------|
| **compact_siamese** | 4 Conv2d layers (1->24->48->72->96), BN, GELU, MaxPool2d (3x), AdaptiveAvgPool2d, Linear(96->128). |
| **mobile_triplet** | Conv2d (1->16) followed by 4 Depthwise Separable blocks (16->24->48->72->96) with stride-2 downsampling. |
| **DW-Separable Conv** | Splits standard conv into 3x3 depthwise conv (groups=in_channels) and 1x1 pointwise conv. Reduces parameters 4x. |
| **L2 Normalization** | Normalizes outputs to unit length: `x = x / ||x||_2`. Restricts embedding space to a hypersphere. |
| **Parameters** | `compact_siamese`: 118,632 parameters. `mobile_triplet`: 31,048 parameters. Both export 128-D vectors. |

---

## STAGE 8 — Metric Learning Loss
**File:** `src/p08_metric_learning.py`

### HLD
Calculates pairwise distances in embedding space and updates network weights using metric learning objectives.

### LLD
| Component | Detail |
|-----------|--------|
| **Pairwise L2 Dist** | Computes Euclidean distances between all embeddings in a batch: `d = sqrt(||x_i - x_j||_2^2 + eps)`. |
| **Contrastive Loss** | Pulls same-finger pairs together; pushes different-finger pairs apart if `d < margin` (margin=0.75). |
| **Contrastive Math** | `Loss = (1 - y) * d^2 + y * max(0, margin - d)^2` (`y=1` for negative pair, `y=0` for positive pair). |
| **BatchHard Triplet** | For each anchor in batch, extracts hardest positive (max AP) and hardest negative (min AN). |
| **Triplet Math** | `Loss = max(0, d_hardest_pos - d_hardest_neg + margin)`. Averaged across all anchors in batch. |

---

## STAGE 9 — Training & Validation
**File:** `src/p09_training.py`

### HLD
Executes training epochs, updates model parameters, checks validation performance, and saves checkpoints.

### LLD
| Component | Detail |
|-----------|--------|
| **Train Loop** | Runs forward pass, computes loss, scales gradients, performs backward pass, and clips grad norm to 1.0. |
| **AMP Float16** | Uses `torch.cuda.amp` (autocast & GradScaler) to execute forward computations in float16 for speed. |
| **Optimization** | Updates weights with AdamW (lr=3e-4, weight_decay=1e-4). Monitors validation EER after each epoch. |
| **Early Stopping** | Stops training if validation EER fails to improve for patience=2 epochs. Restores best weights. |
| **Checkpointing** | Saves best model parameters to `artifacts/checkpoints/` with experiment-encoded filenames. |

---

## STAGE 10 — Biometric Evaluation
**File:** `src/p10_phase_five_evaluation.py`

### HLD
Calculates matching performance metrics: False Acceptance Rate (FAR), False Rejection Rate (FRR), and Equal Error Rate (EER).

### LLD
| Component | Detail |
|-----------|--------|
| **Enrollment** | For each test identity, enrolls the first real print scan as the master reference template embedding. |
| **Genuine Scores** | Calculates L2 distance between master template and all remaining prints of the same finger (7,405 pairs). |
| **Impostor Scores** | Calculates L2 distance between master template and prints of other identities (115,200 pairs). |
| **Threshold Sweep** | Sweeps 512 thresholds from min_dist to max_dist. Computes FAR and FRR at each step. |
| **EER Calculation** | EER is the point where `|FAR - FRR|` is minimized. Best EER: 0.145% (`compact_siamese` + `raw`). |

---

## STAGE 11 — Visualization Generator
**File:** `src/p11_visualization.py`

### HLD
Renders pipeline diagnostic files, including preprocessing strips, augmentation comparisons, and performance curves.

### LLD
| Component | Detail |
|-----------|--------|
| **PNG Strip Stacking** | Concatenates preprocessing steps and augmentation variants along the width dimension for comparison strips. |
| **Grayscale Conversion** | Maps float32 tensors back to uint8 pixel bytes `[0, 255]` before writing files using `write_png`. |
| **SVG Line Chart** | Renders vector charts showing Training/Validation loss and ROC curves with coordinate mapping. |
| **SVG Bar Chart** | Programmatically draws bar charts showing EER comparisons across different preprocessing variants. |
| **Summary Writer** | Saves vector visualizations to custom file names under `artifacts/visualizations/` for the web frontend. |

---

## STAGE 12 — Runner & Orchestration
**File:** `src/p12_experiment_runner.py`

### HLD
High-level pipeline driver orchestrating all experiment phases and exporting final consolidated summaries.

### LLD
| Component | Detail |
|-----------|--------|
| **Phase A Baseline** | Trains a baseline Siamese network on raw inputs to establish primary metrics. |
| **Phase B Search** | Sweeps all 5 augmentation options, identifying 'none' (no augmentation) as the top performer. |
| **Phase C Search** | Sweeps combinations of learning rates, batch sizes, and contrastive loss margins. |
| **Phase D Grid** | Sweeps all 10 combinations of models and preprocessing variants to identify the optimal pipeline. |
| **Consolidated JSON** | Serializes histories, best epoch metadata, and FAR/FRR lists into `pipeline_summary.json` (1.4MB). |
