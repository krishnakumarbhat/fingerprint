# Project Story

This file records what happened while building, debugging, and optimizing the fingerprint verification system. It is written as an engineering narrative, not a marketing summary.

## Objective

The goal was to turn the downloaded SOCOFing dataset into a verification project that does all of the following:

- split the dataset with finger-identity disjointness,
- train a baseline before augmentation,
- implement the full classical enhancement chain,
- benchmark all requested augmentation families,
- compare two different deep metric-learning models across all five feature pipelines,
- evaluate every final combination with biometric metrics,
- produce visual assets that make every phase explainable during an interview.

## Phase 1: Dataset Split and Baseline

The project uses the `identity_key = subject + gender + hand + finger` so that the same finger never appears across train, validation, and test. This is stricter than a plain image-level split and avoids leakage from altered variants of the same finger.

The baseline run was trained on raw images without augmentation. In the balanced benchmark it did not collapse into obvious overfitting, but the baseline still served a critical purpose: it established that the raw pipeline was already very strong on SOCOFing, which changed the interpretation of later augmentation and enhancement results.

Balanced-run baseline snapshot:

- Train loss: `0.05914`
- Validation loss: `0.05169`
- Baseline gap (`val - train`): `-0.00746`

The baseline gap staying close to zero means the model was not simply memorizing training samples under the chosen schedule.

## Phase 2: Classical Enhancement

I implemented every requested classical stage:

1. Global block-wise segmentation via block variance thresholding.
2. Adaptive intensity normalization on the segmented foreground.
3. Ridge orientation estimation using smoothed image gradients.
4. Orientation-selective Gabor filtering with a small multi-angle kernel bank.
5. Adaptive local binarization.
6. Zhang-Suen thinning to reach a one-pixel skeleton.

The interesting outcome is that stronger classical processing did not beat the raw signal in the final ranking. That is a real result, not a missing feature. On this dataset and under this training budget, some preprocessing variants removed texture information that the learned encoder used effectively on its own.

Observations from the final grid:

- `raw` was best for both models.
- `normalized` and `segmented` stayed competitive.
- `gabor` helped readability but not the final EER.
- `skeleton` was the most lossy representation and the weakest feature pipeline for both models.

## Phase 3: Augmentation and Hyperparameter Search

The augmentation search covered all requested categories:

- `none`
- `rotate`
- `elastic`
- `obliterate`
- `combined`

Validation EER ranking from the balanced search:

- `none`: `0.01184`
- `obliterate`: `0.01297`
- `elastic`: `0.01403`
- `rotate`: `0.03077`
- `combined`: `0.04180`

Why `none` won:

- SOCOFing already includes altered prints, so the dataset itself injects distortion.
- Aggressive geometric rotation was more damaging than helpful for this small sensor format.
- Combined augmentation compounded interpolation artifacts and made validation separation worse.

The best hyperparameter configuration was:

- Learning rate: `3e-4`
- Batch size: `48`
- Margin: `0.75`

Those values were reused for the final 10-combination comparison.

## Phase 4: Model Optimization

Two models were built and benchmarked:

### `compact_siamese`

- Compact CNN encoder.
- Contrastive batch loss.
- Lowest memory footprint.
- Most stable optimization on the 4 GB GPU.

### `mobile_triplet`

- Depthwise-separable encoder.
- Batch-hard triplet loss.
- More expressive objective, but harder to optimize under a tight budget.

What happened during optimization:

- The compact Siamese model converged faster and more predictably.
- The mobile triplet model worked, but its margin-based batch-hard objective was more sensitive to preprocessing choice and short schedules.
- On the constrained GPU, the simpler encoder plus contrastive objective consistently produced the strongest validation and test separation.

This is why the final winner was not the more complicated triplet model. The more advanced objective did not translate into better EER in this dataset-specific setting.

## Phase 5: Final Evaluation

The project uses biometric metrics instead of plain classification scores.

- FAR measures impostor acceptance.
- FRR measures genuine rejection.
- EER is the main ranking criterion.

Top final results from the balanced run:

| Rank | Model | Feature Pipeline | Test EER | Test Accuracy |
| --- | --- | --- | ---: | ---: |
| 1 | `compact_siamese` | `raw` | 0.00145 | 0.99847 |
| 2 | `compact_siamese` | `normalized` | 0.00317 | 0.99689 |
| 3 | `compact_siamese` | `segmented` | 0.00611 | 0.99399 |
| 4 | `compact_siamese` | `gabor` | 0.00861 | 0.99130 |
| 5 | `mobile_triplet` | `raw` | 0.00978 | 0.99040 |

Main conclusion:

The most accurate system in this benchmark is the compact Siamese model trained on raw fingerprint images with no extra augmentation and the tuned configuration `lr=3e-4`, `batch=48`, `margin=0.75`.

## Debugging and Issues Faced

### 1. `torchvision` could not read SOCOFing BMP files

The installed `torchvision` build only decoded PNG, JPEG, and GIF through `read_image`, so the first implementation failed immediately on BMP input.

Fix:

- Wrote a custom BMP reader with NumPy and the Python standard library.

### 2. SOCOFing did not use a single BMP encoding

The dataset actually contains three image encodings:

- `49,270` files with `8-bit` paletted BMP
- `5,956` files with `32-bit` bitfield BMP
- `44` files with `24-bit` uncompressed BMP

Fix:

- Extended the reader to support all three observed encodings.

### 3. A subset of altered images had a different spatial resolution

Most images are `103 x 96`, but `433` altered files are `298 x 241`.

Fix:

- Normalized image size inside the dataset class before batching.
- This kept the rest of the pipeline clean and removed shape-specific branching.

### 4. Metric learning needed identity-balanced batches

Plain random batching produced too few positive pairs for contrastive or triplet learning to be reliable.

Fix:

- Added a balanced batch sampler that draws several samples per identity into every minibatch.

### 5. 4 GB VRAM imposed model and batch-size limits

Fix:

- Kept both backbones lightweight.
- Enabled mixed precision.
- Tuned around batch sizes `32` and `48`.
- Used small but still meaningful search budgets in the balanced profile.

## Validation

The project was validated with both direct execution and tests.

- Smoke pipeline: passed.
- Balanced benchmark: passed.
- Pytest suite: `5 passed`.

The tests cover:

- BMP decoding,
- split correctness,
- preprocessing invariants,
- verification metrics,
- a miniature real-data training run.

## What To Say in an Interview

If someone asks why the raw pipeline won even though the project implemented advanced enhancement, the honest answer is:

- the classical enhancement chain was implemented correctly,
- it improved interpretability of the ridge structure,
- but the learned encoder extracted the most discriminative information from the raw image under this dataset and compute budget,
- and the evaluation framework was strict enough to show that rather than assuming preprocessing must always help.

That answer is stronger than claiming every classical step improved accuracy, because it is backed by the actual benchmark table rather than expectation.