# Fingerprint Verification – Complete System Explainer

## What Is This Project?

This project builds a **biometric fingerprint verification system** on the SOCOFing dataset. Given two fingerprint images, the system decides: *are they from the same finger or not?* It does this by converting each fingerprint into a 128-dimensional numeric vector (an "embedding") and measuring the distance between the two vectors.

---

## The SOCOFing Dataset

### What Is "Real" vs "Altered"?

**Real Fingerprints:**
- Genuine scans directly from subjects (600 subjects × 10 fingers = 6,000 unique finger identities).
- Stored in the `/Real/` folder as-is.
- File naming: `100__M_Left_thumb_finger.BMP` → subject 100, Male, Left hand, thumb finger.
- These form the **ground-truth enrollment** for each identity.

**Altered Fingerprints:**
- The same fingerprints, but *synthetically damaged/modified* to mimic forgery or wear.
- Stored in subfolders: `Easy/`, `Medium/`, `Hard/`.
- What each level means:
  - **Easy**: Minor alterations — the fingerprint is almost recognizable. Slight ridge distortions or small obliterations (scratches, partial erasure).
  - **Medium**: Moderate alterations — visible ridge structure changes, larger obliterated areas, rotation-combined with partial damage.
  - **Hard**: Severe alterations — large regions destroyed, major shape distortions; a human expert may struggle to match it.
- "Obliteration" means a rectangular patch of the fingerprint is blacked out (simulates deliberate damage, cuts, or sensor artifact).

### What Are Anchor, Positive, Negative?

These are the three roles in **Triplet Learning**:

| Role | Meaning | Example |
|------|---------|---------|
| **Anchor** | The reference fingerprint (usually a real scan) | Subject 100, thumb, real |
| **Positive** | A different image of the *same* finger | Subject 100, thumb, altered Easy |
| **Negative** | An image from a *different* finger/identity | Subject 200, index, real |

The model learns to pull Anchor–Positive embeddings **close together** and push Anchor–Negative embeddings **far apart**. This is the core of metric learning.

---

## Artifacts Explained

### `artifacts/checkpoints/`

Saved PyTorch model weights (`.pt` files). Each filename encodes the full experiment configuration:

```
compact_siamese__gabor__elastic__lr1em03__bs16__m0p75.pt
│               │      │         │         │     └─ margin=0.75
│               │      │         │         └─ batch_size=16
│               │      │         └─ learning_rate=1e-3
│               │      └─ augmentation=elastic
│               └─ preprocessing=gabor
└─ model=compact_siamese
```

Four checkpoint folders:

| Folder | Purpose |
|--------|---------|
| `baseline/` | Initial reference run: raw pixels, no augmentation. Used to measure how well the simplest setup works. |
| `augmentation_search/` | Systematic test of all 5 augmentation types on Gabor-processed images to find which augmentation helps most. |
| `hyperparameter_search/` | Grid search over learning rate × batch size × margin on the best augmentation. |
| `final_grid/` | Full 5×2 grid: 5 preprocessing variants × 2 model types, trained with the best hyperparameters found. These are the final ranked results. |

### `artifacts/splits/identity_split.csv`

**This is the train/validation/test split manifest.** It records every fingerprint file and which split (train=70%, validation=15%, test=15%) it was assigned to.

- Splits are **identity-disjoint**: no subject appears in two splits. If subject 100's fingers are in training, they are completely absent from validation and test. This prevents data leakage.
- The CSV columns: `split, identity_key, file_name, source, difficulty, alteration`
- Total: 55,270 records split into 38,707 train / 8,258 validation / 8,305 test.

### `artifacts/summaries/pipeline_summary.json`

The full serialized output of every experiment run, including:
- Training loss history per epoch
- Validation and test EER, accuracy, threshold, FAR, FRR
- Positive/negative distances per epoch
- ROC curve data (512-point FAR vs TPR)
- Asset paths for generated charts

---

## The Two Models

### Model 1: `compact_siamese` (Contrastive Loss)

**Architecture – Standard CNN Encoder:**
```
Input (1×103×96 grayscale)
→ Conv2d(1→24, 3×3) + BN + GELU + MaxPool(2×2)   [ch: 1→24, res: ÷2]
→ Conv2d(24→48, 3×3) + BN + GELU + MaxPool(2×2)  [ch: 24→48, res: ÷4]
→ Conv2d(48→72, 3×3) + BN + GELU + MaxPool(2×2)  [ch: 48→72, res: ÷8]
→ Conv2d(72→96, 3×3) + BN + GELU                 [ch: 72→96]
→ AdaptiveAvgPool2d(1×1) → Flatten → Linear(96→128)
→ L2-Normalize → embedding ∈ ℝ¹²⁸
```

**Loss Function: Batch Contrastive Loss**
- Treats all pairs in the batch
- Positive pairs (same identity): penalize `distance²`
- Hard negative pairs (different identity within margin): penalize `(margin - distance)²`
- `margin = 0.75`

### Model 2: `mobile_triplet` (Triplet Loss)

**Architecture – MobileNet-style (Depthwise Separable):**
```
Input (1×103×96 grayscale)
→ Conv2d(1→16, 3×3) + BN + GELU                  [standard conv]
→ DW-Sep(16→24, stride=2) + BN + GELU            [res: ÷2]
→ DW-Sep(24→48, stride=2) + BN + GELU            [res: ÷4]
→ DW-Sep(48→72, stride=2) + BN + GELU            [res: ÷8]
→ DW-Sep(72→96, stride=1) + BN + GELU            [res: ÷8]
→ AdaptiveAvgPool2d(1×1) → Flatten → Linear(96→128)
→ L2-Normalize → embedding ∈ ℝ¹²⁸
```

**DW-Sep = Depthwise Separable Convolution:**
```
Depthwise: Conv2d(C→C, 3×3, groups=C) — spatial filtering per channel
Pointwise: Conv2d(C→C', 1×1)          — channel mixing
```
This is ~8-9× fewer multiply-adds than standard convolution.

**Loss Function: Batch-Hard Triplet Loss**
- For each anchor, find the **hardest positive** (farthest same-identity sample in batch)
- For each anchor, find the **hardest negative** (closest different-identity sample in batch)
- Loss = `ReLU(dist_pos - dist_neg + margin).mean()`
- `margin = 0.75`

### Key Differences Between the Two Models

| Property | compact_siamese | mobile_triplet |
|----------|----------------|----------------|
| Conv type | Standard 3×3 | Depthwise Separable |
| Loss function | Contrastive (pair-based) | Batch-Hard Triplet |
| Param count | ~118K | ~31K |
| Channel sequence | 1→24→48→72→96 | 1→16→24→48→72→96 |
| Stride strategy | MaxPool ×3 | Stride-2 DW blocks ×3 |
| Activation | GELU | GELU |
| Best preprocessing | Raw | Raw |
| Best Test EER | **0.00145** | 0.00978 |
| Distance threshold | ~0.38 | ~0.79 |
| Positive distance | ~0.15–0.20 | ~0.50–0.60 |
| Negative distance | ~0.63–0.69 | ~0.75–1.05 |

---

## Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| **EER** | Where FAR = FRR | Equal Error Rate – lower is better. 0.00145 = 0.145% |
| **FAR** | Impostors accepted / total impostors | False Accept Rate – how often a fake passes |
| **FRR** | Genuine rejected / total genuine | False Reject Rate – how often a real person is blocked |
| **Threshold** | Decision boundary (L2 distance) | Below → "same person", above → "different" |
| **Positive distance** | Mean L2(anchor, genuine probe) | How close same-finger embeddings are |
| **Negative distance** | Mean L2(anchor, impostor) | How far different-finger embeddings are |
| **Accuracy** | At EER threshold | Classification accuracy at the operating point |

---

## Triplet Loss Formula

```
L = max(0,  d(a,p) - d(a,n) + margin)

Where:
  a = anchor embedding
  p = positive (same identity) embedding
  n = negative (different identity) embedding
  d = Euclidean distance
  margin = 0.75

Goal: d(a,p) + margin < d(a,n)
```

When you compare two fingerprints:
- Distance < threshold → **MATCH** (same person)
- Distance ≥ threshold → **NO MATCH** (different person)
