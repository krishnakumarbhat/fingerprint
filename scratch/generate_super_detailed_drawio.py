# Helper script to generate a super detailed uncompressed Draw.io XML file for the Fingerprint pipeline.
import xml.etree.ElementTree as ET
import html

# Define colors matching the website style
COLORS = {
    "accent": "#7c6af7",
    "accent2": "#a78bfa",
    "green": "#34d399",
    "yellow": "#fbbf24",
    "blue": "#60a5fa",
    "orange": "#fb923c",
    "red": "#f87171",
    "cyan": "#22d3ee",
    "magenta": "#f472b6",
    "indigo": "#818cf8",
    "bg": "#0a0a0f",
    "surface": "#1e1e28",
    "surface2": "#252532",
    "bg3": "#111118",
    "text": "#e8e8f0",
    "text_muted": "#8888aa",
    "border2": "#2d2d3a"
}

STAGES = [
    {
        "num": "1",
        "title": "Stage 1: Config Dataclasses",
        "files": "p01_config.py",
        "color": COLORS["accent"],
        "hld": "Global configuration storage using frozen Python dataclasses. Encapsulates dataset directories, neural network layer dimensions, training hyperparameters, and experimental sweeps. Serves as the single source of truth for the entire pipeline to guarantee type-safety and determinism.",
        "lld_intro": "Defines five distinct frozen configuration schemas to guarantee end-to-end type safety, static attributes, and deterministic random seeds.",
        "components": [
            {
                "icon": "⚙️",
                "title": "SplitConfig",
                "desc": "Specifies dataset partition ratios and randomization seed:\n• train_ratio: float = 0.70\n• validation_ratio: float = 0.15\n• test_ratio: float = 0.15\n• seed: int = 42"
            },
            {
                "icon": "📐",
                "title": "RuntimeConfig",
                "desc": "Specifies hardware parameters and tensor dimensions:\n• image_width: int = 96\n• image_height: int = 103\n• embedding_dim: int = 128\n• batch_size: int = 48\n• eval_batch_size: int = 96\n• num_workers: int = 2\n• device: str = 'cuda' / 'cpu'\n• pin_memory: bool = True (if cuda)"
            },
            {
                "icon": "📝",
                "title": "TrainingConfig",
                "desc": "Specifies training and optimization hyperparameters:\n• baseline_epochs: int = 2\n• search_epochs: int = 2\n• final_epochs: int = 2\n• samples_per_identity: int = 4\n• train_steps_per_epoch: int = 120\n• search_steps_per_epoch: int = 60\n• learning_rates: tuple = (1e-3, 3e-4)\n• batch_sizes: tuple = (32, 48)\n• margins: tuple = (0.75, 1.0)\n• weight_decay: float = 1e-4\n• early_stopping_patience: int = 2\n• max_impostor_pairs_per_identity: int = 128"
            },
            {
                "icon": "🔀",
                "title": "ExperimentGrid",
                "desc": "Specifies configuration sweeps for grids:\n• preprocess_variants: tuple = ('raw', 'segmented', 'normalized', 'gabor', 'skeleton')\n• augmentation_variants: tuple = ('none', 'rotate', 'elastic', 'obliterate', 'combined')\n• model_variants: tuple = ('compact_siamese', 'mobile_triplet')"
            },
            {
                "icon": "📂",
                "title": "AppConfig & Loader",
                "desc": "Aggregates nested configs:\n• project_root: Path\n• dataset_root: Path = project_root/archive/SOCOFing\n• output_root: Path = project_root/artifacts\n• split: SplitConfig\n• runtime: RuntimeConfig\n• training: TrainingConfig\n• grid: ExperimentGrid\n• Resolves relative paths dynamically via: AppConfig.from_root(project_root)"
            }
        ]
    },
    {
        "num": "2",
        "title": "Stage 2: BMP Decoder",
        "files": "p02_bmp_reader.py",
        "color": COLORS["accent2"],
        "hld": "Custom low-level BMP image reader implementing binary struct parsing. Bypasses external image processing libraries (like PIL or OpenCV) to operate purely on standard library primitives and numpy. Decodes 8-bit paletted, 24-bit RGB, and 32-bit BGRA images directly into normalized float32 grayscale tensors.",
        "lld_intro": "Decodes binary headers and pixel streams, handling row strides, color mapping tables, NTSC transformations, and vertical orientation corrections.",
        "components": [
            {
                "icon": "🔍",
                "title": "Header Parser",
                "desc": "Unpacks BMP header bytes:\n• Magic bytes: checks if starts with b'BM'\n• Offset 10: pixel_offset (I)\n• Offset 14: DIB size (I)\n• Offset 18: width (i), height (i), planes (H), bits_per_pixel (H), compression (I)\n• Validates planes == 1"
            },
            {
                "icon": "📐",
                "title": "Row Stride Calc",
                "desc": "Handles 4-byte padding alignment:\n• Formula: ((width * bpp + 31) // 32) * 4\n• Prevents pixel misalignment during buffer reads."
            },
            {
                "icon": "🎨",
                "title": "8-bit Paletted Decoder",
                "desc": "Decodes 8-bit color indexed images:\n• Extracts palette at (14 + DIB size)\n• Computes grayscale palette: palette[:, :3].mean(axis=1)\n• Reads pixel index matrix of size (height, stride) via np.frombuffer\n• Maps indices to grayscale colors"
            },
            {
                "icon": "🖼️",
                "title": "Direct Decoder",
                "desc": "Decodes color pixel matrices:\n• 24-bit: reads RGB, applies NTSC: 0.299*R + 0.587*G + 0.114*B\n• 32-bit: unpacks channel bitmasks from header (default BGRA), extracts channels via _mask_to_channel(), scales by alpha: grayscale = NTSC * alpha"
            },
            {
                "icon": "🔢",
                "title": "load_socofing_bmp",
                "desc": "Entry point returning PyTorch Tensor:\n• Loads file bytes\n• Decodes via bits_per_pixel (8, 24, 32)\n• Flips image vertically if height > 0 (via np.flipud) since BMPs are stored bottom-up\n• Scales output to [0.0, 1.0] shape (1, H, W) float32"
            }
        ]
    },
    {
        "num": "3",
        "title": "Stage 3: Indexing & Parsing",
        "files": "p03_dataset_index.py",
        "color": COLORS["green"],
        "hld": "Iterates through the raw dataset folders, parses filenames using tokenization to extract metadata fields, and catalogs all available fingerprint prints into a structured manifest of records. Handles both case-insensitive directory names and varied filename structures.",
        "lld_intro": "Parses filenames into FingerprintRecord objects, walks raw directories, and handles Real and Altered splits.",
        "components": [
            {
                "icon": "🏷️",
                "title": "FingerprintRecord",
                "desc": "Dataclass representing one file:\n• path: Path\n• subject_id: int\n• gender: str ('M'/'F')\n• hand: str ('Left'/'Right')\n• finger: str (thumb/index/middle/ring/little)\n• source: str (real/altered)\n• difficulty: str (Real/Easy/Med/Hard)\n• alteration: str (e.g. CR, Z, W, REAL)"
            },
            {
                "icon": "🔑",
                "title": "Identity Key",
                "desc": "Returns unique identifier:\n• f'{subject_id:03d}__{gender}_{hand}_{finger}_finger'\n• Uniquely groups all variations (real, altered, augmentations) of a single finger."
            },
            {
                "icon": "⚙️",
                "title": "Filename Parser",
                "desc": "Extracts metadata from filename:\n• Splits filename stem by '__' to get subject ID\n• Splits remainder by '_' to extract gender, hand, finger\n• Checks for alteration label at index 4 (defaults to REAL)\n• Source is 'altered' if 'altered' is in path parts"
            },
            {
                "icon": "📂",
                "title": "Dataset Indexer",
                "desc": "Walks root directory:\n• Case-insensitively locates 'Real' and 'Altered' subfolders\n• Gathers paths using glob: *.BMP, *.bmp, *.Bmp, *.BMP*\n• Parses files via parse_record()"
            },
            {
                "icon": "📋",
                "title": "Index Validation",
                "desc": "Assembles and validates lists:\n• Combines real and altered list[FingerprintRecord]\n• Ensures data is sorted and paths are unique\n• Raises FileNotFoundError if no BMP files are found"
            }
        ]
    },
    {
        "num": "4",
        "title": "Stage 4: Identity-Disjoint Split",
        "files": "p04_phase_one_split.py",
        "color": COLORS["yellow"],
        "hld": "Partitions the indexed dataset into Train (70%), Validation (15%), and Test (15%) subsets. Splitting is executed strictly at the subject identity level to prevent data leakage (ensuring that a subject's real scan and all their altered variants are grouped in the same partition).",
        "lld_intro": "Groups files by identity, shuffles keys, assigns splits, saves manifests, and provides custom PyTorch Datasets/Samplers.",
        "components": [
            {
                "icon": "🔀",
                "title": "Disjoint Splitter",
                "desc": "Splits identity keys disjointly:\n• Groups records by identity_key\n• Shuffles keys with np.random.default_rng(seed).permutation()\n• Splits keys at 70% and 85%\n• Maps all associated records into Train/Val/Test subsets"
            },
            {
                "icon": "🚫",
                "title": "Leakage Policy",
                "desc": "Guarantees zero leakage across partitions:\n• All altered versions (Easy/Medium/Hard) of a subject are grouped in the same split as the real scan.\n• Evaluates models on completely unseen fingers."
            },
            {
                "icon": "📄",
                "title": "Manifest CSV",
                "desc": "Serializes the split results to 'identity_split.csv':\n• Columns: split, identity_key, file_name, source, difficulty, alteration\n• Enables visual inspection and exact reproduction of splits."
            },
            {
                "icon": "⚖️",
                "title": "FingerprintDataset",
                "desc": "PyTorch Dataset implementation:\n• __getitem__(index) loads image via load_socofing_bmp()\n• Resizes image using F.interpolate in bilinear mode if it differs from target_size\n• Returns (image_tensor, label_int, dataset_index)"
            },
            {
                "icon": "⚡",
                "title": "IdentityBatchSampler",
                "desc": "Ensures balanced batches during training:\n• Groups indices by identity_key\n• Draws B/4 identity keys per step\n• Samples 4 indices per identity (sampling with replacement if needed)\n• Guarantees positive pairs in every batch for metric learning"
            }
        ]
    },
    {
        "num": "5",
        "title": "Stage 5: Preprocessing",
        "files": "p05_phase_two_preprocessing.py",
        "color": COLORS["blue"],
        "hld": "Enhances fingerprint ridge clarity and filters background noise. Computes 5 distinct preprocessed variants: raw (original), segmented (background mask applied), normalized (adaptive contrast adjustment), gabor (directional Gabor filtering), and skeleton (single-pixel ridge thinning).",
        "lld_intro": "Applies mathematical computer vision algorithms using PyTorch operations to process batch image tensors.",
        "components": [
            {
                "icon": "✂️",
                "title": "Block Segmentation",
                "desc": "Extracts foreground mask:\n• Divides image into 8x8 blocks\n• Computes block mean and variance via F.avg_pool2d\n• Thresholds: variance > mean_variance * 0.35\n• Interpolates to original size, applies 5x5 smoothing filter, thresholds at 0.2"
            },
            {
                "icon": "📐",
                "title": "Intensity Normalization",
                "desc": "Equalizes foreground ridge contrast:\n• Computes mean and variance of pixels under mask\n• Normalizes: (pixel - mean) / sqrt(variance)\n• Scales: normalized * target_std (0.22) + target_mean (0.5)\n• Zeroes out background pixels outside mask"
            },
            {
                "icon": "🌀",
                "title": "Ridge Orientation",
                "desc": "Estimates local ridge angles:\n• Convolves image with Sobel-X and Sobel-Y filters\n• Computes local covariances (grad_xx, grad_yy, grad_xy) using 9x9 average pool\n• Computes angle: 0.5 * atan2(2*grad_xy, grad_xx - grad_yy + 1e-6) + pi"
            },
            {
                "icon": "🌿",
                "title": "Gabor Filter Bank",
                "desc": "Applies 6 directional Gabor kernels:\n• Bank (angles 0 to pi, step pi/7) cached via @lru_cache\n• Convolves image with all 6 kernels via conv2d\n• For each pixel, gathers response of filter closest to local ridge angle\n• Normalizes output and applies sigmoid"
            },
            {
                "icon": "🦴",
                "title": "Zhang-Suen Thinning",
                "desc": "Reduces ridges to 1-pixel skeletons:\n• Binarizes via F.avg_pool2d local mean\n• Runs Zhang-Suen algorithm in numpy\n• Iteratively prunes pixels checking 8-neighbors (degree 2 to 6, transition count == 1, sub-pass masks)\n• Converts back to tensor"
            }
        ]
    },
    {
        "num": "6",
        "title": "Stage 6: Data Augmentation",
        "files": "p06_phase_three_augmentation.py",
        "color": COLORS["orange"],
        "hld": "Applies stochastic spatial deformations, rotations, and structural obliterators to input images. Augmentation prevents model overfitting, makes features invariant to print rotation, and simulates real-world noise (such as scars, smudge, or poor contact).",
        "lld_intro": "Implements geometric and occlusion transforms on-the-fly using PyTorch grid sampling and random coordinate masking.",
        "components": [
            {
                "icon": "🔄",
                "title": "Grid Generator",
                "desc": "Creates baseline coordinate grids:\n• Generates normalized coordinates in [-1.0, 1.0]\n• Shapes grid tensor to (batch, height, width, 2) using torch.linspace and meshgrid\n• Serves as spatial template for coordinate warping"
            },
            {
                "icon": "📐",
                "title": "Affine Rotation",
                "desc": "Stochastic image rotation:\n• Samples batch of angles uniformly in [-45, 45] degrees\n• Builds 2x3 transformation matrices containing cos and sin values\n• Generates grids using F.affine_grid\n• Samples pixels via F.grid_sample in bilinear mode"
            },
            {
                "icon": "🌊",
                "title": "Elastic Distortion",
                "desc": "Simulates skin elasticity:\n• Generates low-res (8x8) random displacement vectors\n• Interpolates to image size using bicubic mode\n• Smoothes displacements with 5x5 average pooling\n• Warps base grid: grid + displacement\n• Resamples image via F.grid_sample"
            },
            {
                "icon": "⬛",
                "title": "Obliteration",
                "desc": "Simulates scars or occlusion:\n• Computes random patch size (10% to 22% of short side)\n• Samples random starting coordinates (x_start, y_start)\n• Overlays zeroed (black) square mask on-the-fly"
            },
            {
                "icon": "🔀",
                "title": "Dispatcher",
                "desc": "Routes the augmentation request:\n• 'none': returns original image\n• 'rotate', 'elastic', 'obliterate': returns single transform\n• 'combined': chains rotation -> elastic distortion -> obliteration"
            }
        ]
    },
    {
        "num": "7",
        "title": "Stage 7: Model Architectures",
        "files": "p07_phase_four_models.py",
        "color": COLORS["red"],
        "hld": "Defines deep neural network encoders that map a 103x96 input image to a normalized 128-dimensional latent embedding. Projects features onto a hypersphere so that L2 Euclidean distances correspond directly to biometric similarity.",
        "lld_intro": "Defines sub-layers and compares a compact standard CNN (compact_siamese) with a parameter-efficient depthwise separable CNN (mobile_triplet).",
        "components": [
            {
                "icon": "🌐",
                "title": "L2Normalize Layer",
                "desc": "Normalizes features to unit L2 length:\n• Class L2Normalize(nn.Module)\n• forward(x) computes: x / ||x||_2 via F.normalize(dim=1)\n• Projects embeddings onto a 128-D hypersphere surface to stabilize metric learning"
            },
            {
                "icon": "⚡",
                "title": "Depthwise Separable Conv",
                "desc": "Implements MobileNet-style block:\n• 3x3 depthwise conv (groups=in_channels, bias=False)\n• BatchNorm2d + GELU activation\n• 1x1 pointwise conv (out_channels, bias=False)\n• BatchNorm2d + GELU activation\n• Reduces parameters 3x to 4x"
            },
            {
                "icon": "🧠",
                "title": "compact_siamese Encoder",
                "desc": "Standard CNN (118K parameters):\n• Conv2d(1 -> 24), BN, GELU, MaxPool2d(2)\n• Conv2d(24 -> 48), BN, GELU, MaxPool2d(2)\n• Conv2d(48 -> 72), BN, GELU, MaxPool2d(2)\n• Conv2d(72 -> 96), BN, GELU\n• AdaptiveAvgPool2d(1) -> Flatten\n• Linear(96 -> 128) -> L2Normalize"
            },
            {
                "icon": "📱",
                "title": "mobile_triplet Encoder",
                "desc": "Lightweight CNN (31K parameters):\n• Conv2d(1 -> 16), BN, GELU\n• DepthwiseSeparableBlock(16 -> 24, stride=2)\n• DepthwiseSeparableBlock(24 -> 48, stride=2)\n• DepthwiseSeparableBlock(48 -> 72, stride=2)\n• DepthwiseSeparableBlock(72 -> 96, stride=1)\n• AdaptiveAvgPool2d(1) -> Flatten\n• Linear(96 -> 128) -> L2Normalize"
            },
            {
                "icon": "⚙️",
                "title": "create_model Factory",
                "desc": "Instantiates network & loss name:\n• 'compact_siamese': returns compact encoder and 'contrastive' loss config\n• 'mobile_triplet': returns mobile encoder and 'triplet' loss config\n• Validates input names"
            }
        ]
    },
    {
        "num": "8",
        "title": "Stage 8: Metric Learning Loss",
        "files": "p08_metric_learning.py",
        "color": COLORS["cyan"],
        "hld": "Calculates distance metrics in the 128-D embedding space. Updates network parameters by minimizing objective functions that pull matching fingers together and push non-matching fingers apart.",
        "lld_intro": "Computes pairwise Euclidean matrices and calculates Contrastive Loss (with hard negative mining) and Batch-Hard Triplet Loss.",
        "components": [
            {
                "icon": "📏",
                "title": "Pairwise L2 Matrix",
                "desc": "Computes all-to-all Euclidean distances:\n• Inputs embeddings tensor of shape (N, 128)\n• Output matrix shape is (N, N) using torch.cdist(p=2)\n• Distance formula: d(x, y) = ||x - y||_2"
            },
            {
                "icon": "🔗",
                "title": "Contrastive Batch Loss",
                "desc": "Computes pairwise contrastive loss:\n• Extracts positive/negative pairs via upper-triangular masks\n• Positive loss = positive_distances.square().mean()\n• Extracts hard negatives: distances < margin\n• Negative loss = relu(margin - hard_negatives).square().mean()"
            },
            {
                "icon": "🎯",
                "title": "Batch-Hard Triplet Loss",
                "desc": "Mines hardest samples in batch:\n• For each anchor in batch, extracts hardest positive distance (maximum same-class distance)\n• Extracts hardest negative distance (minimum diff-class distance)\n• Loss = relu(hardest_pos - hardest_neg + margin).mean()"
            },
            {
                "icon": "📝",
                "title": "LossSummary Dataclass",
                "desc": "Holds metric learning diagnostic metrics:\n• loss: torch.Tensor (the scalar backward gradient source)\n• positive_distance: float (mean positive distance)\n• negative_distance: float (mean negative/hard-negative distance)"
            },
            {
                "icon": "🔀",
                "title": "Loss Dispatcher",
                "desc": "Routes calculations based on model choice:\n• Routes to contrastive_batch_loss(margin) for Siamese\n• Routes to batch_hard_triplet_loss(margin) for Triplet\n• Returns structured LossSummary"
            }
        ]
    },
    {
        "num": "9",
        "title": "Stage 9: Training & Validation",
        "files": "p09_training.py",
        "color": COLORS["magenta"],
        "hld": "Manages the training and validation epochs. Updates encoder model parameters, calculates validation EER and loss, handles early stopping to prevent overfitting, and saves the best model state checkpoints.",
        "lld_intro": "Integrates Mixed Precision (AMP), AdamW optimizer, gradient clipping, evaluation embedding harvesting, and early stopping.",
        "components": [
            {
                "icon": "🔄",
                "title": "AMP Train Step",
                "desc": "Runs single training epoch:\n• Loops batches, applies augmentation\n• Uses torch.autocast(device_type='cuda', dtype=float16)\n• Computes forward pass and metric loss\n• unscale_() & clips gradients: norm <= 1.0\n• Backpropagates and steps optimizer via GradScaler"
            },
            {
                "icon": "⚙️",
                "title": "Optimizer & Loader Setup",
                "desc": "Sets up optimization parameters:\n• Optimizer: AdamW (lr=3e-4 or 1e-3, weight_decay=1e-4)\n• Training loader uses balanced IdentityBatchSampler\n• Validation loader uses regular batching to compute validation loss"
            },
            {
                "icon": "📥",
                "title": "Embedding Harvester",
                "desc": "Extracts embeddings for biometric evaluation:\n• Runs model in evaluation mode under torch.no_grad()\n• Collects batch outputs, maps indices back to records\n• Returns np.ndarray of shape (N, 128) and list[FingerprintRecord]"
            },
            {
                "icon": "🛑",
                "title": "Early Stopping & Save",
                "desc": "Monitors validation performance:\n• Computes validation EER after each epoch\n• If EER improves: best_epoch = epoch, saves state_dict to {experiment_id}.pt\n• If no improvement for patience=2 epochs, breaks loop early"
            },
            {
                "icon": "💾",
                "title": "ExperimentResult",
                "desc": "Consolidates all run metrics:\n• experiment_id, hyperparams (lr, batch_size, margin)\n• history: list[EpochHistory] (loss, validation_eer, etc)\n• best_epoch, test_metrics, validation_metrics, checkpoint_path"
            }
        ]
    },
    {
        "num": "10",
        "title": "Stage 10: Biometric Evaluation",
        "files": "p10_phase_five_evaluation.py",
        "color": COLORS["indigo"],
        "hld": "Computes biometric verification performance (EER, FAR, FRR) by matching probe embeddings against enrolled master templates.",
        "lld_intro": "Extracts templates, computes L2 Euclidean distances, sweeps thresholds, and calculates FAR/FRR.",
        "components": [
            {
                "icon": "🔐",
                "title": "Template Enrollment",
                "desc": "Enrolls master templates:\n• Groups test embeddings by identity key\n• For each identity, selects the first 'real' print as master template. If no 'real' print, falls back to first available print\n• Remaining prints serve as probes"
            },
            {
                "icon": "✅",
                "title": "Genuine Match Scores",
                "desc": "Computes distances of matching fingers:\n• Computes L2 distance between master template and all remaining probes of the same finger\n• Genuine Score = ||template_id - probe_id||_2\n• Yields genuine distance score array"
            },
            {
                "icon": "❌",
                "title": "Impostor Match Scores",
                "desc": "Computes distances of non-matching fingers:\n• For each enrolled master template, randomly selects other identity templates (up to max_impostor_pairs_per_identity)\n• Impostor Score = ||template_id - template_other||_2\n• Yields impostor distance score array"
            },
            {
                "icon": "📈",
                "title": "FAR & FRR Sweeps",
                "desc": "Sweeps 512 thresholds from min to max:\n• Threshold range: [min(all_scores), max(all_scores)]\n• False Acceptance Rate: FAR = (impostor_scores <= threshold).mean()\n• False Rejection Rate: FRR = (genuine_scores > threshold).mean()"
            },
            {
                "icon": "🏆",
                "title": "EER & Accuracy Calculation",
                "desc": "Finds optimal biometric threshold:\n• Equal Error Rate: EER is threshold where |FAR - FRR| is minimized: (FAR + FRR) / 2.0\n• Accuracy: (correct_genuines + correct_impostors) / total\n• Returns VerificationMetrics dataclass"
            }
        ]
    },
    {
        "num": "11",
        "title": "Stage 11: Visualization Generator",
        "files": "p11_visualization.py",
        "color": COLORS["accent"],
        "hld": "Generates diagnostic visual artifacts for debugging and evaluation. Programmatically saves preprocessing strips, augmentation comparisons, and renders custom XML SVG files for training loss curves, ROC traces, and EER bar charts.",
        "lld_intro": "Converts PyTorch float tensors to uint8 PNGs, concatenates arrays, and programmatically generates raw XML SVG text.",
        "components": [
            {
                "icon": "🖼️",
                "title": "PNG Exporter",
                "desc": "Writes image files:\n• Clamps float32 tensor to [0.0, 1.0]\n• Scales to 255.0, rounds, and casts to torch.uint8\n• Exposes single-channel grayscale array\n• Writes to file path via torchvision.io.write_png()"
            },
            {
                "icon": "🎞️",
                "title": "Strip Generator",
                "desc": "Stitches image stages horizontally:\n• Concatenates image list along the width dimension: torch.cat(images, dim=2)\n• Outputs 'preprocess_strip.png' and 'augmentation_strip.png' comparison charts"
            },
            {
                "icon": "📈",
                "title": "SVG Line Chart",
                "desc": "Draws vector line charts:\n• Computes X, Y viewport mappings for datasets\n• Generates SVG path polyline coordinates\n• Draws horizontal grid lines, scale labels, axis headers, and color-coded legends (e.g. for loss curves)"
            },
            {
                "icon": "📊",
                "title": "SVG Bar Chart",
                "desc": "Draws vector bar charts:\n• Calculates bar widths and heights based on maximum data value\n• Renders rect elements with rounded corners (rx=6)\n• Writes data values above bars and rotates label texts -50 degrees"
            },
            {
                "icon": "📝",
                "title": "Visualizer Exporter",
                "desc": "Generates standard output files:\n• baseline_loss.svg: checks training/validation overfitting\n• best_roc.svg: plots True Positive Rate vs False Acceptance Rate\n• final_eer.svg: compares final test EER across preprocessing/models"
            }
        ]
    },
    {
        "num": "12",
        "title": "Stage 12: Runner & Orchestration",
        "files": "p12_experiment_runner.py",
        "color": COLORS["accent2"],
        "hld": "High-level driver script orchestrating the execution of the entire pipeline. Executes splits, coordinates baseline training, sweeps augmentation variants, searches hyperparameters, runs the final comparative grid, and exports a unified pipeline summary JSON file.",
        "lld_intro": "Implements run profiles, sequences the four core pipeline phases, and saves summaries and checkpoints.",
        "components": [
            {
                "icon": "🏁",
                "title": "Profile Configurator",
                "desc": "Overrides configs based on runtime profile:\n• 'smoke': validation checks (1 epoch, 12 steps/epoch, mini batch, 32 impostor pairs)\n• 'balanced': default configuration\n• 'full': robust sweep (baseline_epochs=3, train_steps=180)"
            },
            {
                "icon": "🔍",
                "title": "Phases A & B",
                "desc": "Baseline and Augmentation search:\n• Phase A: Trains baseline compact_siamese on raw images\n• Phase B: Sweeps all 5 augmentation variants using Gabor preprocessing to find the best augmentation variant"
            },
            {
                "icon": "⚙️",
                "title": "Phase C Hyperparameter Sweep",
                "desc": "Grid searches optimization parameters:\n• Sweeps learning rates (1e-3, 3e-4) x batch sizes (32, 48) x contrastive margins (0.75, 1.0) using best augmentation\n• Selects parameters minimizing validation EER"
            },
            {
                "icon": "📊",
                "title": "Phase D Final Grid",
                "desc": "Final comparative evaluation:\n• Trains 10 configurations: preprocess variants ('raw', 'segmented', 'normalized', 'gabor', 'skeleton') x models ('compact_siamese', 'mobile_triplet')\n• Employs optimal parameters found in Phase B/C"
            },
            {
                "icon": "💾",
                "title": "Pipeline Summary JSON",
                "desc": "Consolidates all experimental outputs:\n• Serializes dataset statistics, baseline metrics, sweeps histories, ranked final results, and asset paths into a unified 1.4MB JSON file."
            }
        ]
    }
]

# Build the Draw.io XML
mxfile = ET.Element("mxfile", {
    "host": "Electron",
    "modified": "2026-06-05T08:15:00Z",
    "agent": "5.0",
    "version": "22.1.2",
    "type": "device"
})

diagram = ET.SubElement(mxfile, "diagram", {
    "id": "fingerprint_pipeline_detailed_hld_lld",
    "name": "Fingerprint Pipeline Detailed HLD LLD"
})

# Enlarged width/height of page to fit expanded text sizes nicely
mxGraphModel = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "1800",
    "dy": "1000",
    "grid": "1",
    "gridSize": "10",
    "guides": "1",
    "tooltips": "1",
    "connect": "1",
    "arrows": "1",
    "fold": "1",
    "page": "1",
    "pageScale": "1",
    "pageWidth": "1960",
    "pageHeight": "6100",
    "math": "0",
    "shadow": "0"
})

root = ET.SubElement(mxGraphModel, "root")

# Base elements
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

# Add background style / large container to show it's dark themed
bg_rect = ET.SubElement(root, "mxCell", {
    "id": "bg_container",
    "value": "",
    "style": f"rounded=0;whiteSpace=wrap;html=1;fillColor={COLORS['bg']};strokeColor=none;opacity=100;",
    "vertex": "1",
    "parent": "1"
})
ET.SubElement(bg_rect, "mxGeometry", {
    "x": "0",
    "y": "0",
    "width": "1960",
    "height": "6000",
    "as": "geometry"
})

# Header title block
title_val = (
    f"<b><font size='6' color='{COLORS['text']}'>Fingerprint Biometric Verification Pipeline</font></b><br/>"
    f"<b><font size='4' color='{COLORS['accent2']}'>Detailed High-Level Design (HLD) &amp; Low-Level Design (LLD) Flow Map</font></b><br/><br/>"
    f"<font color='{COLORS['text_muted']}' size='2'>A complete engineering blueprint mapping all 12 Python files in <b>src/</b>, key data representations, functions, layers, metrics, and parameters.</font>"
)
title_box = ET.SubElement(root, "mxCell", {
    "id": "title_block",
    "value": title_val,
    "style": f"text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;",
    "vertex": "1",
    "parent": "1"
})
ET.SubElement(title_box, "mxGeometry", {
    "x": "380",
    "y": "40",
    "width": "1200",
    "height": "100",
    "as": "geometry"
})

# Create stages
current_y = 180
hld_box_width = 380
hld_box_height = 240
lld_box_width = 1350
lld_box_height = 360
vertical_gap = 120

hld_ids = []

for idx, stage in enumerate(STAGES):
    stage_y = current_y + idx * (lld_box_height + vertical_gap)
    hld_y = stage_y + (lld_box_height - hld_box_height) // 2
    
    # 1. HLD Box (Left Column)
    hld_id = f"hld_{idx}"
    hld_ids.append(hld_id)
    
    hld_content = (
        f"<b><font size='4' color='{stage['color']}'>{html.escape(stage['title'])}</font></b><br/>"
        f"<i><font size='2' color='{COLORS['text_muted']}'>File: {html.escape(stage['files'])}</font></i><br/><br/>"
        f"<div align='left'><font size='2' color='{COLORS['text']}'>{html.escape(stage['hld']).replace(chr(10), '<br/>')}</font></div>"
    )
    
    hld_style = (
        f"rounded=1;whiteSpace=wrap;html=1;fillColor={COLORS['surface']};"
        f"strokeColor={stage['color']};strokeWidth=3.5;fontColor={COLORS['text']};"
        f"align=center;verticalAlign=top;spacingLeft=15;spacingRight=15;spacingTop=15;"
    )
    
    hld_cell = ET.SubElement(root, "mxCell", {
        "id": hld_id,
        "value": hld_content,
        "style": hld_style,
        "vertex": "1",
        "parent": "1"
    })
    ET.SubElement(hld_cell, "mxGeometry", {
        "x": "80",
        "y": str(hld_y),
        "width": str(hld_box_width),
        "height": str(hld_box_height),
        "as": "geometry"
    })
    
    # 2. LLD Container (Right Column)
    lld_id = f"lld_{idx}"
    lld_content = (
        f"<b><font color='{stage['color']}' size='3'>Low-Level Design (LLD)</font></b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<font color='{COLORS['text_muted']}' size='2'>{html.escape(stage['lld_intro'])}</font>"
    )
    lld_style = (
        f"rounded=1;whiteSpace=wrap;html=1;fillColor={COLORS['bg3']};"
        f"strokeColor={COLORS['border2']};strokeWidth=2;fontColor={COLORS['text']};"
        f"align=left;verticalAlign=top;spacingLeft=20;spacingTop=15;"
    )
    
    lld_cell = ET.SubElement(root, "mxCell", {
        "id": lld_id,
        "value": lld_content,
        "style": lld_style,
        "vertex": "1",
        "parent": "1"
    })
    ET.SubElement(lld_cell, "mxGeometry", {
        "x": "520",
        "y": str(stage_y),
        "width": str(lld_box_width),
        "height": str(lld_box_height),
        "as": "geometry"
    })
    
    # 3. Connection: HLD to LLD Container (Dashed Horizontal Line)
    conn_hld_lld_id = f"conn_hl_{idx}"
    conn_hld_lld_style = (
        f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
        f"strokeColor={COLORS['text_muted']};strokeWidth=2;dashed=1;endArrow=none;"
    )
    conn_hld_lld = ET.SubElement(root, "mxCell", {
        "id": conn_hld_lld_id,
        "value": "",
        "style": conn_hld_lld_style,
        "edge": "1",
        "parent": "1",
        "source": hld_id,
        "target": lld_id
    })
    ET.SubElement(conn_hld_lld, "mxGeometry", {"relative": "1", "as": "geometry"})
    
    # 4. Add 5 LLD Sub-Boxes inside LLD Container
    comp_width = 250
    comp_height = 280
    comp_gap = 18
    comp_start_x = 18
    comp_y = 60
    
    for comp_idx, comp in enumerate(stage["components"]):
        comp_id = f"comp_{idx}_{comp_idx}"
        comp_val = (
            f"<b><font size='2' color='{stage['color']}'>{html.escape(comp['icon'] + ' ' + comp['title'])}</font></b><br/><br/>"
            f"<div align='left'><font size='1' color='{COLORS['text']}'>{html.escape(comp['desc']).replace(chr(10), '<br/>')}</font></div>"
        )
        comp_style = (
            f"rounded=1;whiteSpace=wrap;html=1;fillColor={COLORS['surface2']};"
            f"strokeColor={COLORS['border2']};strokeWidth=1;fontColor={COLORS['text']};"
            f"align=center;verticalAlign=top;spacingTop=12;spacingLeft=8;spacingRight=8;"
        )
        comp_cell = ET.SubElement(root, "mxCell", {
            "id": comp_id,
            "value": comp_val,
            "style": comp_style,
            "vertex": "1",
            "parent": lld_id  # Parent is the LLD container!
        })
        
        comp_x = comp_start_x + comp_idx * (comp_width + comp_gap)
        ET.SubElement(comp_cell, "mxGeometry", {
            "x": str(comp_x),
            "y": str(comp_y),
            "width": str(comp_width),
            "height": str(comp_height),
            "as": "geometry"
        })

# Add flow arrows between HLD stages
for i in range(len(hld_ids) - 1):
    arrow_id = f"flow_arrow_{i}"
    arrow_style = (
        f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
        f"strokeColor={STAGES[i]['color']};strokeWidth=3.5;fillColor=none;endArrow=classic;"
    )
    arrow = ET.SubElement(root, "mxCell", {
        "id": arrow_id,
        "value": f"Stage {i+2}",
        "style": arrow_style,
        "edge": "1",
        "parent": "1",
        "source": hld_ids[i],
        "target": hld_ids[i+1]
    })
    ET.SubElement(arrow, "mxGeometry", {"relative": "1", "as": "geometry"})

# Save file to both locations
tree = ET.ElementTree(mxfile)

# Write to root flow_map.drawio
with open("/media/pope/projecteo/github_proj/a_resume/fingerprint/flow_map.drawio", "wb") as f:
    tree.write(f, encoding="utf-8", xml_declaration=True)

# Write to web flow_map.drawio
with open("/media/pope/projecteo/github_proj/a_resume/fingerprint/web/flow_map.drawio", "wb") as f:
    tree.write(f, encoding="utf-8", xml_declaration=True)

print("Generated super detailed flow_maps successfully!")
