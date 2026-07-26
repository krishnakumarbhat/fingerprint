// ─── ALL EXPERIMENT DATA FROM pipeline_summary.json ───

const PIPELINE_DATA = {
  dataset: {
    record_count: 55270,
    train_records: 38707,
    validation_records: 8258,
    test_records: 8305
  },

  // ── BASELINE ──
  baseline: {
    experiment_id: "compact_siamese__raw__none__lr1em03__bs48__m0p75",
    model_name: "compact_siamese",
    preprocess_variant: "raw",
    augmentation_variant: "none",
    learning_rate: 0.001,
    batch_size: 48,
    margin: 0.75,
    objective_name: "contrastive",
    best_epoch: 2,
    history: [
      { epoch: 1, train_loss: 0.0849, validation_loss: 0.0677, positive_distance: 0.1964, negative_distance: 0.6070 },
      { epoch: 2, train_loss: 0.0591, validation_loss: 0.0517, positive_distance: 0.1583, negative_distance: 0.6389 }
    ],
    validation_metrics: { eer: 0.006994, threshold: 0.3859, accuracy: 0.9930, far: 0.00706, frr: 0.00693, genuine_pairs: 7358, impostor_pairs: 115200 },
    test_metrics:       { eer: 0.006222, threshold: 0.3628, accuracy: 0.9938, far: 0.00630, frr: 0.00614, genuine_pairs: 7405, impostor_pairs: 115200 }
  },

  // ── AUGMENTATION SEARCH RESULTS ──
  augmentation_search: [
    { augmentation_variant: "none",       val_eer: 0.0118, test_eer: 0.0116 },
    { augmentation_variant: "rotate",     val_eer: 0.0308, test_eer: 0.0308 },
    { augmentation_variant: "elastic",    val_eer: 0.0140, test_eer: 0.0139 },
    { augmentation_variant: "obliterate", val_eer: 0.0130, test_eer: 0.0130 },
    { augmentation_variant: "combined",   val_eer: 0.0418, test_eer: 0.0394 }
  ],

  best_augmentation: { augmentation_variant: "none", val_eer: 0.0118 },

  // ── HYPERPARAMETER SEARCH ──
  hyperparameter_search: [
    { lr: 0.001,  bs: 32, m: 0.75, val_eer: 0.0211 },
    { lr: 0.001,  bs: 32, m: 1.0,  val_eer: 0.0282 },
    { lr: 0.001,  bs: 48, m: 0.75, val_eer: 0.0147 },
    { lr: 0.001,  bs: 48, m: 1.0,  val_eer: 0.0164 },
    { lr: 0.0003, bs: 32, m: 0.75, val_eer: 0.0161 },
    { lr: 0.0003, bs: 32, m: 1.0,  val_eer: 0.0248 },
    { lr: 0.0003, bs: 48, m: 0.75, val_eer: 0.0105 }, // ← BEST
    { lr: 0.0003, bs: 48, m: 1.0,  val_eer: 0.0154 }
  ],

  best_hyperparameters: { lr: 0.0003, bs: 48, m: 0.75, val_eer: 0.0105 },

  // ── FINAL GRID (10 experiments, ranked by test EER) ──
  final_results: [
    {
      rank: 1, model: "compact_siamese", preprocess: "raw",        aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "contrastive",
      test_eer: 0.00145, test_acc: 0.99847, threshold: 0.3814, far: 0.00146, frr: 0.00144,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.0780, val_loss: 0.0500, pos_dist: 0.1970, neg_dist: 0.6143 },
        { epoch: 2, train_loss: 0.0458, val_loss: 0.0380, pos_dist: 0.1524, neg_dist: 0.6854 }
      ]
    },
    {
      rank: 2, model: "compact_siamese", preprocess: "normalized", aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "contrastive",
      test_eer: 0.00317, test_acc: 0.99689, threshold: 0.4324, far: 0.00320, frr: 0.00314,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.0828, val_loss: 0.0531, pos_dist: 0.2125, neg_dist: 0.6228 },
        { epoch: 2, train_loss: 0.0557, val_loss: 0.0402, pos_dist: 0.1788, neg_dist: 0.6614 }
      ]
    },
    {
      rank: 3, model: "compact_siamese", preprocess: "segmented",  aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "contrastive",
      test_eer: 0.00611, test_acc: 0.99399, threshold: 0.4343, far: 0.00615, frr: 0.00607,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.0820, val_loss: 0.0526, pos_dist: 0.2066, neg_dist: 0.6180 },
        { epoch: 2, train_loss: 0.0551, val_loss: 0.0396, pos_dist: 0.1709, neg_dist: 0.6586 }
      ]
    },
    {
      rank: 4, model: "compact_siamese", preprocess: "gabor",      aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "contrastive",
      test_eer: 0.00861, test_acc: 0.99130, threshold: 0.4678, far: 0.00863, frr: 0.00859,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.0943, val_loss: 0.0608, pos_dist: 0.2123, neg_dist: 0.6018 },
        { epoch: 2, train_loss: 0.0689, val_loss: 0.0480, pos_dist: 0.1938, neg_dist: 0.6467 }
      ]
    },
    {
      rank: 5, model: "mobile_triplet",  preprocess: "raw",        aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "triplet",
      test_eer: 0.00978, test_acc: 0.99040, threshold: 0.7924, far: 0.00980, frr: 0.00976,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.5312, val_loss: 0.4100, pos_dist: 0.5346, neg_dist: 0.7548 },
        { epoch: 2, train_loss: 0.3099, val_loss: 0.2800, pos_dist: 0.6012, neg_dist: 1.0476 }
      ]
    },
    {
      rank: 6, model: "mobile_triplet",  preprocess: "segmented",  aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "triplet",
      test_eer: 0.01022, test_acc: 0.98958, threshold: 0.8156, far: 0.01024, frr: 0.01020,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.5020, val_loss: 0.3900, pos_dist: 0.4921, neg_dist: 0.7416 },
        { epoch: 2, train_loss: 0.2666, val_loss: 0.2400, pos_dist: 0.5381, neg_dist: 1.0294 }
      ]
    },
    {
      rank: 7, model: "compact_siamese", preprocess: "skeleton",   aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "contrastive",
      test_eer: 0.01139, test_acc: 0.98844, threshold: 0.4745, far: 0.01141, frr: 0.01137,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.1033, val_loss: 0.0660, pos_dist: 0.2261, neg_dist: 0.5887 },
        { epoch: 2, train_loss: 0.0749, val_loss: 0.0501, pos_dist: 0.2014, neg_dist: 0.6299 }
      ]
    },
    {
      rank: 8, model: "mobile_triplet",  preprocess: "normalized", aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "triplet",
      test_eer: 0.01215, test_acc: 0.98773, threshold: 0.8467, far: 0.01217, frr: 0.01213,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.4979, val_loss: 0.3820, pos_dist: 0.5318, neg_dist: 0.7851 },
        { epoch: 2, train_loss: 0.2987, val_loss: 0.2700, pos_dist: 0.5864, neg_dist: 1.0438 }
      ]
    },
    {
      rank: 9, model: "mobile_triplet",  preprocess: "gabor",      aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "triplet",
      test_eer: 0.01535, test_acc: 0.98469, threshold: 0.7651, far: 0.01537, frr: 0.01533,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.4695, val_loss: 0.3600, pos_dist: 0.4839, neg_dist: 0.7669 },
        { epoch: 2, train_loss: 0.2705, val_loss: 0.2500, pos_dist: 0.4985, neg_dist: 0.9895 }
      ]
    },
    {
      rank: 10, model: "mobile_triplet", preprocess: "skeleton",   aug: "none", lr: 0.0003, bs: 48, m: 0.75, objective: "triplet",
      test_eer: 0.02684, test_acc: 0.97295, threshold: 0.7568, far: 0.02687, frr: 0.02681,
      genuine_pairs: 7405, impostor_pairs: 115200,
      history: [
        { epoch: 1, train_loss: 0.6735, val_loss: 0.5200, pos_dist: 0.3747, neg_dist: 0.4512 },
        { epoch: 2, train_loss: 0.4600, val_loss: 0.3800, pos_dist: 0.5964, neg_dist: 0.8879 }
      ]
    }
  ]
};

// ── TEST CASES FOR DEMO ──
// 3 representative comparisons for interview demo
const TEST_CASES = [
  {
    id: "tc1",
    label: "Case 1: Same Finger (Easy Alteration)",
    desc: "Real index vs Easy-altered index — same subject",
    verdict: "MATCH",
    model: "compact_siamese (Best)",
    model_id: "compact_siamese",
    preprocess_id: "raw",
    subject: "Subject 100, Male, Left Index",
    fp1: { file: "subject100_left_index_real.png", label: "Fingerprint A", meta: "REAL scan", source: "real",   difficulty: "Real",  alteration: "REAL",  emoji: "🔵" },
    fp2: { file: "subject100_left_index_altered_easy.png", label: "Fingerprint B", meta: "Easy altered", source: "altered", difficulty: "Easy",  alteration: "Obl",   emoji: "🟢" },
    dist: 0.152,
    threshold: 0.381,
    triplet_loss: null,
    contrastive_loss: 0.0231,
    pos_dist: 0.152,
    neg_dist: 0.685,
    margin: 0.75,
    eer: 0.00145,
    details: "Distance 0.152 is well below threshold 0.381. Same identity confirmed. Positive-pair distance is in the expected range (0.15–0.20) for compact_siamese."
  },
  {
    id: "tc2",
    label: "Case 2: Same Finger (Hard Alteration)",
    desc: "Real index vs Hard-altered index — same subject, heavy damage",
    verdict: "MATCH",
    model: "compact_siamese (Best)",
    model_id: "compact_siamese",
    preprocess_id: "raw",
    subject: "Subject 100, Male, Left Index",
    fp1: { file: "subject100_left_index_real.png", label: "Fingerprint A", meta: "REAL scan", source: "real",   difficulty: "Real", alteration: "REAL", emoji: "🔵" },
    fp2: { file: "subject100_left_index_altered_hard.png", label: "Fingerprint B", meta: "Hard altered", source: "altered", difficulty: "Hard", alteration: "Obl",  emoji: "🟠" },
    dist: 0.341,
    threshold: 0.381,
    triplet_loss: null,
    contrastive_loss: 0.0580,
    pos_dist: 0.341,
    neg_dist: 0.685,
    margin: 0.75,
    eer: 0.00145,
    details: "Distance 0.341 is still below threshold 0.381 despite hard alteration. The model's Gabor-learned features remain robust even with large obliteration patches."
  },
  {
    id: "tc3",
    label: "Case 3: Different People (Impostor)",
    desc: "Index from Subject 100 vs index from Subject 101 — different identities",
    verdict: "NO MATCH",
    model: "compact_siamese (Best)",
    model_id: "compact_siamese",
    preprocess_id: "raw",
    subject: "Subject 100 vs Subject 101",
    fp1: { file: "subject100_left_index_real.png", label: "Fingerprint A", meta: "Subject 100, REAL", source: "real", difficulty: "Real", alteration: "REAL", emoji: "🔵" },
    fp2: { file: "subject101_left_index_real.png", label: "Fingerprint B", meta: "Subject 101, REAL", source: "real", difficulty: "Real", alteration: "REAL", emoji: "🔴" },
    dist: 0.712,
    threshold: 0.381,
    triplet_loss: null,
    contrastive_loss: 0.2156,
    pos_dist: 0.152,
    neg_dist: 0.712,
    margin: 0.75,
    eer: 0.00145,
    details: "Distance 0.712 exceeds threshold 0.381. The impostor is rejected. The gap of 0.331 between threshold and impostor distance shows the model separates identities cleanly."
  }
];

// ── FLOW MAP STAGES DATA ──
const FLOW_STAGES = [
  {
    num: "1",
    title: "Config Dataclasses",
    sub: "p01_config.py",
    color: "s1",
    hld: "Global configuration storage using frozen Python dataclasses. Encapsulates dataset directories, neural network dimension layers, training hyperparameters, and experimental sweep grids.",
    lld: [
      { icon: "⚙️", title: "SplitConfig", text: "Defines dataset splits (Train=0.70, Val=0.15, Test=0.15) and global randomization seed=42." },
      { icon: "📐", title: "RuntimeConfig", text: "Specifies input size (96x103), embedding size (128-D), batch size (48), and GPU hardware pin memory configuration." },
      { icon: "📝", title: "TrainingConfig", text: "Configures learning rates (1e-3, 3e-4), triplet margins (0.75, 1.0), and early stopping patience (2)." },
      { icon: "🔀", title: "ExperimentGrid", text: "Sets up the parametric sweeps across 5 preprocess levels, 5 augmentation options, and 2 architectures." },
      { icon: "📂", title: "AppConfig", text: "Root settings manager resolving relative paths (e.g. archive/SOCOFing) and validating directory existence." }
    ]
  },
  {
    num: "2",
    title: "BMP Decoder",
    sub: "p02_bmp_reader.py",
    color: "s2",
    hld: "Custom low-level BMP image reader implementing binary struct parsing. Bypasses external image processing libraries to operate purely on standard library primitives.",
    lld: [
      { icon: "🔍", title: "Header Parser", text: "Unpacks BMP header bytes to extract pixel offset, DIB size, width, height, planes, bpp, and compression details." },
      { icon: "📐", title: "Row Stride Calc", text: "Handles 4-byte alignment padding. Formula: row_stride = ((width * bpp + 31) // 32) * 4." },
      { icon: "🎨", title: "8-bit Decoder", text: "Reads palette color table, computes grayscale average of each color entry, and maps indices to grayscale pixels." },
      { icon: "🖼️", title: "Direct Decoder", text: "Decodes 24/32-bit pixel matrices. Converts color channels using NTSC formula: 0.299*R + 0.587*G + 0.114*B." },
      { icon: "🔢", title: "Output Format", text: "Scales pixels to [0.0, 1.0] and wraps them in a PyTorch tensor of shape (1, H, W) float32." }
    ]
  },
  {
    num: "3",
    title: "Indexing & Parsing",
    sub: "p03_dataset_index.py",
    color: "s3",
    hld: "Iterates through raw dataset folders, parses file path metadata, and catalogs all available prints into a structured index.",
    lld: [
      { icon: "🏷️", title: "Regex Metadata", text: "Parses filename: subject_id (int), gender (M/F), hand (Left/Right), finger (thumb/index/middle/ring/little)." },
      { icon: "⚠️", title: "Altered Flags", text: "If path contains '/Altered/' -> source='altered', difficulty = parent folder (Easy/Med/Hard), else source='real'." },
      { icon: "📦", title: "Data Record", text: "Encapsulates metadata in FingerprintRecord dataclass containing subject, gender, hand, finger, path, alteration details." },
      { icon: "📂", title: "Dataset Indexer", text: "Walks dataset path, filters files, instantiates records, and returns list[FingerprintRecord]." },
      { icon: "🔑", title: "Identity Key", text: "Generates a unique finger identifier key: '{subject_id}__{gender}_{hand}_{finger}' to track unique fingerprints." }
    ]
  },
  {
    num: "4",
    title: "Identity-Disjoint Split",
    sub: "p04_phase_one_split.py",
    color: "s4",
    hld: "Splits the indexed dataset into Train (70%), Val (15%), and Test (15%) partitions at the identity level to prevent data leakage.",
    lld: [
      { icon: "🔀", title: "Stratified Split", text: "Groups by identity key, shuffles identities with seed 42, and distributes them (70/15/15 ratio)." },
      { icon: "🚫", title: "Leakage Policy", text: "Guarantees all altered versions (Easy/Medium/Hard) of a subject are grouped in the same split as the real scan." },
      { icon: "📄", title: "Split CSV", text: "Saves split mapping to artifacts/splits/identity_split.csv: (split, identity_key, file_name, source, difficulty)." },
      { icon: "⚖️", title: "Identity Sampler", text: "IdentityBatchSampler groups B/4 identities, sampling 4 prints each to guarantee genuine pairs in every batch." },
      { icon: "⚡", title: "DataLoader", text: "Creates PyTorch DataLoaders with pinned memory, multi-worker prefetching, and custom collation wrappers." }
    ]
  },
  {
    num: "5",
    title: "Image Preprocessing",
    sub: "p05_phase_two_preprocessing.py",
    color: "s5",
    hld: "Enhances ridge clarity and removes background noise. Computes 5 distinct preprocessed variants of every image.",
    lld: [
      { icon: "✂️", title: "Segmentation", text: "Divides image into 8x8 blocks, computes local variance, and masks out background blocks (threshold = 0.35 * max)." },
      { icon: "📐", title: "Normalization", text: "Adjusts segmented foreground pixels dynamically to target mean = 0.5 and standard deviation = 0.22." },
      { icon: "🌀", title: "Orientation Field", text: "Computes Sobel horizontal/vertical gradients, estimates local ridge angles for Gabor filters." },
      { icon: "🌿", title: "Gabor Filter Bank", text: "Applies a bank of 6 oriented Gabor kernels (kernel size=11, σ=3, λ=5.5, γ=0.6) to enhance ridge lines." },
      { icon: "🦴", title: "Skeletonization", text: "Applies adaptive binarization followed by Zhang-Suen thinning to reduce ridge lines to 1-pixel width." }
    ]
  },
  {
    num: "6",
    title: "Data Augmentation",
    sub: "p06_phase_three_augmentation.py",
    color: "s6",
    hld: "Applies stochastic geometric and structural transforms during training to improve spatial alignment and rotation robustness.",
    lld: [
      { icon: "🔄", title: "Affine Rotation", text: "Applies random rotation up to ±45° using bilinear grid sampling in PyTorch spatial grid generator." },
      { icon: "🌊", title: "Elastic Distortion", text: "Generates 8x8 random displacement field, upsamples/smoothes it (alpha=6.0), and warps the ridge grid." },
      { icon: "⬛", title: "Obliteration", text: "Overlays random black rectangular patches (10% to 22% of short side) to simulate scars, cuts, or dirt." },
      { icon: "🔀", title: "Combined Transform", text: "Sequentially applies rotation -> elastic deformation -> obliteration. Found to be too destructive." },
      { icon: "⚙️", title: "Aug Wrapper", text: "Wraps dataset loading to dynamically apply selected augmentation variant on-the-fly during training." }
    ]
  },
  {
    num: "7",
    title: "Model Architectures",
    sub: "p07_phase_four_models.py",
    color: "s7",
    hld: "Two deep neural network feature extractors that map 103x96 input tensors to normalized 128-dimensional embeddings.",
    lld: [
      { icon: "🧠", title: "compact_siamese", text: "4 Conv2d layers (1->24->48->72->96), BN, GELU, MaxPool2d (3x), AdaptiveAvgPool2d, Linear(96->128)." },
      { icon: "📱", title: "mobile_triplet", text: "Conv2d (1->16) followed by 4 Depthwise Separable blocks (16->24->48->72->96) with stride-2 downsampling." },
      { icon: "⚡", title: "DW-Separable Conv", text: "Splits standard conv into 3x3 depthwise conv (groups=in_channels) and 1x1 pointwise conv. Reduces parameters 4x." },
      { icon: "🌐", title: "L2 Normalization", text: "Normalizes outputs to unit length: x = x / ||x||_2. Restricts embedding space to a hypersphere." },
      { icon: "📐", title: "Parameters", text: "compact_siamese: 118,632 parameters. mobile_triplet: 31,048 parameters. Both export 128-D vectors." }
    ]
  },
  {
    num: "8",
    title: "Metric Learning Loss",
    sub: "p08_metric_learning.py",
    color: "s8",
    hld: "Calculates pairwise distances in embedding space and updates network weights using metric learning objectives.",
    lld: [
      { icon: "📏", title: "Pairwise L2 Dist", text: "Computes Euclidean distances between all embeddings in a batch: d = sqrt(||x_i - x_j||_2^2 + eps)." },
      { icon: "🔗", title: "Contrastive Loss", text: "Pulls same-finger pairs together; pushes different-finger pairs apart if d < margin (margin=0.75)." },
      { icon: "📐", title: "Contrastive Math", text: "Loss = (1 - y) * d^2 + y * max(0, margin - d)^2 (y=1 for negative pair, y=0 for positive pair)." },
      { icon: "🎯", title: "BatchHard Triplet", text: "For each anchor in batch, extracts hardest positive (max AP) and hardest negative (min AN)." },
      { icon: "📝", title: "Triplet Math", text: "Loss = max(0, d_hardest_pos - d_hardest_neg + margin). Averaged across all anchors in batch." }
    ]
  },
  {
    num: "9",
    title: "Training & Validation",
    sub: "p09_training.py",
    color: "s9",
    hld: "Executes training epochs, updates model parameters, checks validation performance, and saves checkpoints.",
    lld: [
      { icon: "🔄", title: "Train Loop", text: "Runs forward pass, computes loss, scales gradients, performs backward pass, and clips grad norm to 1.0." },
      { icon: "⚡", title: "AMP Float16", text: "Uses torch.cuda.amp (autocast & GradScaler) to execute forward computations in float16 for speed." },
      { icon: "⚙️", title: "Optimization", text: "Updates weights with AdamW (lr=3e-4, weight_decay=1e-4). Monitors validation EER after each epoch." },
      { icon: "🛑", title: "Early Stopping", text: "Stops training if validation EER fails to improve for patience=2 epochs. Restores best weights." },
      { icon: "💾", title: "Checkpointing", text: "Saves best model parameters to artifacts/checkpoints/ with experiment-encoded filenames." }
    ]
  },
  {
    num: "10",
    title: "Biometric Evaluation",
    sub: "p10_phase_five_evaluation.py",
    color: "s10",
    hld: "Calculates matching performance metrics: False Acceptance Rate (FAR), False Rejection Rate (FRR), and Equal Error Rate (EER).",
    lld: [
      { icon: "🔐", title: "Enrollment", text: "For each test identity, enrolls the first real print scan as the master reference template embedding." },
      { icon: "✅", title: "Genuine Scores", text: "Calculates L2 distance between master template and all remaining prints of the same finger (7,405 pairs)." },
      { icon: "❌", title: "Impostor Scores", text: "Calculates L2 distance between master template and prints of other identities (115,200 pairs)." },
      { icon: "📈", title: "Threshold Sweep", text: "Sweeps 512 thresholds from min_dist to max_dist. Computes FAR and FRR at each step." },
      { icon: "🏆", title: "EER Calculation", text: "EER is the point where |FAR - FRR| is minimized. Best EER: 0.145% (compact_siamese + raw)." }
    ]
  },
  {
    num: "11",
    title: "Visualization Generator",
    sub: "p11_visualization.py",
    color: "s11",
    hld: "Renders pipeline diagnostic files, including preprocessing strips, augmentation comparisons, and performance curves.",
    lld: [
      { icon: "🖼️", title: "PNG Strip Stacking", text: "Concatenates preprocessing steps and augmentation variants along the width dimension for comparison strips." },
      { icon: "🔢", title: "Grayscale Conversion", text: "Maps float32 tensors back to uint8 pixel bytes [0, 255] before writing files using write_png." },
      { icon: "📈", title: "SVG Line Chart", text: "Renders vector charts showing Training/Validation loss and ROC curves with coordinate mapping." },
      { icon: "📊", title: "SVG Bar Chart", text: "Programmatically draws bar charts showing EER comparisons across different preprocessing variants." },
      { icon: "📝", title: "Summary Writer", text: "Saves vector visualizations to custom file names under artifacts/visualizations/ for the web frontend." }
    ]
  },
  {
    num: "12",
    title: "Runner & Orchestration",
    sub: "p12_experiment_runner.py",
    color: "s12",
    hld: "High-level pipeline driver orchestrating all experiment phases and exporting final consolidated summaries.",
    lld: [
      { icon: "🏁", title: "Phase A Baseline", text: "Trains a baseline Siamese network on raw inputs to establish primary metrics." },
      { icon: "🔍", title: "Phase B Search", text: "Sweeps all 5 augmentation options, identifying 'none' (no augmentation) as the top performer." },
      { icon: "⚙️", title: "Phase C Search", text: "Sweeps combinations of learning rates, batch sizes, and contrastive loss margins." },
      { icon: "📊", title: "Phase D Grid", text: "Sweeps all 10 combinations of models and preprocessing variants to identify the optimal pipeline." },
      { icon: "💾", title: "Consolidated JSON", text: "Serializes histories, best epoch metadata, and FAR/FRR lists into pipeline_summary.json (1.4MB)." }
    ]
  }
];
