# Helper script to generate a rich uncompressed Draw.io XML file for the Fingerprint pipeline.
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
        "hld": "Global configuration storage using frozen Python dataclasses. Encapsulates dataset directories, neural network dimension layers, training hyperparameters, and experimental sweep grids.",
        "lld_intro": "Defines five distinct configuration schemas to guarantee end-to-end type safety and deterministic seeds.",
        "components": [
            {"icon": "⚙️", "title": "SplitConfig", "desc": "Defines dataset splits (Train=0.70, Val=0.15, Test=0.15) and global randomization seed=42."},
            {"icon": "📐", "title": "RuntimeConfig", "desc": "Specifies input size (96x103), embedding size (128-D), batch size (48), and GPU hardware pin memory configuration."},
            {"icon": "📝", "title": "TrainingConfig", "desc": "Configures learning rates (1e-3, 3e-4), triplet margins (0.75, 1.0), and early stopping patience (2)."},
            {"icon": "🔀", "title": "ExperimentGrid", "desc": "Sets up the parametric sweeps across 5 preprocess levels, 5 augmentation options, and 2 architectures."},
            {"icon": "📂", "title": "AppConfig", "desc": "Root settings manager resolving relative paths (e.g. archive/SOCOFing) and validating directory existence."}
        ]
    },
    {
        "num": "2",
        "title": "Stage 2: BMP Decoder",
        "files": "p02_bmp_reader.py",
        "color": COLORS["accent2"],
        "hld": "Custom low-level BMP image reader implementing binary struct parsing. Bypasses external image processing libraries to operate purely on standard library primitives.",
        "lld_intro": "Decodes 8-bit paletted, 24-bit RGB, and 32-bit BGRA images directly into normalized float32 grayscale tensors.",
        "components": [
            {"icon": "🔍", "title": "Header Parser", "desc": "Unpacks BMP header bytes to extract pixel offset, DIB size, width, height, planes, bpp, and compression details."},
            {"icon": "📐", "title": "Row Stride Calc", "desc": "Handles 4-byte alignment padding. Formula: row_stride = ((width * bpp + 31) // 32) * 4."},
            {"icon": "🎨", "title": "8-bit Decoder", "desc": "Reads palette color table, computes grayscale average of each color entry, and maps indices to grayscale pixels."},
            {"icon": "🖼️", "title": "Direct Decoder", "desc": "Decodes 24/32-bit pixel matrices. Converts color channels using NTSC formula: 0.299*R + 0.587*G + 0.114*B."},
            {"icon": "🔢", "title": "Output Format", "desc": "Scales pixels to [0.0, 1.0] and wraps them in a PyTorch tensor of shape (1, H, W) float32."}
        ]
    },
    {
        "num": "3",
        "title": "Stage 3: Indexing & Parsing",
        "files": "p03_dataset_index.py",
        "color": COLORS["green"],
        "hld": "Iterates through raw dataset folders, parses file path metadata, and catalogs all available prints into a structured index.",
        "lld_intro": "Scans 55,270 files, parses metadata fields, detects altered variants, and registers identities.",
        "components": [
            {"icon": "🏷️", "title": "Regex Metadata", "desc": "Parses filename: subject_id (int), gender (M/F), hand (Left/Right), finger (thumb/index/middle/ring/little)."},
            {"icon": "⚠️", "title": "Altered Flags", "desc": "If path contains '/Altered/' -> source='altered', difficulty = parent folder (Easy/Med/Hard), else source='real'."},
            {"icon": "📦", "title": "Data Record", "desc": "Encapsulates metadata in FingerprintRecord dataclass containing subject, gender, hand, finger, path, alteration details."},
            {"icon": "📂", "title": "Dataset Indexer", "desc": "Walks dataset path, filters files, instantiates records, and returns list[FingerprintRecord]."},
            {"icon": "🔑", "title": "Identity Key", "desc": "Generates a unique finger identifier key: '{subject_id}__{gender}_{hand}_{finger}' to track unique fingerprints."}
        ]
    },
    {
        "num": "4",
        "title": "Stage 4: Identity-Disjoint Split",
        "files": "p04_phase_one_split.py",
        "color": COLORS["yellow"],
        "hld": "Splits the indexed dataset into Train (70%), Val (15%), and Test (15%) partitions at the identity level to prevent data leakage.",
        "lld_intro": "Groups files by identity, shuffles identities, and writes a split manifest CSV with no leakage across sets.",
        "components": [
            {"icon": "🔀", "title": "Stratified Split", "desc": "Groups by identity key, shuffles identities with seed 42, and distributes them (70/15/15 ratio)."},
            {"icon": "🚫", "title": "Leakage Policy", "desc": "Guarantees all altered versions (Easy/Medium/Hard) of a subject are grouped in the same split as the real scan."},
            {"icon": "📄", "title": "Split CSV", "desc": "Saves split mapping to artifacts/splits/identity_split.csv: (split, identity_key, file_name, source, difficulty)."},
            {"icon": "⚖️", "title": "Identity Sampler", "desc": "IdentityBatchSampler groups B/4 identities, sampling 4 prints each to guarantee genuine pairs in every batch."},
            {"icon": "⚡", "title": "DataLoader", "desc": "Creates PyTorch DataLoaders with pinned memory, multi-worker prefetching, and custom collation wrappers."}
        ]
    },
    {
        "num": "5",
        "title": "Stage 5: Image Preprocessing",
        "files": "p05_phase_two_preprocessing.py",
        "color": COLORS["blue"],
        "hld": "Enhances ridge clarity and removes background noise. Computes 5 distinct preprocessed variants of every image.",
        "lld_intro": "Pipeline stages: Raw -> Segmented -> Normalized -> Gabor Enhanced -> Skeletonized.",
        "components": [
            {"icon": "✂️", "title": "Segmentation", "desc": "Divides image into 8x8 blocks, computes local variance, and masks out background blocks (threshold = 0.35 * max)."},
            {"icon": "📐", "title": "Normalization", "desc": "Adjusts segmented foreground pixels dynamically to target mean = 0.5 and standard deviation = 0.22."},
            {"icon": "🌀", "title": "Orientation Field", "desc": "Computes Sobel horizontal/vertical gradients, estimates local ridge angles for Gabor filters."},
            {"icon": "🌿", "title": "Gabor Filter Bank", "desc": "Applies a bank of 6 oriented Gabor kernels (kernel size=11, σ=3, λ=5.5, γ=0.6) to enhance ridge lines."},
            {"icon": "🦴", "title": "Skeletonization", "desc": "Applies adaptive binarization followed by Zhang-Suen thinning to reduce ridge lines to 1-pixel width."}
        ]
    },
    {
        "num": "6",
        "title": "Stage 6: Data Augmentation",
        "files": "p06_phase_three_augmentation.py",
        "color": COLORS["orange"],
        "hld": "Applies stochastic geometric and structural transforms during training to improve spatial alignment and rotation robustness.",
        "lld_intro": "Supports 5 augmentation variants: none, rotate, elastic, obliterate, and combined.",
        "components": [
            {"icon": "🔄", "title": "Affine Rotation", "desc": "Applies random rotation up to ±45° using bilinear grid sampling in PyTorch spatial grid generator."},
            {"icon": "🌊", "title": "Elastic Distortion", "desc": "Generates 8x8 random displacement field, upsamples/smoothes it (alpha=6.0), and warps the ridge grid."},
            {"icon": "⬛", "title": "Obliteration", "desc": "Overlays random black rectangular patches (10% to 22% of short side) to simulate scars, cuts, or dirt."},
            {"icon": "🔀", "title": "Combined Transform", "desc": "Sequentially applies rotation -> elastic deformation -> obliteration. Found to be too destructive."},
            {"icon": "⚙️", "title": "Aug Wrapper", "desc": "Wraps dataset loading to dynamically apply selected augmentation variant on-the-fly during training."}
        ]
    },
    {
        "num": "7",
        "title": "Stage 7: Model Architectures",
        "files": "p07_phase_four_models.py",
        "color": COLORS["red"],
        "hld": "Two deep neural network feature extractors that map 103x96 input tensors to normalized 128-dimensional embeddings.",
        "lld_intro": "Architectures compared: compact_siamese (~118K parameters) vs mobile_triplet (~31K parameters).",
        "components": [
            {"icon": "🧠", "title": "compact_siamese", "desc": "4 Conv2d layers (1->24->48->72->96), BN, GELU, MaxPool2d (3x), AdaptiveAvgPool2d, Linear(96->128)."},
            {"icon": "📱", "title": "mobile_triplet", "desc": "Conv2d (1->16) followed by 4 Depthwise Separable blocks (16->24->48->72->96) with stride-2 downsampling."},
            {"icon": "⚡", "title": "DW-Separable Conv", "desc": "Splits standard conv into 3x3 depthwise conv (groups=in_channels) and 1x1 pointwise conv. Reduces parameters 4x."},
            {"icon": "🌐", "title": "L2 Normalization", "desc": "Normalizes outputs to unit length: x = x / ||x||_2. Restricts embedding space to a hypersphere."},
            {"icon": "📐", "title": "Parameters", "desc": "compact_siamese: 118,632 parameters. mobile_triplet: 31,048 parameters. Both export 128-D vectors."}
        ]
    },
    {
        "num": "8",
        "title": "Stage 8: Metric Learning Loss",
        "files": "p08_metric_learning.py",
        "color": COLORS["cyan"],
        "hld": "Calculates pairwise distances in embedding space and updates network weights using metric learning objectives.",
        "lld_intro": "Contrastive Loss pulls/pushes pairs. Batch-Hard Triplet Loss mines hardest samples in a batch.",
        "components": [
            {"icon": "📏", "title": "Pairwise L2 Dist", "desc": "Computes Euclidean distances between all embeddings in a batch: d = sqrt(||x_i - x_j||_2^2 + eps)."},
            {"icon": "🔗", "title": "Contrastive Loss", "desc": "Pulls same-finger pairs together; pushes different-finger pairs apart if d < margin (margin=0.75)."},
            {"icon": "📐", "title": "Contrastive Math", "desc": "Loss = (1 - y) * d^2 + y * max(0, margin - d)^2 (y=1 for negative pair, y=0 for positive pair)."},
            {"icon": "🎯", "title": "BatchHard Triplet", "desc": "For each anchor in batch, extracts hardest positive (max AP) and hardest negative (min AN)."},
            {"icon": "📝", "title": "Triplet Math", "desc": "Loss = max(0, d_hardest_pos - d_hardest_neg + margin). Averaged across all anchors in batch."}
        ]
    },
    {
        "num": "9",
        "title": "Stage 9: Training & Validation",
        "files": "p09_training.py",
        "color": COLORS["magenta"],
        "hld": "Executes training epochs, updates model parameters, checks validation performance, and saves checkpoints.",
        "lld_intro": "Orchestrates training iterations with mixed precision, gradient scaling, and early stopping.",
        "components": [
            {"icon": "🔄", "title": "Train Loop", "desc": "Runs forward pass, computes loss, scales gradients, performs backward pass, and clips grad norm to 1.0."},
            {"icon": "⚡", "title": "AMP Float16", "desc": "Uses torch.cuda.amp (autocast & GradScaler) to execute forward computations in float16 for speed."},
            {"icon": "⚙️", "title": "Optimization", "desc": "Updates weights with AdamW (lr=3e-4, weight_decay=1e-4). Monitors validation EER after each epoch."},
            {"icon": "🛑", "title": "Early Stopping", "desc": "Stops training if validation EER fails to improve for patience=2 epochs. Restores best weights."},
            {"icon": "💾", "title": "Checkpointing", "desc": "Saves best model parameters to artifacts/checkpoints/ with experiment-encoded filenames."}
        ]
    },
    {
        "num": "10",
        "title": "Stage 10: Biometric Evaluation",
        "files": "p10_phase_five_evaluation.py",
        "color": COLORS["indigo"],
        "hld": "Calculates matching performance metrics: False Acceptance Rate (FAR), False Rejection Rate (FRR), and Equal Error Rate (EER).",
        "lld_intro": "Enrolls template fingerprints and matches them against genuine and impostor probe prints.",
        "components": [
            {"icon": "🔐", "title": "Enrollment", "desc": "For each test identity, enrolls the first real print scan as the master reference template embedding."},
            {"icon": "✅", "title": "Genuine Scores", "desc": "Calculates L2 distance between master template and all remaining prints of the same finger (7,405 pairs)."},
            {"icon": "❌", "title": "Impostor Scores", "desc": "Calculates L2 distance between master template and prints of other identities (115,200 pairs)."},
            {"icon": "📈", "title": "Threshold Sweep", "desc": "Sweeps 512 thresholds from min_dist to max_dist. Computes FAR and FRR at each step."},
            {"icon": "🏆", "title": "EER Calculation", "desc": "EER is the point where |FAR - FRR| is minimized. Best EER: 0.145% (compact_siamese + raw)."}
        ]
    },
    {
        "num": "11",
        "title": "Stage 11: Visualization Generator",
        "files": "p11_visualization.py",
        "color": COLORS["accent"],
        "hld": "Renders pipeline diagnostic files, including preprocessing strips, augmentation comparisons, and performance curves.",
        "lld_intro": "Exports PNG image strips and XML/SVG vector charts of ROC curves and model metrics to the artifacts folder.",
        "components": [
            {"icon": "🖼️", "title": "PNG Strip Stacking", "desc": "Concatenates preprocessing steps and augmentation variants along the width dimension for comparison strips."},
            {"icon": "🔢", "title": "Grayscale Conversion", "desc": "Maps float32 tensors back to uint8 pixel bytes [0, 255] before writing files using write_png."},
            {"icon": "📈", "title": "SVG Line Chart", "desc": "Renders vector charts showing Training/Validation loss and ROC curves with coordinate mapping."},
            {"icon": "📊", "title": "SVG Bar Chart", "desc": "Programmatically draws bar charts showing EER comparisons across different preprocessing variants."},
            {"icon": "📝", "title": "Summary Writer", "desc": "Saves vector visualizations to custom file names under artifacts/visualizations/ for the web frontend."}
        ]
    },
    {
        "num": "12",
        "title": "Stage 12: Runner & Orchestration",
        "files": "p12_experiment_runner.py",
        "color": COLORS["accent2"],
        "hld": "High-level pipeline driver orchestrating all experiment phases and exporting final consolidated summaries.",
        "lld_intro": "Runs baseline validation, augmentation grid searches, hyperparameter grids, and final 10-way comparison sweeps.",
        "components": [
            {"icon": "🏁", "title": "Phase A Baseline", "desc": "Trains a baseline Siamese network on raw inputs to establish primary metrics."},
            {"icon": "🔍", "title": "Phase B Search", "desc": "Sweeps all 5 augmentation options, identifying 'none' (no augmentation) as the top performer."},
            {"icon": "⚙️", "title": "Phase C Search", "desc": "Sweeps combinations of learning rates, batch sizes, and contrastive loss margins."},
            {"icon": "📊", "title": "Phase D Grid", "desc": "Sweeps all 10 combinations of models and preprocessing variants to identify the optimal pipeline."},
            {"icon": "💾", "title": "Consolidated JSON", "desc": "Serializes histories, best epoch metadata, and FAR/FRR lists into pipeline_summary.json (1.4MB)."}
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
    "pageWidth": "1600",
    "pageHeight": "4800",
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
    "width": "1560",
    "height": "4650",
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
    "x": "180",
    "y": "40",
    "width": "1200",
    "height": "100",
    "as": "geometry"
})

# Create stages
current_y = 180
hld_box_width = 320
hld_box_height = 140
lld_box_width = 1000
lld_box_height = 260
vertical_gap = 100

hld_ids = []

for idx, stage in enumerate(STAGES):
    stage_y = current_y + idx * (lld_box_height + vertical_gap)
    hld_y = stage_y + (lld_box_height - hld_box_height) // 2
    
    # 1. HLD Box (Left Column)
    hld_id = f"hld_{idx}"
    hld_ids.append(hld_id)
    
    hld_content = (
        f"<b><font size='4' color={stage['color']}>{stage['title']}</font></b><br/>"
        f"<i><font size='2' color='{COLORS['text_muted']}'>File: {stage['files']}</font></i><br/><br/>"
        f"<div align='left'><font size='2' color='{COLORS['text']}'>{stage['hld']}</font></div>"
    )
    
    hld_style = (
        f"rounded=1;whiteSpace=wrap;html=1;fillColor={COLORS['surface']};"
        f"strokeColor={stage['color']};strokeWidth=3.5;fontColor={COLORS['text']};"
        f"align=center;verticalAlign=middle;spacingLeft=10;spacingRight=10;"
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
        f"<font color='{COLORS['text_muted']}' size='2'>{stage['lld_intro']}</font>"
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
        "x": "480",
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
    comp_width = 176
    comp_height = 160
    comp_gap = 18
    comp_start_x = 18
    comp_y = 80
    
    for comp_idx, comp in enumerate(stage["components"]):
        comp_id = f"comp_{idx}_{comp_idx}"
        comp_val = (
            f"<b><font size='2' color='{stage['color']}'>{comp['icon']} {comp['title']}</font></b><br/><br/>"
            f"<div align='left'><font size='1' color='{COLORS['text']}'>{comp['desc']}</font></div>"
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

print("Generated detailed flow_maps successfully!")
