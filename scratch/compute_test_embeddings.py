import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import functional as TF

from src.p07_phase_four_models import create_model
from src.p05_phase_two_preprocessing import apply_preprocess_variant

def main():
    project_root = Path("/media/pope/projecteo/github_proj/a_resume/fingerprint")
    checkpoints_dir = project_root / "artifacts/checkpoints/final_grid"
    images_dir = project_root / "web/data/test_prints"
    output_js = project_root / "web/embeddings.js"
    
    # Check images exist
    image_paths = list(images_dir.glob("*.png"))
    print(f"Found {len(image_paths)} test PNG images.")
    for p in image_paths:
        print(f"  - {p.name}")
        
    # Checkpoints to load
    # Format: (model_name, preprocess_variant, checkpoint_filename)
    models_config = [
        ("compact_siamese", "raw", "compact_siamese__raw__none__lr3em04__bs48__m0p75.pt"),
        ("compact_siamese", "normalized", "compact_siamese__normalized__none__lr3em04__bs48__m0p75.pt"),
        ("compact_siamese", "segmented", "compact_siamese__segmented__none__lr3em04__bs48__m0p75.pt"),
        ("compact_siamese", "gabor", "compact_siamese__gabor__none__lr3em04__bs48__m0p75.pt"),
        ("compact_siamese", "skeleton", "compact_siamese__skeleton__none__lr3em04__bs48__m0p75.pt"),
        ("mobile_triplet", "raw", "mobile_triplet__raw__none__lr3em04__bs48__m0p75.pt"),
        ("mobile_triplet", "segmented", "mobile_triplet__segmented__none__lr3em04__bs48__m0p75.pt"),
        ("mobile_triplet", "normalized", "mobile_triplet__normalized__none__lr3em04__bs48__m0p75.pt"),
        ("mobile_triplet", "gabor", "mobile_triplet__gabor__none__lr3em04__bs48__m0p75.pt"),
        ("mobile_triplet", "skeleton", "mobile_triplet__skeleton__none__lr3em04__bs48__m0p75.pt"),
    ]
    
    embeddings_data = {}
    
    # Process each image
    for img_path in image_paths:
        img_key = img_path.stem  # e.g. "subject100_left_index_real"
        embeddings_data[img_key] = {}
        
        # Load image in grayscale
        pil_img = Image.open(img_path).convert("L")
        img_tensor = TF.to_tensor(pil_img)  # Shape (1, H, W), range [0, 1]
        
        # Ensure correct shape
        if img_tensor.shape != (1, 103, 96):
            print(f"Warning: {img_path.name} has shape {img_tensor.shape}, expected (1, 103, 96)")
            
        print(f"Processing {img_key}...")
        
        for model_name, preprocess, ckpt_name in models_config:
            ckpt_path = checkpoints_dir / ckpt_name
            if not ckpt_path.exists():
                print(f"Error: checkpoint {ckpt_path} not found.")
                continue
                
            # Instantiate model
            model, objective = create_model(model_name, 128)
            # Load state dict
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            
            # Apply preprocessing
            preprocessed_tensor = apply_preprocess_variant(img_tensor, preprocess)
            # Shape should be (1, 1, 103, 96) for batch
            if preprocessed_tensor.dim() == 3:
                preprocessed_tensor = preprocessed_tensor.unsqueeze(0)
                
            # Forward pass
            with torch.no_grad():
                embedding = model(preprocessed_tensor)  # Shape (1, 128)
                
            # Flatten to list of floats
            emb_list = embedding.view(-1).tolist()
            model_key = f"{model_name}__{preprocess}"
            embeddings_data[img_key][model_key] = emb_list
            
    # Write to a javascript file
    js_content = f"// Precomputed 128D embeddings for the 6 test fingerprints\n"
    js_content += f"const PRECOMPUTED_EMBEDDINGS = {json.dumps(embeddings_data, indent=2)};\n"
    
    with open(output_js, "w") as f:
        f.write(js_content)
        
    print(f"Successfully computed embeddings and wrote to {output_js}")

if __name__ == "__main__":
    main()
