import json
from pathlib import Path

summary_path = Path("/media/pope/projecteo/github_proj/a_resume/fingerprint/artifacts/summaries/pipeline_summary.json")
with open(summary_path, "r") as f:
    data = json.load(f)

final_results = data.get("final_results", [])
print(f"Total final results: {len(final_results)}")

for idx, res in enumerate(final_results):
    model = res.get("model_name")
    prep = res.get("preprocess_variant")
    aug = res.get("augmentation_variant")
    lr = res.get("learning_rate")
    bs = res.get("batch_size")
    margin = res.get("margin")
    
    val_m = res.get("validation_metrics", {})
    test_m = res.get("test_metrics", {})
    
    print(f"{idx}: Model: {model:16s} | Preprocess: {prep:10s} | Augment: {aug:8s} | lr: {lr} | bs: {bs} | margin: {margin}")
    print(f"   Val EER: {val_m.get('eer'):.4f} | Test EER: {test_m.get('eer'):.4f} | Test Threshold: {test_m.get('threshold'):.4f} | Test Acc: {test_m.get('accuracy'):.4f}")
    print(f"   Checkpoint: {res.get('checkpoint_path')}")
