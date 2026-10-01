import torch
from transformers import AutoModel
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path
import argparse
import numpy as np
import h5py

parser = argparse.ArgumentParser()

# Input options
PROJECT_DIR = Path(__file__).resolve().parent.parent

parser.add_argument('--output_image_dir', default=PROJECT_DIR / 'output/images',
    help="The directory where all images are stored, features shoudl be generated from.")
parser.add_argument('--output_dir', default=PROJECT_DIR / 'output/features',
    help="THe directory where the generated features shoudl be stored in.")
parser.add_argument('--model_id', default="facebook/dinov3_vitb16_pretrain_lvd1689m",
    help="The model ID to use for feature extraction.")
parser.add_argument('--checkpoint', default=PROJECT_DIR / 'checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    help="The path to the model checkpoint to use for feature extraction.")
parser.add_argument('--repo_dir', default=PROJECT_DIR.parent / 'dinov3',
    help="The path to the repository directory containing the DINOv3 model code.")


def main(args):
    IMAGE_DIR = args.output_image_dir
    OUTPUT_DIR = args.output_dir
    CHECKPOINT = args.checkpoint
    REPO_DIR = args.repo_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ── Load model ───────────────────────────────────────────────────────────
    model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=CHECKPOINT)
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    patch_size = model.patch_size                   # 16 for vits16

    # ── Process each image ───────────────────────────────────────────────────
    def extract_features(image_path):
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

        with torch.inference_mode():
            features = model.forward_features(x)

        # Patch-level features — spatial, one vector per image region
        patch_features = features['x_norm_patchtokens']   # [1, num_patches, 384]

        # Global image embedding — single vector summarizing the whole image
        cls_token = features['x_norm_clstoken']            # [1, 384]

        return patch_features.squeeze(0), cls_token.squeeze(0)

    # ── Run on all PNGs ──────────────────────────────────────────────────────
    count = 0
    for fname in sorted(os.listdir(str(IMAGE_DIR))):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(IMAGE_DIR, fname)
        patch_feat, cls_feat = extract_features(img_path)

        stem = os.path.splitext(fname)[0]
        torch.save({
            "patch_features": patch_feat,
            "cls_token":      cls_feat,      # shape: [384]
        }, os.path.join(OUTPUT_DIR, f"{stem}_dino.pt"))

        print(f"{fname}:  patch_features={patch_feat.shape}, cls_token={cls_feat.shape}")
        
        if count >= 2500:
            break
        
        count = count + 1
        

    print("Done! Features saved to", OUTPUT_DIR)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)