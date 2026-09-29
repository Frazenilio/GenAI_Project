import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class LatentDataset(Dataset):
    ## Root dir is the directory used to get the latents (quality specifies which ones to get)
    def __init__(self, root_dir: str, quality: str = None):
        self.root_dir = Path(root_dir)
        self.quality = quality
        self.samples = []
        
        # Label mapping: real=0, fake=1 (matching the detector's classification logic)
        classes = {'real': 0, 'fake': 1}
        
        for cls_name, label in classes.items():
            ## Hardcoded path to get the "latent" directory (JPEG AI produce it when requested)
            latent_dir = self.root_dir / cls_name / 'latent'
            if not latent_dir.exists():
                print(f"Warning: {latent_dir} does not exist.")
                continue
                
            ## Iterate all images
            for pt_file in latent_dir.glob('*.pt'):
                ## If requesting a specific quality, check it's the one otherwise skip it
                if self.quality and not pt_file.stem.endswith(f"_{self.quality}"):
                    continue
                ## Add file path and actual label to the samples
                self.samples.append((str(pt_file), label))
                
        print(f"Loaded {len(self.samples)} latent samples from {self.root_dir}" + (f" (quality: {self.quality})" if self.quality else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        try:
            ### AI
            # Use map_location='cpu' to prevent CUDA device mismatch errors during loading
            latent = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
            
            if isinstance(latent, dict):
                for k, v in latent.items():
                    if isinstance(v, torch.Tensor):
                        latent = v
                        break

            latent = latent.float()
            
            ## If the latent comes with a batch dimension of 1 (e.g., [1, 256, 64, 64]),
            ## we squeeze it out to [256, 64, 64] so DataLoader can batch it properly.
            if latent.dim() == 4 and latent.shape[0] == 1:
                latent = latent.squeeze(0)
                
            ### END AI
            
            ## Resize the latent to exactly 64x64 if it isn't already (e.g., 80x80)
            ## Without this, we can't handle different latent dimension 
            if latent.shape[-2:] != (64, 64):
                # F.interpolate requires a batch dimension, so we unsqueeze then squeeze
                latent = F.interpolate(latent.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
                
            return latent, label, Path(file_path).stem
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a zero tensor as a fallback to avoid crashing the epoch
            return torch.zeros((256, 64, 64)), label, Path(file_path).stem
