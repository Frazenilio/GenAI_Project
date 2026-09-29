import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bridge import LatentBridge, LatentPatchBridge, SplitEfficientNet, BridgedDetector
from latent_dataset import LatentDataset
from train_bridge import extract_random_patches

### AI
# Import original detector class
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
detector_dir = os.path.join(project_root, 'synthetic-image-detection')
sys.path.append(project_root)
sys.path.append(detector_dir)
try:
    from synthetic_image_detection.test_real_vs_synthetic_singleimg import RealvsSyntheticDetector
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_real_vs_synthetic_singleimg", 
            os.path.join(os.path.dirname(__file__), '..', 'synthetic-image-detection', 'test_real_vs_synthetic_singleimg.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    RealvsSyntheticDetector = module.RealvsSyntheticDetector

### END AI

def evaluate(model, dataloader, device, variant, args):
    model.eval()
    results = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        ## Iterating batches
        for latents, labels, filenames in pbar:
            labels = labels.to(device)
            
            ## Variant single means no patching
            if variant == 'single':
                latents = latents.to(device)
                ## Logits are the results of the model, the last two nodes scores.
                logits = model(latents) # [B, 2]
                
                ## We get the score for the fake (they are in the gpu so we move them)
                scores = logits[:, 1].cpu().numpy()
                ## We get the index of the highest score (so the most probable one)
                ## This format makes it coherent with the labels
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                ## Get the actual label
                lbls = labels.cpu().numpy()
                
                ## Add the results for each image in the batch
                for i in range(len(filenames)):
                    results.append({
                        'Filename': filenames[i],
                        'True_Label': 'fake' if lbls[i] == 1 else 'real',
                        'Pred_Label': 'fake' if preds[i] == 1 else 'real',
                        'Score': scores[i],
                        'Correct': bool(preds[i] == lbls[i])
                    })
                    
            ## Patching variant
            elif variant == 'patching':
                ## No image batching since a lot of patching may be used (and takes a lot
                ## of memory)
                for i in range(latents.size(0)):
                    latent = latents[i]
                    lbl = labels[i].item()
                    filename = filenames[i]
                    
                    ## Use the train_bridge method to exrtact patches
                    patches = extract_random_patches(latent, args.num_patches, args.patch_size).to(device)
                    ## Get the logits for each patch, treated as they were batches
                    logits = model(patches)
                    
                    ## Get all scores
                    patch_scores = logits[:, 1]
                    ## Compute the total score as an avg
                    img_score = torch.mean(torch.sort(patch_scores)[0][-args.M:]).item()
                    
                    pred_class = 1 if img_score > 0 else 0
                    
                    ## Same appending as for the single variant
                    results.append({
                        'Filename': filename,
                        'True_Label': 'fake' if lbl == 1 else 'real',
                        'Pred_Label': 'fake' if pred_class == 1 else 'real',
                        'Score': img_score,
                        'Correct': bool(pred_class == lbl)
                    })
                    
    return results

def main():
    ### AI
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test split directory (contains real/fake)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained bridge weights (.pth)')
    parser.add_argument('--variant', type=str, choices=['single', 'patching'], required=True)
    parser.add_argument('--output_csv', type=str, default='bridge_eval_results.csv')
    parser.add_argument('--M', type=int, default=600)
    parser.add_argument('--num_patches', type=int, default=800)
    parser.add_argument('--patch_size', type=int, default=12)
    parser.add_argument('--quality', type=str, default=None, help='Filter latents by quality suffix (e.g. 012)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ### END AI
       
    ## Create dataset
    test_dataset = LatentDataset(args.test_dir, quality=args.quality)
    test_loader = DataLoader(test_dataset, batch_size=16 if args.variant == 'single' else 1, shuffle=False)
    
    ### AI
    original_cwd = os.getcwd()
    os.chdir(detector_dir)
    original_detector = RealvsSyntheticDetector(device=device)
    os.chdir(original_cwd)
    ### END AI
    
    split_detector = SplitEfficientNet(original_detector.net)
    
    ## Construct the bridge based on required variant
    if args.variant == 'single':
        bridge = LatentBridge()
    else:
        bridge = LatentPatchBridge()
        
    ## Load the weights
    print(f"Loading trained bridge weights from {args.checkpoint}...")
    bridge.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    
    ## Prepare the bridge
    model = BridgedDetector(bridge, split_detector).to(device)
    
    results = evaluate(model, test_loader, device, args.variant, args)
    
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    
    ### AI
    acc = df['Correct'].mean() * 100
    real_acc = df[df['True_Label'] == 'real']['Correct'].mean() * 100
    fake_acc = df[df['True_Label'] == 'fake']['Correct'].mean() * 100
    
    print(f"\nEvaluation Complete! Results saved to {args.output_csv}")
    print(f"Overall Accuracy: {acc:.2f}%")
    print(f"Real Images Accuracy: {real_acc:.2f}%")
    print(f"Fake Images Accuracy: {fake_acc:.2f}%")

    ### END AI

if __name__ == '__main__':
    main()
