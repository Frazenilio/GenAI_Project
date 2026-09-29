import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from bridge import LatentBridge, LatentPatchBridge, SplitEfficientNet, BridgedDetector
from latent_dataset import LatentDataset

### AI
# Import the original detector class to instantiate the pre-trained EfficientNet
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
detector_dir = os.path.join(project_root, 'synthetic-image-detection')
sys.path.append(project_root)
sys.path.append(detector_dir)
try:
    from synthetic_image_detection.test_real_vs_synthetic_singleimg import RealvsSyntheticDetector
except ImportError:
    # Handle the hyphenated folder name
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_real_vs_synthetic_singleimg", 
            os.path.join(os.path.dirname(__file__), '..', 'synthetic-image-detection', 'test_real_vs_synthetic_singleimg.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    RealvsSyntheticDetector = module.RealvsSyntheticDetector

### END AI

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_random_patches(latent, num_patches=800, patch_size=12):
    ## We don't need the channels
    _, H, W = latent.shape
    patches = []
    for _ in range(num_patches):
        ## Take a random point
        x = random.randint(0, W - patch_size)
        y = random.randint(0, H - patch_size)
        ## Create the patch starting from the random point
        patch = latent[:, y:y+patch_size, x:x+patch_size]
        patches.append(patch)
    return torch.stack(patches, dim=0)

def train_epoch(model, dataloader, criterion, optimizer, device, variant, args):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for latents, labels, _ in pbar:
        # labels are 0 (real) or 1 (fake)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if variant == 'single':
            latents = latents.to(device)
            ## Let's get the scores from the last two nodes
            logits = model(latents)
            ## Compute loss
            loss = criterion(logits, labels)
            
            ## backprop
            loss.backward()
            ## update all weights
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        elif variant == 'patching':
            batch_loss = 0
            batch_correct = 0
            ## For patching, we process one latent (image) at a time because each produces many patches
            ## This is why we set batch size of 1
            for i in range(latents.size(0)):
                latent = latents[i]
                label = labels[i].unsqueeze(0) # [1]
                
                ## Extract *num_patches* 12x12 @ 256
                patches = extract_random_patches(latent, args.num_patches, args.patch_size).to(device)
                
                ## Forward pass all patches through the bridge and split detector
                logits = model(patches)
                
                ## Aggregate patch scores using majority voting (like the original detector)
                patch_scores = logits[:, 1]
                ## Average the top M scores
                img_score = torch.mean(torch.sort(patch_scores)[0][-args.M:])
                
                ## We are using Cross Entropy Loss. Cross Entropy expects the true label
                ## and the two final scores. Given that we are using patches, logits
                ## are not just two scores but num_patches * 2 scores. So we average them
                ## to get the mean so we can actually use it
                top_m_indices = torch.argsort(patch_scores)[-args.M:]
                img_logits = torch.mean(logits[top_m_indices], dim=0, keepdim=True) # [1, 2]
                
                ## Here's just like the single variant
                loss = criterion(img_logits, label)
                loss.backward()
                
                batch_loss += loss.item()
                _, predicted = torch.max(img_logits.data, 1)
                batch_correct += (predicted == label).sum().item()
                
            optimizer.step()
            
            total += latents.size(0)
            correct += batch_correct
            loss = torch.tensor(batch_loss / latents.size(0)) # just for reporting
            
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{100.*correct/total:.2f}%"})
        
    return running_loss / len(dataloader), correct / total

def val_epoch(model, dataloader, criterion, device, variant, args):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for latents, labels, _ in pbar:
            labels = labels.to(device)
            
            if variant == 'single':
                latents = latents.to(device)
                ## Detector score the img
                ## Logits are [RealScore, FakeScore]
                logits = model(latents)
                loss = criterion(logits, labels)
                
                ## Take the highest value: if the score for Fake is higher, it is predicted to be fake
                ## First arg is the actual score (not interested) while second is the idx of the 
                ## highest value. This means that it is our label
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                
            elif variant == 'patching':
                ## Iterate all image to process them on at a time
                for i in range(latents.size(0)):
                    latent = latents[i]
                    label = labels[i].unsqueeze(0)
                    
                    ## Patch as described
                    patches = extract_random_patches(latent, args.num_patches, args.patch_size).to(device)
                    ## Predict
                    logits = model(patches)
                    
                    patch_scores = logits[:, 1]
                    ## Take the top M: the original architecture used the top ones as well
                    ## so we do the same to copy the "patch voting"
                    top_m_indices = torch.argsort(patch_scores)[-args.M:]
                    ## Average scores
                    img_logits = torch.mean(logits[top_m_indices], dim=0, keepdim=True)
                    
                    loss = criterion(img_logits, label)
                    running_loss += loss.item()
                    
                    _, predicted = torch.max(img_logits.data, 1)
                    correct += (predicted == label).sum().item()
                total += latents.size(0)
                
    return running_loss / len(dataloader), correct / total

def main():
    ### AI
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dir', type=str, required=True, help='Path to the directory containing train/val folders')
    parser.add_argument('--variant', type=str, choices=['single', 'patching'], required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Where to save the trained model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--M', type=int, default=600, help='Top M patches for voting')
    parser.add_argument('--num_patches', type=int, default=800, help='Number of patches to extract')
    parser.add_argument('--patch_size', type=int, default=12, help='Latent patch spatial size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quality', type=str, default=None, help='Filter latents by quality suffix (e.g. 012)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ### END AI

    ## Load Data
    train_dir = os.path.join(args.latent_dir, 'train')
    val_dir = os.path.join(args.latent_dir, 'val')
    
    train_dataset = LatentDataset(train_dir, quality=args.quality)
    val_dataset = LatentDataset(val_dir, quality=args.quality)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
    
    ## Setup Model
    ### AI
    print("Loading original detector weights...")
    original_cwd = os.getcwd()
    os.chdir(detector_dir)
    original_detector = RealvsSyntheticDetector(device=device)
    os.chdir(original_cwd)
    split_detector = SplitEfficientNet(original_detector.net)
    
    if args.variant == 'single':
        bridge = LatentBridge()
    else:
        bridge = LatentPatchBridge()
        
    model = BridgedDetector(bridge, split_detector).to(device)
    
    ### END AI

    ## Being only fake or real it's cross entropy
    criterion = nn.CrossEntropyLoss()
    ## Optimize the bridge parameters, not the split detector
    optimizer = optim.Adam(bridge.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    
    ## Training Loop
    print(f"Starting training for {args.epochs} epochs ({args.variant} variant)...")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args.variant, args)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device, args.variant, args)
        
        ### AI
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.checkpoint_dir, f'best_bridge_{args.variant}.pth')
            torch.save(bridge.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")

        ### END AI

if __name__ == '__main__':
    main()
