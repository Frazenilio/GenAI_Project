import sys
import os
import torch
import torch.nn as nn

## For using the detector in the VM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## No-Patching Bridge
class LatentBridge(nn.Module):
    def __init__(self):
        super().__init__()
        ## Latent Input: 64x64 @ 256
        ## We use the 10th block of EfficientNetB4 as Input: 12x12 @ 56 (because of patching dimension)
        
        self.bridge = nn.Sequential(
            ## 64x64 @ 256 -> 32x32 @ 128
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            ## 32x32 @ 128 -> 16x16 @ 64
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            ## 16x16 @ 64 -> 16x16 @ 56
            nn.Conv2d(64, 56, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
            
            ## 16x16 @ 56 -> 12x12 @ 56
            nn.AdaptiveAvgPool2d((12, 12))
        )

    def forward(self, x):
        return self.bridge(x)


class LatentPatchBridge(nn.Module):
    def __init__(self):
        super().__init__()
        ## We assyme the input is a 12x12 @ 256, this is the patching dimension
        ## To mantain coherence, we inject it directly to the 10th block of EfficientNetB4
        ## because it expects an input of 12x12 @ 56, this means we only have to change
        ## the channels numbers.
        
        self.bridge = nn.Sequential(
            ## 12x12 @ 256 -> 12x12 @ 128
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            ## 12x12 @ 128 -> 12x12 @ 56
            nn.Conv2d(128, 56, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bridge(x)

## https://arxiv.org/pdf/1905.11946, EfficientNetB4 architecture
class SplitEfficientNet(nn.Module):
    def __init__(self, full_net):
        super().__init__()
        ## We have to "decompose" the EfficientNet in order to inject to the desired block
        self.efficientnet = full_net.efficientnet
        self.classifier = full_net.classifier
        
        ## We assume injection at block 10 (12x12 @ 56).
        # EfficientNet-B4 blocks: 0-1 (48c), 2-5 (24c), 6-9 (32c), 10-13 (56c)...
        self.start_block = 10
        
        ## Freeze all parameters in the detector, we don't have to train it
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        # OVERRIDE: We NEVER want the frozen detector to be in training mode.
        super().train(False)
        return self

    def forward(self, x):
        ### AI
        ## x is the latent after the bridge
        ## Pass through the efficient next blocks
        for idx, block in enumerate(self.efficientnet._blocks):
            if idx >= self.start_block:
                x = block(x)
                
        # Pass through Head
        x = self.efficientnet._conv_head(x)
        x = self.efficientnet._bn1(x)
        x = self.efficientnet._swish(x)
        
        # Global Average Pooling & Flatten
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        
        # Dropout & Classification
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        
        ### END AI
        return x


## This is the whole architecture combined, using a bridge and the splitted efficient net
class BridgedDetector(nn.Module):
    def __init__(self, bridge_module, split_detector):
        super().__init__()
        self.bridge = bridge_module
        self.split_detector = split_detector
        
    def forward(self, x):
        ## x is the latent OR the patch. This is handled by the bridge anyway
        features = self.bridge(x)
        ## At the end of the detector, we have the logits to tell us if a latent referred to a real or fake image
        logits = self.split_detector(features)
        return logits
