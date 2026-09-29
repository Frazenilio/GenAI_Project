import torch

file_path = r"c:\Users\Franc\Documents\UniBS\Generative AI for Media\Project\JAI_1196_012.pt"

try:
    # Load the PyTorch file
    data = torch.load(file_path, weights_only=False, map_location=torch.device('cpu'))
    
    # Check if it's a tensor or a dictionary containing tensors
    if isinstance(data, torch.Tensor):
        print(f"File contains a single Tensor.")
        print(f"Dimension (Shape): {data.shape}")
        print(f"Data Type: {data.dtype}")
    elif isinstance(data, dict):
        print(f"File contains a Dictionary with keys:")
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                print(f"  - '{key}': Tensor shape {val.shape}, dtype {val.dtype}")
            else:
                print(f"  - '{key}': Type {type(val)}")
    else:
        print(f"File contains an unknown object of type: {type(data)}")
        
except Exception as e:
    print(f"Error loading the .pt file: {e}")
