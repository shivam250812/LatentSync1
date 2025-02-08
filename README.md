# LatentSync

LatentSync is a lip-syncing framework that enhances video quality and synchronization.

## Super-Resolution Enhancement

To improve the quality of the generated lip-synced frames, you can apply super-resolution using GFPGAN or CodeFormer. Use the `--superres` parameter in the `inference.sh` script to specify the desired method.

### Usage

```bash
./inference.sh --superres GFPGAN
```

Replace `GFPGAN` with `CodeFormer` to use the CodeFormer model.

### Implementation Steps

#### 1. Modify `inference.sh` to Include Super-Resolution Parameter

Add a new parameter `--superres` to specify the super-resolution method:

```bash
#!/bin/bash

# Default super-resolution method is none
SUPERRES="none"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --superres) SUPERRES="$2"; shift ;;
        # Add other parameters here
    esac
    shift
done

# Run the inference script with the specified super-resolution method
python inference.py --superres $SUPERRES # Add other parameters as needed
```

#### 2. Update `inference.py` to Apply Super-Resolution

In your `inference.py` script, integrate the super-resolution process after generating the lip-synced frames:

```python
import argparse
from super_resolution import apply_super_resolution

def main():
    parser = argparse.ArgumentParser(description="LatentSync Inference")
    parser.add_argument('--superres', type=str, default='none', choices=['none', 'GFPGAN', 'CodeFormer'], help='Super-resolution method to use')
    # Add other arguments as needed
    args = parser.parse_args()

    # Your existing inference code here

    if args.superres != 'none':
        apply_super_resolution(generated_frames, method=args.superres)

if __name__ == "__main__":
    main()
```

#### 3. Implement the Super-Resolution Functionality

Create a new file named `super_resolution.py` to handle the super-resolution process:

```python
import cv2
import numpy as np
from gfpgan import GFPGANer
from codeformer import CodeFormer

def apply_super_resolution(frames, method='GFPGAN'):
    if method == 'GFPGAN':
        # Initialize GFPGAN
        gfpgan = GFPGANer()
        enhanced_frames = [gfpgan.enhance(frame) for frame in frames]
    elif method == 'CodeFormer':
        # Initialize CodeFormer
        codeformer = CodeFormer()
        enhanced_frames = [codeformer.enhance(frame) for frame in frames]
    else:
        raise ValueError(f"Unsupported super-resolution method: {method}")

    return enhanced_frames
```

### Requirements

Ensure that the required dependencies for GFPGAN and CodeFormer are installed. You can add them to your `requirements.txt` file:

```
torch
gfpgan
codeformer
```

Alternatively, provide installation instructions in your `README.md`.

### Testing

After making these changes, thoroughly test the implementation to ensure that the super-resolution enhancement works as expected and integrates seamlessly with the existing LatentSync workflow.
