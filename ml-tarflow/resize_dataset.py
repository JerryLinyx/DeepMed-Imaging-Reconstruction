
import torch
import argparse
import pathlib

def resize_dataset():
    parser = argparse.ArgumentParser(description="Resize PET dataset tensor")
    parser.add_argument('--input', type=pathlib.Path, default='data/pet/data_re.pt', help='Input path to original .pt file')
    parser.add_argument('--output', type=pathlib.Path,  help='Output path for resized .pt file')
    parser.add_argument('--size', type=int, required=True, help='Target number of samples (e.g. 12000)')
    
    args = parser.parse_args()

    # Determine output path if not specified
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_{args.size}{args.input.suffix}"

    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist.")
        return

    print(f"Loading data from {args.input}...")
    data = torch.load(args.input)
    
    print(f"Original shape: {data.shape}")
    
    current_size = data.shape[0]
    if args.size > current_size:
        print(f"Warning: Requested size {args.size} is larger than original size {current_size}. returning.")
        return

    # Randomly select indices
    perm = torch.randperm(current_size)
    indices = perm[:args.size]
    
    new_data = data[indices]
    
    print(f"New shape: {new_data.shape}")
    
    # Ensure output path includes size if not manually specified
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_{args.size}{args.input.suffix}"
    
    print(f"Saving to {args.output}...")
    torch.save(new_data, args.output)
    print("Done!")

if __name__ == "__main__":
    resize_dataset()
