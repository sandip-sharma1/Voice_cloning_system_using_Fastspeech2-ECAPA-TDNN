# convert_single_ecapa_to_pt.py
# Convert one specific .ecapa_averaged_embedding file to .pt

import os
import torch
import sys

def convert_ecapa_to_pt(ecapa_path: str):
    """
    Convert a single .ecapa_averaged_embedding file to .pt
    """
    if not os.path.exists(ecapa_path):
        print(f"❌ Error: File not found!\n   {ecapa_path}")
        return False

    if not ecapa_path.endswith(".ecapa_averaged_embedding"):
        print("❌ Error: File must end with .ecapa_averaged_embedding")
        return False

    # Create new .pt path in the same folder
    directory = os.path.dirname(ecapa_path)
    filename = os.path.basename(ecapa_path)
    sid = filename.replace(".ecapa_averaged_embedding", "")
    
    pt_path = os.path.join(directory, f"{sid}.pt")

    try:
        # Load the embedding
        embedding = torch.load(ecapa_path, map_location='cpu')
        
        # Make sure it's in correct shape and normalized (recommended)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)   # Shape: (1, embedding_dim)
        
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        # Save as .pt
        torch.save(embedding, pt_path)

        print("✅ Conversion Successful!")
        print(f"   From : {filename}")
        print(f"   To   : {os.path.basename(pt_path)}")
        print(f"   Path : {pt_path}")
        print(f"   Shape: {embedding.shape}")
        print(f"   Norm : {embedding.norm():.4f}")

        return True

    except Exception as e:
        print(f"❌ Conversion Failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use path from command line
        file_path = sys.argv[1]
    else:
        # Ask user to input path
        print("Enter the full path to your .ecapa_averaged_embedding file:")
        file_path = input("> ").strip().strip('"\'')   # remove quotes if user pastes with them

    convert_ecapa_to_pt(file_path)