"""
CMFO D13: Scientific Paper Fetcher
==================================
Downloads raw source (LaTeX/PDF) from arXiv mirrors.
Target: nick007x/arxiv-papers (Hugging Face)
Current Focus: OP_C (Math First)
"""

import requests
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Config
REPO_BASE = "https://huggingface.co/datasets/nick007x/arxiv-papers/resolve/main"
TARGET_FILE = "math-00-part-1.zip"
DEST_DIR = Path("D:/CMFO_DATA/science/raw/math")

def download_sample():
    print(f"CMFO Science Fetcher (Option C: Math)")
    print(f"=====================================")
    
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    url = f"{REPO_BASE}/{TARGET_FILE}"
    dest_path = DEST_DIR / TARGET_FILE
    
    print(f"Target: {url}")
    print(f"Dest:   {dest_path}")
    
    if dest_path.exists():
        print("File already exists. Skipping download.")
        return dest_path

    try:
        # Stream download
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=TARGET_FILE,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
                    
        print("\nDownload Complete.")
        return dest_path
        
    except Exception as e:
        print(f"\nDownload Error: {e}")
        if dest_path.exists():
            dest_path.unlink() # Cleanup partial
        return None

if __name__ == "__main__":
    download_sample()
