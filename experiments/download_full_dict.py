import requests
import sys
import os
from pathlib import Path

URL = "https://kaikki.org/dictionary/Spanish/kaikki.org-dictionary-Spanish.jsonl"
DEST_DIR = Path("D:/CMFO_DATA/ontology/raw_source")
DEST_FILE = DEST_DIR / "kaikki_es.jsonl"

def download_dictionary():
    print(f"Starting download from {URL}")
    print(f"Destination: {DEST_FILE}")
    
    # Ensure dir exists
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with requests.get(URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(DEST_FILE, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192 * 4):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Simple progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rDownloading: {downloaded / 1024 / 1024:.1f} MB ({percent:.1f}%)")
                    else:
                        sys.stdout.write(f"\rDownloading: {downloaded / 1024 / 1024:.1f} MB")
                    sys.stdout.flush()
            
            print(f"\nDownload complete. Saved to {DEST_FILE}")
            return True
            
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

if __name__ == "__main__":
    success = download_dictionary()
    if not success:
        sys.exit(1)
