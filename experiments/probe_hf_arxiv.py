import requests
import re

# Target Repo
REPO_URL = "https://huggingface.co/datasets/nick007x/arxiv-papers/tree/main"
API_URL = "https://huggingface.co/api/datasets/nick007x/arxiv-papers"

def search_files():
    print(f"Probing {API_URL}...")
    try:
        r = requests.get(API_URL)
        r.raise_for_status()
        data = r.json()
        
        # Look for the actual file in 'siblings'
        siblings = data.get("siblings", [])
        print(f"Found {len(siblings)} files.")
        
        for s in siblings:
            fname = s.get("rfilename")
            print(f" - {fname}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_files()
