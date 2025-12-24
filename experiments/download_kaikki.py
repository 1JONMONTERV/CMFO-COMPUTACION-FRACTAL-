import requests
import sys

URL = "https://kaikki.org/dictionary/Spanish/words.jsonl"
OUTPUT = "kaikki_sample.jsonl"

def download_sample():
    print(f"Attempting to download sample from {URL}...")
    try:
        with requests.get(URL, stream=True, timeout=10) as r:
            r.raise_for_status()
            print("Connection established. Reading first 5MB...")
            
            with open(OUTPUT, 'wb') as f:
                count = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    count += len(chunk)
                    if count > 5 * 1024 * 1024: # 5MB limit
                        break
            
            print(f"Successfully downloaded {count} bytes to {OUTPUT}")
            return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = download_sample()
    if not success:
        sys.exit(1)
