import requests
import re

INDEX_URL = "https://kaikki.org/dictionary/Spanish/index.html"

def find_jsonl_link():
    print(f"Fetching index from {INDEX_URL}...")
    try:
        r = requests.get(INDEX_URL, timeout=10)
        r.raise_for_status()
        html = r.text
        
        # Look for .jsonl link (href="...jsonl")
        links = re.findall(r'href="([^"]+\.jsonl)"', html)
        
        if not links:
            # Try .jsonl.gz
            links = re.findall(r'href="([^"]+\.jsonl\.gz)"', html)
            
        if links:
            # Construct full URL if relative
            link = links[0]
            if not link.startswith("http"):
                # Handle relative paths properly
                from urllib.parse import urljoin
                full_url = urljoin(INDEX_URL, link)
            else:
                full_url = link
                
            print(f"Found JSONL link: {full_url}")
            return full_url
        else:
            print("No JSONL links found in index page.")
            return None
            
    except Exception as e:
        print(f"Error fetching index: {e}")
        return None

if __name__ == "__main__":
    url = find_jsonl_link()
    if url:
        # Try a quick HEAD request
        try:
            h = requests.head(url, timeout=5)
            print(f"Link status: {h.status_code}")
        except:
            print("Could not verify link head.")
