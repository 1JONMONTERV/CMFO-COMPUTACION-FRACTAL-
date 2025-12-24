import requests

def search_hub():
    url = "https://huggingface.co/api/datasets"
    params = {
        "search": "arxiv",
        "sort": "downloads",
        "direction": "-1",
        "limit": 5
    }
    try:
        r = requests.get(url, params=params)
        data = r.json()
        print("Top ArXiv Datasets on HF:")
        for d in data:
            print(f"- {d['id']} (Downloads: {d.get('downloads', '?')})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_hub()
