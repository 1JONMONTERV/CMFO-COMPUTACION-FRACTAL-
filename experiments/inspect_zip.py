import zipfile
from pathlib import Path

TARGET = Path("D:/CMFO_DATA/science/raw/math/math-00-part-1.zip")

def inspect():
    if not TARGET.exists():
        print("Not found.")
        return
        
    try:
        with zipfile.ZipFile(TARGET, 'r') as zf:
            print(f"Total files: {len(zf.namelist())}")
            print("First 10 files:")
            for n in zf.namelist()[:10]:
                print(f" - {n}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
