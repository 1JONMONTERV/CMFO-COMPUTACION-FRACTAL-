import os
import sys

def find_vcvars():
    search_roots = [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio"
    ]
    
    print("Searching for vcvars64.bat...")
    
    for root in search_roots:
        if not os.path.exists(root):
            continue
            
        for dirpath, dirnames, filenames in os.walk(root):
            if "vcvars64.bat" in filenames:
                full_path = os.path.join(dirpath, "vcvars64.bat")
                print(f"FOUND: {full_path}")
                with open("compiler_path.txt", "w") as f:
                    f.write(full_path)
                return
    
    print("NOT FOUND")

if __name__ == "__main__":
    find_vcvars()
