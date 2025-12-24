
try:
    with open('build_final.log', 'r', errors='ignore') as f:
        lines = f.readlines()
        print(f"Total lines: {len(lines)}")
        for i, line in enumerate(lines):
            if "error" in line.lower() or "fatal" in line.lower() or "fail" in line.lower():
                print(f"LINE {i}: {line.strip()}")
                # Print next 2 lines for context
                if i + 1 < len(lines): print(f"  +1: {lines[i+1].strip()}")
                if i + 2 < len(lines): print(f"  +2: {lines[i+2].strip()}")
except Exception as e:
    print(e)
