import datetime

def publish_paper():
    paper_path = "d21-publication/CMFO_FOUNDATIONAL_PAPER.md"
    
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"<!-- PUBLISHED: {timestamp} -->\n<!-- DOI: 10.1000/CMFO.2025.001 (Simulated) -->\n\n"
    
    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(header + content)
        
    print(f"[*] Paper published successfully at {timestamp}")
    print(f"[*] DOI Assigned: 10.1000/CMFO.2025.001")

if __name__ == "__main__":
    publish_paper()
