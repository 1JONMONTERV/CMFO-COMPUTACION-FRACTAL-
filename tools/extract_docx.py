import os
import zipfile
import xml.etree.ElementTree as ET
import sys
import glob

def docx_to_text(path):
    """
    Extract text from a .docx file using standard libraries.
    """
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read('word/document.xml')
        
        tree = ET.fromstring(xml_content)
        
        # XML namespace for Word
        wp_namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        text_parts = []
        
        # Find all text nodes (<w:t>)
        for node in tree.iter(f'{wp_namespace}t'):
            if node.text:
                text_parts.append(node.text)
            
        return '\n'.join(text_parts)
    except Exception as e:
        return f"[ERROR reading {os.path.basename(path)}: {str(e)}]"

def process_directory(directory):
    """
    Scan directory for .docx and convert to .md
    """
    print(f"Scanning {directory}...")
    files = glob.glob(os.path.join(directory, "**/*.docx"), recursive=True)
    
    if not files:
        print("No .docx files found.")
        return

    full_report = "# CMFO Claims Audit\n\n"
    
    for f in files:
        print(f"Processing: {f}")
        text = docx_to_text(f)
        filename = os.path.basename(f)
        
        full_report += f"## Document: {filename}\n"
        full_report += "```text\n"
        full_report += text[:5000] # Limit to first 5000 chars to avoid huge output
        if len(text) > 5000:
            full_report += "\n...[TRUNCATED]...\n"
        full_report += "```\n\n"
        
    output_path = "docs/experimental/claims_audit.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    
    print(f"Audit saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_docx.py <directory>")
    else:
        process_directory(sys.argv[1])
