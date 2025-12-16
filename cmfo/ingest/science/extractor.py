"""
CMFO D13: Scientific Extractor
==============================
Parses raw LaTeX source to extract formal algebraic propositions.
Focus: Definitions, Relations, Transformations, Constraints.
"""

import re
import os
import tarfile
import gzip
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Generator

class ScientificExtractor:
    def __init__(self, raw_zip_path: Path):
        self.zip_path = raw_zip_path
        
    def stream_papers(self) -> Generator[Dict, None, None]:
        """
        Yields papers one by one from the massive zip using stream processing.
        Handles PDF files via pypdf.
        """
        if not self.zip_path.exists():
            return

        try:
            import io
            from pypdf import PdfReader
            
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                for name in zf.namelist():
                    # Check for PDF
                    if name.endswith('.pdf'):
                        with zf.open(name) as pdf_file:
                            try:
                                # Read bytes into memory buffer for pypdf
                                pdf_bytes = io.BytesIO(pdf_file.read())
                                reader = PdfReader(pdf_bytes)
                                
                                text = ""
                                # Extract text from all pages
                                for page in reader.pages:
                                    extracted = page.extract_text()
                                    if extracted:
                                        text += extracted + "\n"
                                
                                # Extract ID
                                paper_id = Path(name).stem
                                
                                yield {"id": paper_id, "text": text}
                                
                            except Exception as e:
                                # print(f"PDF Error {name}: {e}")
                                continue
        except Exception as e:
            print(f"Zip Access Error: {e}")

    def extract_layers(self, text: str) -> Dict[str, List[str]]:
        """
        Applies the Expanded Heuristic Extraction on PLAIN TEXT.
        Layers: Definitions, Relations, Theorems, Lemmas, Proofs.
        """
        layers = {
            "definitions": [],
            "relations": [],
            "theorems": [],
            "lemmas": [],
            "proofs": [],
            "transformations": [],
            "constraints": []
        }
        
        # 0. Structural Blocks (Theorems, Lemmas)
        # "Theorem 1.1. <content>"
        struct_patterns = {
            "theorems": r"(?:Theorem|Thm\.?)\s*\d+(?:\.\d+)*\.?\s*(.*?)(\.|\n\n)",
            "lemmas": r"(?:Lemma)\s*\d+(?:\.\d+)*\.?\s*(.*?)(\.|\n\n)",
            "corollaries": r"(?:Corollary)\s*\d+(?:\.\d+)*\.?\s*(.*?)(\.|\n\n)"
        }
        
        for key, pat in struct_patterns.items():
            matches = re.finditer(pat, text, re.IGNORECASE | re.DOTALL)
            for m in matches:
                clean = self._clean_text(m.group(1))
                if len(clean) > 20: # Filters out "Theorem 1." empty headers
                     # For D17 we might want to store the ID too, but let's keep it simple text first
                     # Or append ID: "1.1: Content"
                     if key == "corollaries": # Map corollaries to theorems list or own? Let's assume user wants main structs
                         layers["theorems"].append(f"Corollary: {clean}")
                     else:
                        layers[key].append(clean)

        # 1. Definitions
        # "We define X as..."
        # "Definition 1. Let X be..."
        def_patterns = [
            r"Definition \d+\.(.*?)(\.|\n)", # Explicit Definition block
            r"We define (.*?) as",
            r"Let (.*?) be",
            r"(.*?) is defined to be"
        ]
        
        for p in def_patterns:
            matches = re.finditer(p, text, re.IGNORECASE)
            for m in matches:
                clean = self._clean_text(m.group(0)) # Capture full phrase
                if len(clean) > 10 and len(clean) < 300: # Sanity limits
                    layers["definitions"].append(clean)
                    
        # 2. Relations (Equations are hard in PDF text, look for implication keywords)
        # "implies", "leads to", "equivalent to"
        rel_patterns = [
            r"(.*?) implies (.*?)(\.|\n)",
            r"(.*?) is equivalent to (.*?)(\.|\n)",
            r"equation (.*?) holds"
        ]
        for p in rel_patterns:
            matches = re.finditer(p, text, re.IGNORECASE)
            for m in matches:
                clean = self._clean_text(m.group(0))
                if len(clean) > 10:
                    layers["relations"].append(clean)

        # 3. Proofs (Hardest to limit end, but look for QED or "Proof." start)
        # Heuristic: "Proof. <content> <QED/Box>"
        # For now, extract just the first few lines of proof or try to find block
        proof_matches = re.finditer(r"Proof\.(.*?)(\u220e|Q\.E\.D\.|Box|\n\s*\n)", text, re.DOTALL | re.IGNORECASE)
        for m in proof_matches:
             # Limit proof size to avoid capturing whole paper if QED missing
             content = m.group(1).strip()
             if len(content) < 2000: 
                 layers["proofs"].append(content)
                
        # 4. Transformations
        if "transform" in text.lower() or " map " in text.lower():
             # Look for "maps X to Y"
             matches = re.finditer(r"maps (.*?) to (.*?)(\.|\n)", text, re.IGNORECASE)
             for m in matches:
                 layers["transformations"].append(f"{m.group(1).strip()} -> {m.group(2).strip()}")

        return layers

    def _clean_text(self, text: str) -> str:
        """Removes PDF artifacts (ligatures, excessive whitespace)"""
        text = re.sub(r"\s+", " ", text)
        # Fix common PDF definition artifacts like "Defi nition" if needed, 
        # but pypdf is usually okay-ish.
        return text.strip()

if __name__ == "__main__":
    # Test runner
    target = Path("D:/CMFO_DATA/science/raw/math/math-00-part-1.zip")
    if target.exists():
        extractor = ScientificExtractor(target)
        count = 0
        for paper in extractor.stream_papers():
            layers = extractor.extract_layers(paper["text"])
            if layers["definitions"]:
                print(f"--- Paper {paper['id']} ---")
                print(f"Defs: {len(layers['definitions'])}")
                print(f"Sample: {layers['definitions'][0][:100]}...")
                count += 1
                if count >= 5: break
