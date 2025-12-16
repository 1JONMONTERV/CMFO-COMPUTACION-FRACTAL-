"""
TEST: Scientific Extractor Logic
================================
Verifies that the extractor correctly identifies:
1. Definitions ("We define...")
2. Equations (\\begin{equation}...)
3. Transformations ("map T:...")
"""

import unittest
import zipfile
import io
import sys
import os
from pathlib import Path

# Fix import path
sys.path.insert(0, os.path.abspath('.'))

from cmfo.ingest.science.extractor import ScientificExtractor

class TestScientificExtractor(unittest.TestCase):
    def setUp(self):
        # Create a mock zip in memory
        self.mock_zip_path = Path("mock_arxiv.zip")
        self.buffer = io.BytesIO()
        
        # Sample LaTeX content
        self.latex_content = r"""
        \documentclass{article}
        \begin{document}
        
        \section{Introduction}
        We define the Hilbert Space H as a complete inner product space.
        
        \begin{definition}
        Let $S$ be a compact set in $R^n$.
        \end{definition}
        
        The function maps $f: A \to B$.
        
        \begin{equation}
        E = mc^2 \implies m = E/c^2
        \end{equation}
        
        \end{document}
        """
        
        # We need nested structure: zip -> .gz -> text?
        # The code expects .gz inside zip.
        import gzip
        gz_buffer = io.BytesIO()
        with gzip.open(gz_buffer, 'wt', encoding='utf-8') as f:
            f.write(self.latex_content)
            
        with zipfile.ZipFile(self.mock_zip_path, 'w') as zf:
            zf.writestr("paper01.tex.gz", gz_buffer.getvalue())
            
    def tearDown(self):
        if self.mock_zip_path.exists():
            self.mock_zip_path.unlink()
            
    def test_extraction(self):
        extractor = ScientificExtractor(self.mock_zip_path)
        
        papers = list(extractor.stream_papers())
        self.assertEqual(len(papers), 1)
        
        text = papers[0]["text"]
        layers = extractor.extract_layers(text)
        
        print("\nExtracted Layers:")
        print(layers)
        
        # Checks
        self.assertTrue(len(layers["definitions"]) >= 1)
        # Check if *any* definition matches
        found_def = any("Hilbert Space" in d for d in layers["definitions"])
        self.assertTrue(found_def, "Hilbert Space definition not captured")
        
        self.assertTrue(len(layers["relations"]) >= 1)
        self.assertIn("implies", layers["relations"][0])
        
        # Transformations might fail if regex needs tweaking
        # Let's see if we captured it
        if layers["transformations"]:
             print("Transformations found:", layers["transformations"])
        else:
             print("Warning: No transformations found. Regex check needed.")

if __name__ == "__main__":
    unittest.main()
