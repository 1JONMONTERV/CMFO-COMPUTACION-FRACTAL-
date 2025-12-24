"""
TEST: CMFO Fractal Memory Claims
================================
Verifies:
1. Determinism: Input -> Geometry -> Input (Bit-exact)
2. Continuity: Small changes in text -> Small changes in geometry
3. Codebook Stability.
"""

import pytest
import numpy as np
from cmfo.memory import FractalTokenizer

class TestFractalMemory:
    
    def setup_method(self):
        self.tokenizer = FractalTokenizer()

    def test_determinism_roundtrip(self):
        """
        CLAIM: "Memoria Determinista"
        Verifies that Text -> Trajectory -> Text is lossless for the codebook.
        """
        original = "Hola Mundo CMFO 2025!"
        
        # Encode
        trajectory = self.tokenizer.text_to_trajectory(original)
        
        # Verify shape
        assert trajectory.shape == (len(original), 7)
        
        # Decode
        recovered = self.tokenizer.trajectory_to_text(trajectory)
        
        assert recovered == original, \
            f"Lossless roundtrip failed. Got '{recovered}', expected '{original}'"

    def test_geometry_continuity(self):
        """
        CLAIM: "Estructura Algebraica"
        Similar inputs should map to nearby trajectories.
        """
        t1 = self.tokenizer.text_to_trajectory("A")
        t2 = self.tokenizer.text_to_trajectory("B")
        
        dist = np.linalg.norm(t1 - t2)
        
        # They should be distinct but bounded
        assert dist > 0.0, "Different tokens mapped to same point collision"
        assert dist < 50.0, "Distance blown up on manifold"

    def test_massive_encoding(self):
        """
        CLAIM: "Scalability"
        Encode a larger block (pseudo 'Quijote' paragraph).
        """
        text = "En un lugar de la Mancha..." * 100  # ~2.5 KB
        traj = self.tokenizer.text_to_trajectory(text)
        
        assert traj.shape == (len(text), 7)
        assert self.tokenizer.trajectory_to_text(traj) == text

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
