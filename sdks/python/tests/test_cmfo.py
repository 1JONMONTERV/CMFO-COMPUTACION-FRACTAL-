"""
CMFO Python SDK - Tests
"""
import pytest
from cmfo import CMFOIntegrated, get_version

def test_version():
    """Test version retrieval"""
    ver = get_version()
    assert len(ver) == 3
    assert ver[0] == 1  # Major version

def test_parse():
    """Test text parsing"""
    with CMFOIntegrated() as cmfo:
        vec = cmfo.parse("verdad")
        assert len(vec) == 7
        assert vec[1] == 1.0  # Truth axis

def test_solve():
    """Test equation solving"""
    with CMFOIntegrated() as cmfo:
        solution = cmfo.solve("2x + 3 = 7")
        assert "x = 2" in solution or "x = 2.0" in solution

def test_compose():
    """Test vector composition"""
    with CMFOIntegrated() as cmfo:
        v1 = [1, 0, 0, 0, 0, 0, 0]
        v2 = [0, 1, 0, 0, 0, 0, 0]
        result = cmfo.compose(v1, v2)
        assert len(result) == 7
        # Result should be normalized
        norm = sum(x**2 for x in result)
        assert abs(norm - 1.0) < 0.01

def test_distance():
    """Test fractal distance"""
    with CMFOIntegrated() as cmfo:
        v1 = [1, 0, 0, 0, 0, 0, 0]
        v2 = [0, 1, 0, 0, 0, 0, 0]
        d = cmfo.distance(v1, v2)
        assert d > 0
        assert d < 10  # Reasonable bound

def test_negate():
    """Test negation"""
    with CMFOIntegrated() as cmfo:
        v = [0, 1, 0, 0, 0, 0, 0]
        neg_v = cmfo.negate(v)
        assert len(neg_v) == 7
        # Truth axis should be inverted
        assert neg_v[1] < 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
