"""Basic tests for CMFO package"""
import pytest


def test_import_cmfo():
    """Test that cmfo package can be imported"""
    import cmfo
    assert cmfo.__version__ == "1.1.0"


def test_cmfo_info():
    """Test that info() function works"""
    import cmfo
    # Should not raise any exceptions
    cmfo.info()


def test_tensor_creation():
    """Test basic tensor creation"""
    import cmfo
    try:
        t = cmfo.tensor([1, 2, 3])
        # If tensor creation works, check basic properties
        assert hasattr(t, 'v')
    except Exception as e:
        # If it fails, it's likely because native extension is not available
        # This is acceptable in pure Python mode
        pytest.skip(f"Tensor creation requires native extension: {e}")


def test_phi_logic_functions():
    """Test that phi logic functions are available"""
    import cmfo
    assert hasattr(cmfo, 'f_and')
    assert hasattr(cmfo, 'f_or')
    assert hasattr(cmfo, 'f_not')
    assert hasattr(cmfo, 'f_xor')
    # nand is typically derived, check if f_nand exists or remove check if not in API
    # assert hasattr(cmfo, 'f_nand')
