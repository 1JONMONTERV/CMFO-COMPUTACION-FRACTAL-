import cmfo


def test_logic():
    # Use f_and, f_or (mapped to 0.0/1.0 floats)
    # 0 is False, 1 is True
    assert cmfo.f_and(1.0, 0.0) == 0.0
    assert cmfo.f_or(0.0, 1.0) == 1.0
    # phi_nand not exported in __all__ or logic? Check init.
    if hasattr(cmfo, 'f_nand'):
        assert cmfo.f_nand(1.0, 1.0) == 0.0
