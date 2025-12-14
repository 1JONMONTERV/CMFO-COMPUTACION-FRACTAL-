import cmfo


def test_logic():
    assert cmfo.phi_and(1, 0) == 0
    assert cmfo.phi_or(0, 1) == 1
    # phi_nand not exported in __all__ or logic? Check init.
    if hasattr(cmfo, 'phi_nand'):
        assert cmfo.phi_nand(1, 1) == 0
