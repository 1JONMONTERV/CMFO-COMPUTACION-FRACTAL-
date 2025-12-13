import cmfo


def test_logic():
    assert cmfo.phi_and(1, -1) == -1
    assert cmfo.phi_or(-1, 1) == 1
    assert cmfo.phi_nand(1, 1) == -1
