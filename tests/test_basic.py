import cmfo


def test_tensor_creation():
    t = cmfo.tensor([1, 2, 3])
    assert t.v[0] == 1


def test_tensor_evolve():
    t = cmfo.tensor([1, 2, 3])
    y = t.evolve(1)
    assert isinstance(y, cmfo.T7Tensor)
