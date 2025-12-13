import cmfo_compute as cmfo

x = cmfo.tensor([1, 2, 3, 4, 5, 6, 7])
y = x.evolve(10)

print("Tensor T7 inicial:", x)
print("Tensor T7 evolucionado:", y)
