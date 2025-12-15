import math

PHI = (1 + 5 ** 0.5) / 2


def phi_pow(x):
    return PHI ** x


def phi_norm(v):
    v = list(v) if not isinstance(v, list) else v
    return math.sqrt(sum(x * x for x in v))
