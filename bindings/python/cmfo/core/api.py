PHI = (1 + 5 ** 0.5) / 2


def tensor7(a, b):
    # Convert to lists if needed
    a = list(a) if not isinstance(a, list) else a
    b = list(b) if not isinstance(b, list) else b
    # The formula provided in the audit: (a * b + PHI) / (1 + PHI)
    return [(x * y + PHI) / (1 + PHI) for x, y in zip(a, b)]
