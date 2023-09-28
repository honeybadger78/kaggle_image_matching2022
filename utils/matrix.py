def FlattenMatrix(M, num_digits=8):
    """Convenience function to write CSV files."""
    return " ".join([f"{v:.{num_digits}e}" for v in M.flatten()])
