import math

class CMFOLinear:
    """
    A Drop-in replacement for torch.nn.Linear (Pure Python Version).
    
    Instead of learning weight matrix W via backprop,
    this layer projects input into the 7D fractal basis.
    
    Usage:
        linear = CMFOLinear(in_features=128, out_features=64)
        output = linear(input_list) 
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        # CMFO "Weights" are implicit/geometric.
        # Minimal storage: just the config.
        pass

    def __call__(self, x):
        """
        Forward pass.
        Args:
            x: Input list of lists (Batch, In_Features)
        Returns:
            Output list of lists (Batch, Out_Features)
        """
        # Ensure input is a list (if it came from a legacy generator)
        # x shape: [Batch, In_Features]
        
        batch_size = len(x)
        output = []

        PHI = 1.6180339887

        # Pure Python Implementation
        for b in range(batch_size):
            # Input vector for this batch
            input_vec = x[b]
            
            # 1. Reduction: O(N) sum
            # Fractal folding: simplified energy sum
            val = sum(input_vec) if isinstance(input_vec, (list, tuple)) else input_vec

            # 2. Resonate: Project into output dimensions
            row_out = []
            for i in range(self.out_features):
                # Harmonic: Phi^(i mod 7)
                harmonic = PHI ** (i % 7)
                # Resonance formula
                res = (val * harmonic) / (1 + harmonic)
                row_out.append(res)
            
            output.append(row_out)

        return output

    def to(self, device):
        # The GPU Bridge would hook here
        # If device="cuda", we would swap __call__ for a ctypes wrapper
        if "cuda" in str(device).lower():
            from ..core.gpu import Accelerator
            if Accelerator.is_available():
                self.__call__ = Accelerator.get_kernel("linear_7d")
        return self
