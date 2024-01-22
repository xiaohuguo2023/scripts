import torch

def softmax_einsum(tensor, dim):
    # Exponentiate the elements of the tensor
    exp_tensor = torch.exp(tensor)

    # Sum the exponentiated elements along the specified dimension using einsum
    # '...i->...' sums across the specified dimension 'i'
    sum_exp = torch.einsum('...i->...', exp_tensor)

    # Normalize by dividing the exponentiated tensor by the sum
    # We use None (newaxis) to keep the dimensions consistent for broadcasting
    softmax_tensor = exp_tensor / sum_exp.unsqueeze(dim)

    return softmax_tensor

# Example usage
tensor = torch.randn(2, 3)  # Example 2D tensor
softmax_tensor = softmax_einsum(tensor, dim=1)
print("Softmax Result:\n", softmax_tensor)

