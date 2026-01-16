"""
Gradient checking utility for verifying backpropagation implementations.
"""

def grad_check(f, x, analytic_grad, epsilon=1e-5):
    """
    Perform gradient checking on the function f at point x.
    """
    num_grad = (f(x + epsilon) - f(x- epsilon)) / (2 * epsilon)
    return abs(num_grad - analytic_grad) < 1e-6
