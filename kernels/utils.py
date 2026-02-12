import triton
import triton.language as tl

@triton.jit
def tanh(x):
    e2x = tl.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)

@triton.jit
def gelu_tanh(x):
    k0 = 0.7978845608028654
    k1 = 0.044715
    x3 = x * x * x
    return 0.5 * x * (1.0 + tanh(k0 * (x + k1 * x3)))

@triton.jit
def gelu_bwd(x):
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff

@triton.jit
def gelu_bwd_derivative(x):
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))

    term1 = 0.79788456
    x2 = x * x
    term2 = 6 * 0.79788456 * 0.044715 * x2
    term3 = (0.79788456 + 3 * 0.79788456 * 0.044715 * x2) * (0.79788456 + 3 * 0.79788456 * 0.044715 * x2)
    term3 = x * tanh_out * term3
    
    derivative = (1 - tanh_out * tanh_out) * (term1 + term2 - term3)
    return derivative