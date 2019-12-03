import torch
import numpy as np
####################################################
def KBest(inp, k, dim=-1, shrink=False, ternarize=False, symmetric=True, convexified=False):
    """
    Keeps elements with k largest magnitudes at each appropriate tensor dimension and zeros the rest.
    This non-linearity is developed and well-motivated in (https://arxiv.org/abs/1901.08437).
    
    :param inp (tensor)
    :param k: (int)   How many elements to keep?
    :param dim: (int) Which dimension to apply?
    :param shrink: (bool)   In soft thresholding, the kept values will be shrunk.
    :param ternarize: (bool) Should the values be quantized?
    :param symmetric: (bool) If not symmetric, only positive values will be preserved.
    :param convexified: (bool) Simply takes the absolute value of the output to make it convex.
    :return: out (tensor)
    """
    out = torch.zeros_like(inp)
    if symmetric:
        vals, sorted_idx = torch.topk(torch.abs(inp), k, dim=dim)
        out = out.scatter(dim,sorted_idx,vals)
        out *= torch.sign(inp)
    else:
        vals, sorted_idx = torch.topk(inp, k, dim=dim)
        out = out.scatter(dim, sorted_idx, vals)
    
    if shrink:
        out -= vals.narrow(dim, -1, 1) * torch.sign(out)
    
    if convexified:
        out = torch.abs(out)
    if ternarize:
        out.data = torch.sign(out.data)
    
    return out


########################################

### Utilities to generate fake code values similar to original codes.
### They follow the complementary truncated Gaussian distribution.
def estimate_lmda(code):
    """
    code is (b, c, m).
    Calculates an equivalent lmda per each code channel.
    Assumes all channels are k-sparse, i.e., k nnz values out of m.
    """
    (b, c, m) = code.shape

    code = code.cpu().transpose(0,1).reshape(c, -1)
    nz = code.cpu().nonzero()
    code = code[nz[:,0], nz[:,1]].cpu().view(b, c, -1)
    lmda = torch.abs(code).min(dim=2)[0].reshape(-1, b).mean(dim=1)
    return lmda
###
def estimate_std(lmda, k, m):
    std = lmda / (np.sqrt(2) * torch.erfinv(torch.tensor([1 - k/m])))
    return std
###
def iCDF_G(x, std=1):

    return  std * torch.erfinv(2 * x - 1 ) * np.sqrt(2.)
###
def iCDF_CTG(x, lmda, std=1):
    coeff = 1 / (torch.erfc(lmda / (std * torch.sqrt(torch.tensor([2.])))))
    val = 0.5 * torch.erfc(lmda / (std * torch.sqrt(torch.tensor([2.]))))
    y =  torch.zeros_like(x)
    y[x < val] =  (coeff) *  iCDF_G (x[x < val], std=std)
    y[x > 1 - val] =  (coeff) * iCDF_G (x[x > 1 - val], std=std)
        
    return y    
###
def generate_CTG(shape, lmda, std=1):
    "Generates complementary truncated Gaussian samples"
    val =  0.5 * torch.erfc(lmda / (std * torch.sqrt(torch.tensor([2.]))))
    x = torch.rand(shape) * val
    x[torch.rand(shape) < 0.5] += 1 - val
    y = iCDF_CTG(x, lmda, std)
    return y
###
def ambiguate(code, k_prime):
    """
    Highly non-efficient way to add ambiguation noise to zero values.
    Again assuming k-sparsity for all codes.
    """
    (b, c, m) = code.shape
    zs = (code == 0).nonzero()
    k = m - int(zs.shape[0] / (b * c))
    lmda = estimate_lmda(code)
    std = estimate_std(lmda, k, m)
    zs = zs.view(b, c, -1)[:,:,2::3]
    for j in range(c):
        for i in range(b):
            ind_p = zs[i, j, torch.randperm(m - k)[0:k_prime - k]]
            code[i, j, ind_p] =  generate_CTG((1, 1, k_prime - k), lmda[j], std[j])
    return code    

### 
def random_guess(code, k):
    """
    Highly non-efficient way to randomly pick k out of k_prime
    Assuming that the ambiguated code is k_prime sparse
    """
    (b, c, m) = code.shape
    zs = code.nonzero()
    k_prime = int(zs.shape[0] / (b * c))
    zs = zs.view(b, c, -1)[:,:,2::3]
    for j in range(c):
        for i in range(b):
            ind_p = zs[i, j, torch.randperm(k_prime)[0: k]]
            code[i, j, ind_p] = 0
    return code    
    
    
########################################################################################

### Functions to calculate the rate

def calculate_KBytes(m, k, L):
    H = -(k/m) * np.log2((k/m)) - (1 - (k/m)) * np.log2(1 - (k/m))
    return H * m * L  /(8 * 1024)
    
def calculate_psnr(m, k, L, im_size):
    H = -(k/m) * np.log2((k/m)) - (1 - (k/m)) * np.log2(1 - (k/m))
    return H * m * L / np.prod(im_size)


