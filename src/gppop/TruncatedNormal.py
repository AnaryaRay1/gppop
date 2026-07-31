"""
Truncated Normal Distribution in Pyro/PyTorch
"""

import torch
import pyro.distributions as dist
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.distributions.normal import Normal
from torch.distributions.utils import _standard_normal, broadcast_all


def standard_normal_cdf(x):
    """Computes the CDF of the standard normal distribution."""
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

def standard_normal_icdf(p):
    """Approximates the inverse CDF (quantile function) of the standard normal distribution."""
    return torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * p - 1)

def normal_cdf(x, loc, scale):
    """Computes the CDF of a normal distribution with given loc and scale."""
    return standard_normal_cdf((x - loc) / scale)

def normal_icdf(p, loc, scale):
    """Computes the inverse CDF (quantile function) of a normal distribution."""
    return loc + scale * standard_normal_icdf(p)

def normal_pdf(x, loc, scale):
    """Computes the PDF of a normal distribution."""
    return (1 / (scale * torch.sqrt(torch.tensor(2.0) * torch.pi))) * torch.exp(-0.5 * ((x - loc) / scale) ** 2)

class TruncatedNormalTransform(Transform):
    def __init__(self, loc, scale, low, high):
        super().__init__()
        
        # Broadcast parameters to ensure they have the same shape
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        
        # Compute CDF values for truncation bounds
        self.cdf_low = normal_cdf(self.low, self.loc, self.scale)
        self.cdf_high = normal_cdf(self.high, self.loc, self.scale)
        
    def __call__(self, x):
        """Transforms a uniform sample to a truncated normal sample."""
        u = x * (self.cdf_high - self.cdf_low) + self.cdf_low
        return normal_icdf(u, self.loc, self.scale)
    
    def _inverse(self, y):
        """Transforms a truncated normal sample back to a uniform sample."""
        u = normal_cdf(y, self.loc, self.scale)
        return (u - self.cdf_low) / (self.cdf_high - self.cdf_low)
    
    def log_abs_det_jacobian(self, x, y):
        """Computes the log absolute determinant of the Jacobian."""
        return torch.log(self.cdf_high - self.cdf_low) - torch.log(normal_pdf(y, self.loc, self.scale))
    
    @property
    def domain(self):
        return constraints.unit_interval
    
    @property
    def codomain(self):
        return constraints.interval(self.low, self.high)
    
    def with_cache(self, cache_size=1):
        return self  # This transform does not need caching
