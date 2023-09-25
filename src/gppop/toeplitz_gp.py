#!/usr/bin/env python
__author__="Anarya Ray"

import pymc as pm
from pymc.gp.cov import Constant
from pymc.gp.mean import Zero
from pymc.gp.util import JITTER_DEFAULT, stabilize,cholesky
from pymc.math import cartesian, kron_dot
import aesara.tensor as at
from aesara.tensor.slinalg import Cholesky
import scipy as sp
import numpy as np
from .lib import toeplitz_cholesky as tc


def toeplitz_cholesky_wrapper(r):
    l = tc.t_cholesky_lower(len(r),r)
    if np.isnan(l).any():
        raise sp.linalg.LinAlgError
    else:
        return l

class Cholesky_tp(Cholesky):
    
    __props__ = ("lower", "destructive", "on_error")

    def __init__(self, lower=True, on_error="raise"):
        super().__init__(lower=True, on_error="raise")
        
    
    def perform(self,node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        t = x[0,:]/np.sqrt(x[0,0])
        try:
            z[0] = (toeplitz_cholesky_wrapper(t)).astype(x.dtype)
        except sp.linalg.LinAlgError:
            if self.on_error == "raise":
                raise
            else:
                z[0] = (np.zeros(x.shape) * np.nan).astype(x.dtype)
        

cholesky_tp = Cholesky_tp()

class Latent(pm.gp.Latent):
    
    def __init__(self, *, mean_func=Zero(), cov_func=Constant(0.0)):
        super().__init__(mean_func=mean_func, cov_func=cov_func)
        
    
    def _build_prior(self, name, X, reparameterize=True, jitter=JITTER_DEFAULT, **kwargs):
        mu = self.mean_func(X)
        cov = stabilize(self.cov_func(X), jitter)
        if reparameterize:
            size = np.shape(X)[0]
            v = pm.Normal(name + "_rotated_", mu=0.0, sigma=1.0, size=size, **kwargs)
            f = pm.Deterministic(name, mu + cholesky_tp(cov).dot(v), dims=kwargs.get("dims", None))
        else:
            f = pm.MvNormal(name, mu=mu, cov=cov, **kwargs)
        return f

class LatentKron(pm.gp.LatentKron):
    def __init__(self, *, mean_func=Zero(), cov_funcs=(Constant(0.0))):
        super().__init__(mean_func=mean_func, cov_funcs=cov_funcs)
        
    def _build_prior(self, name, Xs, jitter, **kwargs):
        self.N = int(np.prod([len(X) for X in Xs]))
        mu = self.mean_func(cartesian(*Xs))
        chols = [cholesky_tp(stabilize(cov(X), jitter)) for cov, X in zip(self.cov_funcs, Xs)]
        v = pm.Normal(name + "_rotated_", mu=0.0, sigma=1.0, size=self.N, **kwargs)
        f = pm.Deterministic(name, mu + at.flatten(kron_dot(chols, v)))
        return f