#!/usr/bin/env python
"""
Utilities for Gravitational Wave data handling and prior computation. A lot of the spin priors are copied from the public code gwpopulation_pipe https://git.ligo.org/RatesAndPopulations/gwpopulation_pipe/-/blob/master/gwpopulation_pipe/analytic_spin_prior.py?ref_type=heads
"""

__author__="Anarya Ray <anarya.ray@ligo.org>; Omkar Sridhar <omkar.sridhar@ligo.org>; Siddharth Mohite <siddharth.mohite@ligo.org>, Authors of gwpopulation_pipe <https://git.ligo.org/RatesAndPopulations/gwpopulation_pipe>"


import numpy as np
from scipy.stats import multivariate_normal,norm,halfnorm,lognorm
from astropy.cosmology import Planck15,z_at_value
from astropy import units as u
from scipy.interpolate import interp1d
import warnings
import tqdm
import numpy as xp
from scipy.special import spence as scipyspence

############################
#  Support Functions       #
############################
def log_prob_spin(sx,sy,sz,m):
    '''
    Function that computes the default spin priors
    used to generate spin-parameters of injections.
    
    Author = Siddharth Mohite
    
    Parameters
    ----------
    
    sx    :: float
             x component of spin of the binary component
    
    sy    :: float
             y component of spin of the binary component
    
    sz    :: float
             z component of spin of the binary component
    
    m     :: float
             mass of the binary component
    '''
    s_max = np.where(m<2.5,0.4,0.99)
    return np.log(1./(4*np.pi*s_max*(sx**2 + sy**2 + sz**2)))

def reweight_pinjection(tril_weights):
    '''
    A function that converts log weights of injections to
    weights. Since each injection will have non-zero weights
    in only one bin, the log weights (output of Vt_Utils_spins_with_q.log_reweight_pinjection_mixture)
    are set to zero at all other bins. This function acts as a 
    wrapper around exp such that only non-zero log weights
    are exponentiated.
    
    Author = Siddharth Mohite
    
    Parameters
    ----------
    
    tril_weights    :: numpy.ndarray
                       1d array containing the output of Vt_Utils.log_reweight_pinjection_mixture
                       for one injection.
                       
    Returns
    -------
    
    exponential of tril_weights : numpy.ndarray
                                  1d array containing the exponential of the log
                                  of injection weights.
    '''
    return np.where((tril_weights!=0),np.exp(tril_weights),0)

def construct_arg_mat_out_spins(mbins, qbins, x_bins, y_bins):
    '''
    Function that enables removal of unphysical bins in the m1-q space. 
    
    Parameters
    ----------
    mbins :: numpy.ndarray 
                 1d array containing mass bin edges
        
    qbins :: numpy.ndarray 
                 1d array containing mass-ratio bin edges    
        
    chi_bins :: numpy.ndarray
                 1d array containing effective spin bin edges.
    
    '''
    
    nbins_m = len(mbins)-1
    nbins_q = len(qbins) - 1
    nbins_x = len(x_bins) - 1
    nbins_y = len(y_bins) - 1
    m1_bin_centers = 0.5 * (mbins[1:] + mbins[:-1])
    qbin_centers = 0.5 * (qbins[1:] + qbins[:-1])
    x_bin_centers = 0.5 * (x_bins[1:] + x_bins[:-1])
    y_bin_centers = 0.5 * (y_bins[1:] + y_bins[:-1])
    
    M, Q, x, y = np.meshgrid(m1_bin_centers, qbin_centers, x_bin_centers, y_bin_centers, indexing='ij')
    
    arg_mat_spin = np.ones((nbins_m, nbins_q, nbins_x, nbins_y))
    for i in range(1, nbins_m):
        for j in range(nbins_q):
            for k in range(nbins_x):
                for l in range(nbins_y):
                    if qbin_centers[j] <= np.min(m1_bin_centers) / M[i, j, k,l] :
                        arg_mat_spin[i - 1, j, k,l] = 0
    arg_mat_flat = np.matrix.flatten(arg_mat_spin)    
    
    return arg_mat_spin


def compute_q_chieff_chip(m1, m2, z,s1x, s1y, s1z, s2x, s2y, s2z):
    chi_eff = (m1*s1z+m2*s2z)/(m1+m2)
    q = m2/m1
    chi1p = np.sqrt(s1x**2+s1y**2)
    chi2p = np.sqrt(s2x**2+s2y**2)
    chi_p = max(chi1p, (4 * q + 3) / (3 * q + 4) * q * chi2p)
    return q, chi_eff, chi_p

def compute_q_chieff_z(m1, m2, z,s1x, s1y, s1z, s2x, s2y, s2z):
    chi_eff = (m1*s1z+m2*s2z)/(m1+m2)
    q = m2/m1
    
    return q, chi_eff, z



# Relative tolerance for the series
TOL = 2.220446092504131e-16
PISQ_6 = 1.6449340668482264365
MAX_ITER = 500


def I1(chieff, chip, q):
    x1max = xp.minimum(
        xp.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
        (1 + q) * chieff + xp.sqrt(1 - chip**2),
    )
    x1min = xp.maximum(
        -xp.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
        (1 + q) * chieff - xp.sqrt(1 - chip**2),
    )
    cond1 = chip > 0
    cond2 = q >= (4 + 3 * q) * chip / (3 + 4 * q)
    cond3 = x1max >= x1min

    Fterm = F(x1max, (1 + q) * chieff, chip, (4 + 3 * q) / (3 + 4 * q) * chip, q) - F(
        x1min, (1 + q) * chieff, chip, (4 + 3 * q) / (3 + 4 * q) * chip, q
    )
    return xp.where(cond1 & cond2 & cond3, (1 + q) / (8 * q) * Fterm, 0)


def I2(chieff, chip, q):
    x2max = xp.minimum(q, (1 + q) * chieff + xp.sqrt(1 - chip**2))
    x2min = xp.maximum(-q, (1 + q) * chieff - xp.sqrt(1 - chip**2))

    cond1 = chip > 0
    cond2 = chip < 1
    cond3 = x2max >= x2min

    Fterm = F(x2max, (1 + q) * chieff, chip, 0, q) - F(
        x2min, (1 + q) * chieff, chip, 0, q
    )
    return xp.where(cond1 & cond2 & cond3, -(1 + q) / (8 * q) * Fterm, 0)


def I3(chieff, chip, q):
    x3max = xp.minimum(
        xp.sqrt(1 - chip**2),
        (1 + q) * chieff
        + xp.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
    )
    x3min = xp.maximum(
        -xp.sqrt(1 - chip**2),
        (1 + q) * chieff
        - xp.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
    )

    cond1 = chip > 0
    cond2 = q >= (4 + 3 * q) * chip / (3 + 4 * q)
    cond3 = x3max > x3min

    Fterm = F(x3max, (1 + q) * chieff, (4 + 3 * q) / (3 + 4 * q) * chip, chip, 1) - F(
        x3min, (1 + q) * chieff, (4 + 3 * q) / (3 + 4 * q) * chip, chip, 1
    )

    return xp.where(
        cond1 & cond2 & cond3, (1 + q) / (8 * q) * (4 + 3 * q) / (3 + 4 * q) * Fterm, 0
    )


def I4(chieff, chip, q):
    x4max = xp.minimum(
        1,
        (1 + q) * chieff
        + xp.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
    )
    x4min = xp.maximum(
        -1,
        (1 + q) * chieff
        - xp.sqrt(q**2 - ((4 + 3 * q) / (3 + 4 * q)) ** 2 * chip**2),
    )

    cond1 = chip > 0
    cond2 = q >= (4 + 3 * q) * chip / (3 + 4 * q)
    cond3 = x4max > x4min

    Fterm = F(x4max, (1 + q) * chieff, (4 + 3 * q) / (3 + 4 * q) * chip, 0, 1) - F(
        x4min, (1 + q) * chieff, (4 + 3 * q) / (3 + 4 * q) * chip, 0, 1
    )

    return xp.where(
        cond1 & cond2 & cond3, -(1 + q) / (8 * q) * (4 + 3 * q) / (3 + 4 * q) * Fterm, 0
    )


def F(x, a, b, c, d):
    return G(x / b, a / b, c / b) + xp.log(b**2 / d**2) * (
        xp.arctan((x - a) / b) + xp.arctan(a / b)
    )


def G(x, alpha, beta):
    pre = xp.where(x >= 0, 1, -1)
    alpha = xp.where(x >= 0, alpha, -alpha)
    x = xp.where(x >= 0, x, -x)
    g1 = g(x, alpha, beta)
    g2 = g(x, alpha, -beta)
    g3 = -g(0, alpha, beta)
    g4 = -g(0, alpha, -beta)

    out = pre * xp.imag(g1 + g2 + g3 + g4)
    return out


def g(x, alpha, beta):
    cond1 = xp.abs(beta) < 1
    cond2 = (beta == 1) & (alpha <= 0)
    x_ = xp.where((x == 0) & (beta == 0), 0.01, x)
    ret = xp.nan_to_num(
        xp.where(
            cond1,
            xp.log(x_ - beta * 1j)
            * xp.log((alpha - x_ + 1j) / (alpha + 1j - beta * 1j))
            + Li2((x_ - beta * 1j) / (alpha + 1j - beta * 1j)),
            xp.where(
                cond2,
                0.5 * (xp.log(x_ - alpha - 1j)) ** 2 + Li2(-alpha / (x_ - alpha - 1j)),
                xp.log(alpha + 1j - beta * 1j) * xp.log(alpha - x_ + 1j)
                - Li2((alpha - x_ + 1j) / (alpha + 1j - beta * 1j)),
            ),
        )
    )
    return xp.where((x == 0) & (beta == 0), 0, ret)


def jaxspence(z):
    """
    From scipy.special.spence's implementation:
    Compute Spence's function for complex arguments. The strategy is:
    - If z is close to 0, use a series centered at 0.
    - If z is far away from 1, use the reflection formula

    spence(z) = -spence(z/(z - 1)) - pi**2/6 - ln(z - 1)**2/2

    to move close to 1.
    - If z is close to 1, use a series centered at 1.

    """
    return jax.lax.cond(
        xp.abs(z) < 0.5,
        lambda z: cspence_series0(z),
        lambda z: jax.lax.cond(
            xp.abs(1 - z) > 1,
            lambda z: -cspence_series1(z / (z - 1)) - PISQ_6 - 0.5 * xp.log(z - 1) ** 2,
            lambda z: cspence_series1(z),
            z,
        ),
        z,
    )


def cspence_series0(z):
    """
    A series centered at z = 0; see

    http://functions.wolfram.com/10.07.06.0005.02

    """
    z_ = xp.where(z == 0, 0.01, z)

    def condition(args):
        n, zfac, term1, sum1, term2, sum2 = args
        return (
            (xp.abs(term1) > TOL * xp.abs(sum1)) | (xp.abs(term2) > TOL * xp.abs(sum2))
        ) & (n < MAX_ITER)

    def body(args):
        n, zfac, term1, sum1, term2, sum2 = args
        zfac *= z_
        term1 = zfac / n**2
        sum1 += term1
        term2 = zfac / n
        sum2 += term2
        return (n + 1, zfac, term1, sum1, term2, sum2)

    def body_fori(i, args):
        return body(args)

    n, zfac, term1, sum1, term2, sum2 = jax.lax.fori_loop(
        1, MAX_ITER, body_fori, (1, 1, 0, 0, 0, 0)
    )

    # n, zfac, term1, sum1, term2, sum2 = jax.lax.while_loop(
    # condition, body, (1, 1, xp.inf, 0, xp.inf, 0)
    # )
    return xp.where(z == 0, PISQ_6, PISQ_6 - sum1 + xp.log(z_) * sum2)


def cspence_series1(z):
    """
    A series centered at z = 1 which enjoys faster convergence than
    the Taylor series. The number of terms used comes from
    bounding the absolute tolerance at the edge of the radius of
    convergence where the sum is O(1).

    """

    z_ = xp.where(z == 1, 1 - 1e-10, z)

    z_ = 1 - z_
    zz = z_**2

    def condition(args):
        n, zfac, res, term = args
        return (xp.abs(term) > TOL * xp.abs(res)) & (n < MAX_ITER)

    def body(args):
        n, zfac, res, _ = args
        zfac *= z_
        term = ((zfac / n**2) / (n + 1) ** 2) / (n + 2) ** 2
        res += term
        return (n + 1, zfac, res, term)

    def body_fori(i, args):
        return body(args)

    n, zfac, res, term = jax.lax.fori_loop(1, MAX_ITER, body_fori, (1, 1, 0, xp.inf))
    # n, zfac, res, term = jax.lax.while_loop(condition, body, (1, 1, 0, xp.inf))
    res *= 4 * zz
    res += 4 * z_ + 5.75 * zz + 3 * (1 - zz) * xp.log(1 - z_)
    res /= 1 + 4 * z_ + zz
    return xp.where(z == 1, 0.0, res)


def Li2(z):
    if "jax" in xp.__name__:
        spence = jax.vmap(jaxspence)
    else:
        from scipy.special import spence
    z = xp.atleast_1d(z)
    return spence(1 - z)



def prior_chieff_chip_isotropic(chieff, chip, q, amax=1):
    chieff = chieff / amax
    chip = chip / amax
    return (
        I1(chieff, chip, q)
        + I2(chieff, chip, q)
        + I3(chieff, chip, q)
        + I4(chieff, chip, q)
    ) / amax**2


def chi_effective_prior_from_isotropic_spins(chi_eff, q, amax=1):
    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors.

    Taken from https://github.com/tcallister/effective-spin-priors/blob/cd5813890de043b2dc59bfaaf9f5eb7d57882641/priors.py
    with some fairly significant refactoring so the branching only depends
    on whether the two components can have opposite signs.

    Parameters
    ==========
    chi_eff: array-like
        Chi_effective value or values at which we wish to compute prior
    q: array-like
        Mass ratio value (according to the convention q<1)
    amax: array-like
        Maximum allowed dimensionless component spin magnitude

    Returns
    =======
    array-like
        The prior values
    """

    # Ensure that `xs` is an array and take absolute value
    chi_eff = xp.abs(xp.array(chi_eff))

    max_primary = amax / (1 + q)
    max_secondary = q * amax / (1 + q)
    max_difference = amax * (1 - q) / (1 + q)

    # Set up various piecewise cases, these are applied consecutively so lower bounds are implicit
    opposite_signs_allowed = chi_eff < max_difference
    same_sign_required = chi_eff < amax

    with np.errstate(divide="ignore", invalid="ignore"):
        secondary_ratio = max_secondary / chi_eff
        primary_ratio = max_primary / chi_eff

        lower = (
            (4 - xp.log(q**2) - xp.log(xp.abs(1 / secondary_ratio**2 - 1)))
            + xp.nan_to_num(
                xp.log(xp.abs(1 - secondary_ratio) / (1 + secondary_ratio))
                + (Li2(-secondary_ratio + 0j) - Li2(secondary_ratio + 0j)).real
            )
            / secondary_ratio
        ) / (4 * max_primary)

        # these terms diverge on boundaries and so we manually regularize
        primary_term = xp.log(xp.abs(1 / primary_ratio - 1) + (primary_ratio == 1))
        secondary_term = xp.log(
            xp.abs(1 / secondary_ratio - 1) + (secondary_ratio == 1)
        )

        upper = (
            2 * (amax - chi_eff)
            + max_difference * xp.log(q)
            + (chi_eff - max_secondary) * secondary_term
            + (chi_eff - max_primary) * primary_term
            + chi_eff * xp.log(primary_ratio) * (primary_term - xp.log(q))
            + chi_eff * (Li2(1 - primary_ratio + 0j) - Li2(secondary_ratio + 0j)).real
        ) / (4 * max_primary * max_secondary)

    pdfs = xp.select([opposite_signs_allowed, same_sign_required], [lower, upper], 0.0)

    return pdfs.squeeze()
