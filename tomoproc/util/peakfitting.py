#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usefule functions for peak fitting
"""

import numpy          as     np
import scipy          as     sp

from   typing         import Tuple
from   scipy.special  import wofz
from   scipy.optimize import curve_fit
from   lmfit.models   import VoigtModel


def gauss1d(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    """
    1D Gaussian function used for curve fitting.

    Parameters
    ----------
    x  :  np.ndarray
        1D array for curve fitting
    amp: float
        amplitude
    mu: float
        peak center
    sigma: float
        variance

    Returns
    -------
    np.ndarray
        1d Gaussian distribution evaluted at x with p

    NOTE
    ----
    The half-width at half-maximum (HWHM) for a gaussian distribution is
        alpha = sigma * np.sqrt(2*np.log(2))
    """
    return amp/sigma/np.sqrt(2*np.pi)*np.exp(-(x-mu)**2/2/(sigma**2))


def lorentz1d(x: np.ndarray, amp: float, mu: float, gamma:float) -> np.ndarray:
    """
    Description
    -----------
    1D Lorentz distrubtion function

    Parameters
    ----------
    x: np.ndarray
        input data
    amp: float
        height of peak
    mu: float
        center of peak
    gamma: float
        half-width at half-maximum (HWHM) of peak

    Returns
    -------
    np.ndarray
        1D Lorentz distrubtion with respect to the input array
    """
    return amp*(gamma**2)/((x-mu)**2 + gamma**2)


def voigt1d_convolve(
    x: np.ndarray, 
    amp:float, 
    mu:float, 
    sigma:float, 
    gamma:float,
    ) -> np.ndarray:
    """
    Description
    -----------
    1D Voigt distribution using numpy convolve (expensive and slow)

    Parameters
    ----------
    x: np.ndarray
        input array
    amp: float
        amplitude
    mu: float
        peak center
    sigma: float
        gaussian variance
    gamma: float
        lorentz HWHM

    Returns
    -------
    np.ndarray
        Voigt distribution with respect to x
    """
    return np.convolve(
        gauss1d(x, 1, mu, sigma),
        lorentz1d(mu-x, amp*gamma, mu, gamma),
        'same',
    )


def voigt1d(
    x:np.ndarray, 
    amp:float, 
    pos:float, 
    fwhm:float, 
    shape:float,
    ) -> np.ndarray:
    """
    Description
    -----------
    Fast Voigt distribution

    Parameters
    ----------
    x: np.ndarray
        input array
    amp: float
        amplitude
    pos: float
        peak center
    fwhm: float
        full width at half max
    shape: float
        peak shape, negative leads to under cut at both edges

    Returns
    -------
    np.ndarray
        Voigt distribution with respect to input array

    Note
    ----
    As pointed by 
        https://stackoverflow.com/questions/53156135/wrong-voigt-output-convolution-with-asymmetric-x-input, 
    convolution is expensive in terms of computation time which can get annoying when used as fit model.
    Wofz is a good approximation for the convolution
    """
    tmp = 1/wofz(np.zeros((len(x))) +1j*np.sqrt(np.log(2.0))*shape).real
    return tmp*amp*wofz(2*np.sqrt(np.log(2.0))*(x-pos)/fwhm+1j*np.sqrt(np.log(2.0))*shape).real


def fit_peak_1d(
    xdata:np.ndarray, 
    ydata:np.ndarray, 
    engine:str='lmfit',
    ) -> np.ndarray:
    """
    Description
    -----------
    Perform 1D peak fitting using Voigt function

    Parameters
    ----------
    xdata: np.ndarray
        independent var array
    ydata: np.ndarray
        dependent var array
    engien: str
        engine name, [lmfit, tomoproc]
    
    Returns
    -------
    dict
        dictionary of peak parameters

    NOTE
    ----
    Return dictionary have different entries.
    """
    if engine.lower() in ['lmfit', 'external']:
        mod = VoigtModel()
        pars = mod.guess(ydata, x=xdata)
        out = mod.fit(ydata, pars, x=xdata)
        return out.best_values
    else:
        popt, pcov = curve_fit(voigt1d, xdata, ydata, 
                               maxfev=int(1e6),
                               p0=[ydata.max(), xdata.mean(), 1, 1],
                               bounds=([0,              xdata.min(), 0,                       0],
                                       [ydata.max()*10, xdata.max(), xdata.max()-xdata.min(), np.inf]),
                            )
        return {
            'amplitude': popt[0],
            'center': popt[1], 
            'fwhm': popt[2],
            'shape': popt[3],
            }


def sigmoid(
    x: np.ndarray, 
    xc: float=0, 
    a: float=1,
    ) -> np.ndarray:
    """
    Description
    -----------
    1D sigmoid (logistic) function

    Parameters
    ----------
    x: np.ndarray
        independent var array
    xc: float
        distribution center
    a: float
        distribution shape
    
    Returns
    -------
    np.ndarray
        1D sigmoid distribution with given parameters
    """
    return 1.0/(1.0 + np.exp(-a*(x - xc)))


def fit_sigmoid(
    xdata: np.ndarray, 
    ydata: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Fit 1D sigmoid distribution

    Parameters
    ----------
    xdata: np.ndarray
        independent var array
    ydata: np.ndarray
        dependent var array
    
    Returns
    -------
    Tuple[opt, cov]
        Return fitting results from curve_fit

    NOTE
    ----
    This is mostly used for fine slit corner detection
    """
    # normalize ydata to standard range (0,1)
    ydata -= ydata.min()
    ydata /= ydata.max()
    r = xdata.mean()
    
    return curve_fit(
        sigmoid, xdata, ydata,
        p0=[r, 1],
        bounds=([r-r*0.2, -np.inf],
                [r+r*0.2,  np.inf],
               ),
    )


if __name__ == "__main__":
    # example usage
    amp, pos, fwhm, shape = np.random.random(4)*np.pi
    N = 101
    xdata = np.linspace(-np.pi, np.pi, N)
    ydata = voigt1d(xdata, amp, pos, fwhm, shape) + (np.random.random(N)-0.5)*amp/10
    print(f"intput:\nA={amp}, mu={pos}, gamma={fwhm/2}, shape={shape}")

    paras = fit_peak_1d(xdata, ydata, engine='lmfit')
    print(f"fit with lmfit:\n{paras}")

    paras = fit_peak_1d(xdata, ydata, engine='tomoproc')
    print(f"fit with tomoproc:\n{paras}")
