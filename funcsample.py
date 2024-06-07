import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import skewnorm
from time import time

from astropy.cosmology import WMAP9, Planck18
import astropy.units
# Useful constants
keV_to_erg = 1 * astropy.units.keV
keV_to_erg = keV_to_erg.to_value("erg")
Gpc_to_cm = 1 * astropy.units.Gpc
Gpc_to_cm = Gpc_to_cm.to_value("cm")


def normalisation_calc(ind1, ind2):
  """

  """
  xb = (ind1-ind2) / (ind1+2)

  def temp_func(x):
    return x ** (ind1 + 1) * np.exp(-(ind1 + 2) * x)

  IntEner = np.logspace(-8, np.log10(xb), 100000)
  IntFlu = temp_func(IntEner)
  IntNorm = trapezoid(IntFlu, x=IntEner)
  norm = 1 / (IntNorm - np.exp(ind2 - ind1) / (ind2 + 2) * xb ** (ind1 + 2))
  return norm


def band_norm(ener, norm, ind1, ind2):
  """
  Normalized Band function as described in Sarah Antier's thesis
  """
  xb = (ind1-ind2) / (ind1+2)
  if type(ener) is float or type(ener) is int:
    if ener <= xb:
      return norm * ener**ind1 * np.exp(-(ind1+2) * ener)
    else:
      return norm * ener**ind2 * np.exp(ind2-ind1) * xb**(ind1-ind2)
  elif type(ener) is np.ndarray:
    return np.where(ener <= xb, norm * ener**ind1 * np.exp(-(ind1+2) * ener), norm * ener**ind2 * np.exp(ind2-ind1) * xb**(ind1-ind2))
  else:
    raise TypeError("Please use a correct type for ener, only accepted are float or numpy ndarray")


def norm_band_spec_calc(band_low, band_high, red, dl, ep, liso, ener_range, verbose=False):
  """
  Calculates the spectrum of a band function based on indexes and energy/luminosity values
  """
  # Normalisation value
  ampl_norm = normalisation_calc(band_low, band_high)
  # Redshift corrected and normalized observed energy range
  x = (1 + red) * ener_range / ep

  # Normalized spectrum
  spec_norm = band_norm(x, ampl_norm, band_low, band_high)

  # Spectrum norm
  norm = (1 + red) ** 2 / (4 * np.pi * (dl * Gpc_to_cm) ** 2) * liso / (ep**2 * keV_to_erg)
  pflux_norm = (1 + red) / (4 * np.pi * (dl * Gpc_to_cm) ** 2) * liso / (ep * keV_to_erg)

  spectrum = norm * spec_norm
  int_norm_spec = trapezoid(spec_norm, x)
  peak_flux = pflux_norm * int_norm_spec

  # Spectrum equivalent to usual Band spectrum with :
  # amp = norm_val * ampl_norm * (ep_rest_temp / 100) ** (-band_low_obs_temp)
  # band((1 + z_obs_temp) * ener_range, amp, band_low_obs_temp, band_high_obs_temp, ep_rest_temp, pivot=100)

  if verbose:
    ratio_norm = trapezoid(x * spec_norm, x)
    print("==========================================================================================")
    print("Spectrum information :")
    print("Integral of x*B(x), supposed to be 1 from 0 to +inf : ")
    print(f"   With energy range : {np.min(ener_range)}-{np.max(ener_range)} keV  :  ", ratio_norm)
    xtest = (1 + red) * np.logspace(-4, 8, 100000) / ep
    spec_norm_test = band_norm(xtest, ampl_norm, band_low, band_high)
    print(f"   With energy range : {1e-4}-{1e8} keV  :  ", trapezoid(xtest * spec_norm_test, xtest))
    print("Peak photon flux method 1 direct : ")
    print(f"   Between : {np.min(ener_range)}-{np.min(ener_range)} keV  :  {peak_flux} ph/cm²/s")
    print(f"   Between : {1e-4}-{1e8} keV  :  {pflux_norm * trapezoid(spec_norm_test, xtest)} ph/cm²/s")
    print("Integrated normalized spectrum value : ", int_norm_spec)
    print(f"Part of total luminosity on energy range {np.min(ener_range)}-{np.max(ener_range)} keV : ", ratio_norm)

  return norm, spectrum, peak_flux


def calc_flux_sample(catalog, index, ergcut):
  """
  Calculates the fluence per unit time of a given source using an energy cut and its spectrum
  :param catalog: GBM catalog containing sources' information
  :param index: index of the source in the catalog
  :param ergcut: energy window over which the fluence is calculated
  :returns: the number of photons per cm² for a given energy range, averaged over the duration of the sim : ncount/cm²/s
  """
  ener_range = np.logspace(np.log10(ergcut[0]), np.log10(ergcut[1]), 100001)
  norm_val, spec, pflux = norm_band_spec_calc(catalog.band_low[index], catalog.band_high[index], catalog.red[index], catalog.dl[index], catalog.ep[index], catalog.liso[index], ener_range)
  return pflux


def acc_reject(func, func_args, xmin, xmax):
  """
  Proceeds to an acceptance rejection method
  """
  loop = True
  max_func = np.max(func(np.linspace(xmin, xmax, 1000), *func_args)) * 1.05
  while loop:
    variable = xmin + np.random.random() * (xmax - xmin)
    thresh_value = func(variable, *func_args)
    test_value = np.random.random() * max_func
    if test_value <= thresh_value:
      # loop = False
      return variable


def lpeak_function_long(lpeak):
  """
  Version
    Luminosity function (distribution) for short GRBs
    Function and associated parameters and cases are taken from Ghirlanda et al. 2016
  :param lpeak: float or array of float containing peak luminosities
  """
  al1_l, al2_l, lb_l = -0.65, -3, 10 ** 52.05
  if type(lpeak) is float or type(lpeak) is int:
    if lpeak < lb_l:
      return (lpeak / lb_l)**(al1_l)
    else:
      return (lpeak / lb_l)**(al2_l)
  elif type(lpeak) is np.ndarray:
    return np.where(lpeak < lb_l, (lpeak / lb_l)**(al1_l), (lpeak / lb_l)**(al2_l))
  else:
    raise TypeError("Please use a correct type for luminosity, only accepted are float or numpy ndarray")


def redshift_distribution_long(red):
  """
  Version
    redshift distribution for long GRBs
    Function and associated parameters and cases are taken from Sarah Antier's thesis
  :param red: float or array of float containing redshifts
  """
  red0, n1, n2, z1 = 0.42, 2.07, -0.7, 3.6
  if type(red) is float or type(red) is int:
    if red <= z1:
      return red0 * (1 + red)**n1
    else:
      return red0 * (1 + z1)**(n1 - n2) * (1 + red)**n2
  elif type(red) is np.ndarray:
    return np.where(red <= z1, red0 * (1 + red)**n1, red0 * (1 + z1)**(n1 - n2) * (1 + red)**n2)
  else:
    raise TypeError("Please use a correct type for red, only accepted are float or numpy ndarray")


def redshift_distribution_short(red):
  """
  Version
    redshift distribution for short GRBs
    Function and associated parameters and cases are taken from Ghirlanda et al. 2016
  :param red: float or array of float containing redshifts
  """
  p1, zp, p2 = 2.8, 2.3, 3.5
  if type(red) is float or type(red) is int or type(red) is np.ndarray:
    return (1 + p1 * red) / (1 + (red / zp)**p2)
  else:
    raise TypeError("Please use a correct type for red, only accepted are float or numpy ndarray")


def red_rate_long(red):
  """
  Version
    Function to obtain the number of long GRB and to pick them according to their distribution
    Parameters from Sarah Antier's thesis
  """
  vol_com = Planck18.differential_comoving_volume(red).to_value("Gpc3 / sr")  # Change from Mpc3 / sr to Gpc3 / sr
  return 4 * np.pi * redshift_distribution_long(red) / (1 + red) * vol_com


def red_rate_short(red, rate0):
  """
  Version
    Function to obtain the number of short GRB and to pick them according to their distribution
    Parameters from Ghirlanda et al. 2016
  """
  vol_com = Planck18.differential_comoving_volume(red).to_value("Gpc3 / sr")  # Change from Mpc3 / sr to Gpc3 / sr
  return rate0 * 4 * np.pi * redshift_distribution_short(red) / (1 + red) * vol_com


def sbplaw(x, A, xb, alpha1, alpha2, delta):
  return A * (x/xb)**(-alpha1) * (1/2*(1+(x/xb)**(1/delta)))**((alpha1-alpha2)*delta)


def epeak_distribution_long(epeak, version):
  """
  Version 1
    Peak energy distribution for short GRBs
    Function and associated parameters and cases are taken from Ghirlanda et al. 2016
  Version 2
    Distribution corresponding to the Epeak obs by Fermi GBM
    !! Returns the log of Epeak !!
  :param epeak: float or array of float containing peak energies
  """
  if version == 1:
    a1, a2, epb = -0.53, 4, 1600  # Took -a1 because the results were coherent only this way
    if type(epeak) is float or type(epeak) is int:
      if epeak <= epb:
        return (epeak / epb)**(-a1)
      else:
        return (epeak / epb)**(-a2)
    elif type(epeak) is np.ndarray:
      return np.where(epeak <= epb, (epeak / epb)**(-a1), (epeak / epb)**(-a2))
    else:
      raise TypeError("Please use a correct type for epeak, only accepted are float or numpy ndarray")
  elif version == 2:
    ampl, skewness, mu, sigma = 80,  1.6,  1.9,  0.44
    return ampl * skewnorm.pdf(epeak, skewness, mu, sigma)
  else:
    raise ValueError("Please use a correct version for epeak_distribution_short")


def epeak_distribution_short(epeak, version):
  """
  Version 1
    Peak energy distribution for short GRBs
    Function and associated parameters and cases are taken from Ghirlanda et al. 2016
  Version 2
    Distribution corresponding to the Epeak obs by Fermi GBM
    !! Returns the log of Epeak !!
  :param epeak: float or array of float containing peak energies
  """
  if version == 1:
    a1, a2, epb = -0.53, 4, 1600  # Took -a1 because the results were coherent only this way
    if type(epeak) is float or type(epeak) is int:
      if epeak <= epb:
        return (epeak / epb)**(-a1)
      else:
        return (epeak / epb)**(-a2)
    elif type(epeak) is np.ndarray:
      return np.where(epeak <= epb, (epeak / epb)**(-a1), (epeak / epb)**(-a2))
    else:
      raise TypeError("Please use a correct type for epeak, only accepted are float or numpy ndarray")
  elif version == 2:
    ampl, skewness, mu, sigma = 16, -5.2, 3.15, 0.66
    return ampl * skewnorm.pdf(epeak, skewness, mu, sigma)
  else:
    raise ValueError("Please use a correct version for epeak_distribution_short")


def amati_long(epeak,):
  """
  Version
    Amatie relation (Amati, 2006) linking Epeak and Eiso (peak energy and isotropic equivalent energy)
  :param epeak: float or array of float containing peak energies if reversed = True or isotropic equivalent energies if False
  :returns: Eiso
  """
  if type(epeak) is float or type(epeak) is int or type(epeak) is np.float64 or type(epeak) is np.ndarray:
    return 10**(52 + np.log10(epeak / 110) / 0.51)
  else:
    raise TypeError("Please use a correct type for energy, only accepted are float or numpy ndarray")


def amati_short(epeak):
  """
  Version
    Amatie relation (Amati, 2006) linking Epeak and Eiso (peak energy and isotropic equivalent energy)
  :param epeak: float or array of float containing peak energies if reversed = True or isotropic equivalent energies if False
  :returns: Eiso
  """
  ma, qa = 1.1, 0.042
  if type(epeak) is float or type(epeak) is int or type(epeak) is np.float64 or type(epeak) is np.ndarray:
    return 10**(51 + (np.log10(epeak / 670) - qa) / ma)
  else:
    raise TypeError("Please use a correct type for energy, only accepted are float or numpy ndarray")


def yonetoku_long(epeak):
  """
  Version
    Yonetoku relation for long GRBs (Yonetoku et al, 2010)
  :returns: Peak Luminosity
  """
  id1, s_id1, id2, s_id2 = 52.43, 0.037, 1.60, 0.082
  rand1 = np.random.normal(id1, s_id1)
  rand2 = np.random.normal(id2, s_id2)
  return 10 ** rand1 * (epeak / 355) ** rand2


def yonetoku_reverse_long(lpeak, version):
  """
  Version 0
    Yonetoku relation (Yonetoku et al, 2010)
    Coefficients from Sarah Antier's thesis
  Version 1
    Yonetoku relation for long GRBs (Yonetoku et al, 2010)
  :returns: Peak energy
  """
  if version == 0:
    ampl, l0, ind1 = 372, 1e52, 0.5
    return ampl * (lpeak / l0) ** ind1
  elif version == 1:
    id1, s_id1, id2, s_id2 = 52.43, 0.037, 1.60, 0.082
    rand1 = np.random.normal(id1, s_id1)
    rand2 = np.random.normal(id2, s_id2)
    return 355 * (lpeak/10**rand1)**(1/rand2)
  else:
    raise ValueError("Please use a correct version for Yonetoku long")


def yonetoku_short(epeak, version):
  """
  Version 0
    Yonetoku relation (Yonetoku et al, 2014) from Ghirlanda et al, 2016
  Version 1-2
    Yonetoku relation for short GRBs (Tsutsui et al, 2013)
  :returns: Peak luminosity
  """
  if version == 0:
    qy, my = 0.034, 0.84
    if type(epeak) is float or type(epeak) is int or type(epeak) is np.ndarray:
      return 10 ** (52 + (np.log10(epeak / 670) - qy) / my)
    else:
      raise TypeError("Please use a correct type for energy, only accepted are float or numpy ndarray")
  elif version == 1 or version == 2:
    id1, s_id1, id2, s_id2 = 52.29, 0.066, 1.59, 0.11
    rand1 = np.random.normal(id1, s_id1)
    rand2 = np.random.normal(id2, s_id2)
    return 10 ** rand1 * (epeak / 774.5) ** rand2
  else:
    raise ValueError("Please use a correct version for Yonetoku short")


def t90_long_log_distri(time, version):
  """
  Version 0-1-2
    Distribution of T90 based on GBM t90 >= 2 with a sbpl
      Not optimal as no correlation with other parameters is considered
  """
  if version == 0 or version == 1 or version == 2:
    ampl, skewness, mu, sigma = 54.18322289, -2.0422097,  1.89431034,  0.74602339
    return ampl * skewnorm.pdf(time, skewness, mu, sigma)
  else:
    raise ValueError("Please use a correct version for t90_long_log_distri")


def t90_short_distri(time, version):
  """
  Version 0-1-2
    Distribution of T90 based on GBM t90 < 2
      Not optimal as lGRB might biase the distribution
  """
  if version == 0 or version == 1 or version == 2:
    if type(time) is float or type(time) is int:
      if time <= 0.75:
        return 10**(1.7 + np.log10(time))
      else:
        return 10**(0.13 - 3.1 * np.log10(time))
    elif type(time) is np.ndarray:
      return np.where(time <= 0.75, 10**(1.7*np.log10(time)), 10**(0.13 - 3.17 * np.log10(time)))
    else:
      raise TypeError("Please use a correct type for time, only accepted are float or numpy ndarray")
  else:
    raise ValueError("Please use a correct version for t90_short_distri")

