import numpy as np
import gzip
from scipy.integrate import quad


def horizonAngle(h, EarthRadius=6371, AtmosphereHeight=40):
  """
  Calculates the angle between the zenith and the horizon for a LEO satellite
  :param h: altitude of the satellite (km)
  :param EarthRadius: radius of the Earth (km), default=6371
  :param AtmosphereHeight: height of the atmosphere (km), default=40
  :returns: horizon angle (deg)
  """
  return 90 + np.rad2deg(np.arccos((EarthRadius + AtmosphereHeight) / (EarthRadius + h)))  # deg


def treatCE(s):
  """
  Function to sum the 2 energy deposits given by trafiles for a compton event
  """
  return float(s[0]) + float(s[4])


def inwindow(E, ergcut):
  """
  Checks whether E is in the energy window defined by ergcut
  :param E: energy
  :param ergcut: (Emin, Emax)
  :returns: bool
  """
  if type(E) == np.ndarray:
    return np.where(E > ergcut[0], np.where(E < ergcut[1], True, False), False)
  elif type(E) == list:
    E = np.array(E)
    return np.where(E > ergcut[0], np.where(E < ergcut[1], True, False), False)
  else:
    return E > ergcut[0] and E < ergcut[1]


def readevt(fname, ergcut=None):
  """
  Reads a .tra or .tra.gz file and returns the Compton scattered gamma-ray vector
  :param fname: str, name of .tra file
  :param ergcut: couple (Emin, Emax) or None, energy range in which events have to be to be processed, default=None(=no selection)
  :returns:     list of 3-uple of float
  """
  if fname.endswith(".tra"):
    with open(fname) as f:
      data = f.read().split("SE")[1:]
  elif fname.endswith(".tra.gz"):
    with gzip.open(fname, "rt") as f:
      data = "".join(f).split("SE")[1:]
  else:
    raise TypeError("{} has unknown extension (known: .tra ou .tra.gz)".format(fname))

  ret = []
  if ergcut is None:
    for evt in data:
      if "ET CO" in evt:
        for line in evt.split("\n"):
          if line.startswith("CD"):
            dat = list(map(float, line.replace("   ", " ").split(" ")[1:]))
            ret.append((dat[6] - dat[0], dat[7] - dat[1], dat[8] - dat[2]))
  else:
    for evt in data:
      if "ET CO" in evt:
        lines = evt.split("\n")
        for i, line in enumerate(lines):
          if line.startswith("CE"):
            E = [float(e) for e in line.replace("   ", " ").split(" ")[1:]]
            if inwindow(E[0] + E[2], ergcut):
              dat = list(map(float, lines[i + 1].replace("   ", " ").split(" ")[1:]))
              ret.append((dat[6] - dat[0], dat[7] - dat[1], dat[8] - dat[2]))
  return ret


def angle(c, theta, phi):
  """
  Calculate the azimuthal Compton angle
  :param c:     3-uple, Compton scattered gamma-ray vector
  :param theta: float,  source polar angle in sky in rad
  :param phi:   float,  source azimuthal angle in sky in rad
  :returns:     float,  angle in deg
  """
  # Pluging in some MEGAlib magic
  c = (np.cos(-phi) * c[0] - np.sin(-phi) * c[1], np.sin(-phi) * c[0] + np.cos(-phi) * c[1], c[2])
  c = (np.sin(-theta) * c[2] + np.cos(-theta) * c[0], c[1], np.cos(-theta) * c[2] - np.sin(-theta) * c[0])
  # Figure out a good arctan
  if c[0] > 0:
    return np.arctan(c[1] / c[0]) * 180 / np.pi
  elif c[0] == 0:
    return 90
  else:
    if c[1] > 0:
      return np.arctan(c[1] / c[0]) * 180 / np.pi + 180
    else:
      return np.arctan(c[1] / c[0]) * 180 / np.pi - 180


def analyzetra(fname, theta=0, phi=0, pa=0, corr=False, ergcut=None):
  """
  Reads a .tra file and returns the azimuthal angles used in polarigrams (corrected from the source sky position and from cosima's "RelativeY" polarization definition)
  :param fname: str,   name of file to read from
  :param theta: float, polar angle of source in sky in rad, default=0
  :param phi:   float, azimuthal angle of source in sky in rad, default=0
  :param pa:    float, polarization angle in source file in rad, default=0
  :param corr:  bool,  wether to correct for the source sky position and cosima's "RelativeY" polarization definition or not, default=False
  :param ergcut: couple (Emin,Emax) or None, energy range in which to perform polarization analysis, default=None (no selection)
  :returns:     list of float, azimuthal angles of Compton scattered gamma-rays
  """
  data = readevt(fname, ergcut)
  angles = []
  if corr:
    for evt in data:
      angles.append(angle(evt, theta, phi) + np.rad2deg(np.arctan(np.cos(theta) * np.tan(phi)) + pa))
  else:
    for evt in data:
      angles.append(angle(evt, theta, phi))
  return angles


def modulation_func(x, pa, mu, S):
  """
  Polarigram model
  :param x:  float or np.array, azimutal compt angle
  :param pa: float,             polarization angle
  :param mu: float,             contrast of polarization signal
  :param S:  float,             source flux
  :returns:  float or np.array
  """
  return (S / (2 * np.pi)) * (1 - mu * np.cos(np.pi * (x - pa) / 90))


def fname2decra(fname, polmark="inc1"):
  """
  Infers dec and RA from file name
  :param fname: *.tra or *.tra.gz filename
  :param polmark: str that identifies polarized files, default='inc1'
  :returns: dec, RA, polarized
  """
  data = fname.split("_")
  return float(data[4]), float(".".join(data[5].split(".")[:2]))  # , polmark in data[3]


def decra2tp(dec, ra, s, unit="deg"):
  """
  Converts dec,ra (declination, right ascension) world coordinates into satellite coordinate
  :param dec: declination (except it is 0 at north pole, 90° at equator and 180° at south pole)
  :param ra : Right ascension (0->360°)
  :param s: satellite from infos['satellites']
  :param unit: unit in which are given dec and ra, default="deg"
  :returns: theta_sat, phi_sat in rad
  """
  if unit=='deg':
    dec, ra = np.deg2rad(dec), np.deg2rad(ra)
  theta = np.arccos( np.product(np.sin(np.array([dec, ra, s[0], s[1]]))) + np.sin(dec)*np.cos(ra)*np.sin(s[0])*np.cos(s[1]) + np.cos(dec)*np.cos(s[0]) )
  source = [np.sin(dec)*np.cos(ra), np.sin(dec)*np.sin(ra), np.cos(dec)]
  yprime = [-np.cos(s[0])*np.cos(s[1]), -np.cos(s[0])*np.sin(s[1]), np.sin(s[0])]
  xprime = [-np.sin(s[1]), np.cos(s[1]), 0]
  phi = np.mod(np.arctan2(np.dot(source, yprime), np.dot(source, xprime)), 2*np.pi)
  return theta, phi


def decra2tpPA(dec, ra, s, unit="deg"):
  """
  Converts dec,ra (declination, right ascension) world coordinates into satellite attitude parameters
  Polarization angle calculation rely on the fact that the polarization angle in is the plane generated by the direction of the source and the dec=0 direction, as it is the case in mamr.py.
  :param dec: declination (except it is 0 at north pole, 90° at equator and 180° at south pole)
  :param ra : Right ascension (0->360°)
  :param s: satellite from info_sat : [thetasat, phisat, horizonAngle]
  :param unit: unit in which are given dec and ra, default="deg"
  :returns: theta_sat, phi_sat, polarization angle in deg with MEGAlib's RelativeY convention
  """
  if unit == 'deg':
    dec, ra = np.deg2rad(dec), np.deg2rad(ra)
  theta = np.arccos(
    np.product(np.sin(np.array([dec, ra, s[0], s[1]]))) + np.sin(dec) * np.cos(ra) * np.sin(s[0]) * np.cos(
      s[1]) + np.cos(dec) * np.cos(s[0]))
  source = [np.sin(dec) * np.cos(ra), np.sin(dec) * np.sin(ra), np.cos(dec)]
  yprime = [-np.cos(s[0]) * np.cos(s[1]), -np.cos(s[0]) * np.sin(s[1]), np.sin(s[0])]
  xprime = [-np.sin(s[1]), np.cos(s[1]), 0]
  phi = np.mod(np.arctan2(np.dot(source, yprime), np.dot(source, xprime)), 2 * np.pi)
  # Polarization
  dec_p, ra_p = np.mod(.5 * np.pi - dec,
                       np.pi), ra + np.pi  # polarization direction in world coordinates (towards north or south pole)
  vecpol = [np.sin(dec_p) * np.cos(ra_p), np.sin(dec_p) * np.sin(ra_p),
            np.cos(dec_p)]  # polarization vector in world coordinates
  return np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(np.arccos(np.dot(vecpol, np.cross(source, yprime))))


def SNR(S, B, C=0):
  """
  Calculates the signal to noise ratio of a GRB in a time bin
  :param S: number of counts in the source (background included)
  :param B: expected number of background counts
  :param C: minimum number of counts in the source to consider the detection
  :returns: SNR (as defined in Sarah Antier's PhD thesis
  """
  return (S - B) / np.sqrt(B + C)


def MDP(S, B, mu100, nsigma=4.29):
  """
  Calculates the minimum detectable polarization for a burst
  :param S: number of expected counts from the burst
  :param B: number of expected counts from the background
  :param mu100: modulation factor
  :param nsigma: significance of the result in number of sigmas, default=4.29 for 99% CL
  """
  return nsigma * np.sqrt(S + B) / (mu100 * S)


#######################################################################################################
# Functions to create spectra                                                                         #
#######################################################################################################
def plaw(e, A, l, pivot=100):
  """
  Power-law spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  :param l: spectral index
  :param pivot: pivot energy (keV), depends only on the instrument, default=100 keV for Fermi/GBM
  :returns: ph/cm2/keV/s
  """
  return A * (e / pivot) ** l


def comp(e, A, l, ep, pivot=100):
  """
  Comptonized spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  :param l: spectral index
  :param ep: peak energy (keV)
  :param pivot: pivot energy (keV), depends only on the instrument, default=100 keV for Fermi/GBM
  :returns: ph/cm2/keV/s
  """
  return A * (e / pivot) ** l * np.exp(-(l + 2) * e / ep)


def glog(e, A, ec, s):
  """
  log10-gaussian spectrum model
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  :param ec: central energy (keV)
  :param s: distribution width
  :returns: ph/cm2/keV/s
  """
  return A / np.sqrt(2 * np.pi * s) * np.exp(-.5 * (np.log10(e / ec) / s) ** 2)


def band(e, A, alpha, beta, ep, pivot=100):
  """
  Band spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  :param alpha: low-energy spectral index
  :param beta: high-energy spectral index
  :param ep: peak energy (keV)
  :param pivot: pivot energy (keV), depends only on the instrument, default=100 keV for Fermi/GBM
  :returns: ph/cm2/keV/s
  """
  c = (alpha - beta) * ep / (alpha + 2)
  if e > c:
    return A * (e / pivot) ** beta * np.exp(beta - alpha) * (c / pivot) ** (alpha - beta)
  else:
    return A * (e / pivot) ** alpha * np.exp(-(alpha + 2) * e / ep)


def sbpl_sa(e, A, l1, l2, eb, delta, pivot=100):
  """
  Smoothly broken power law spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  """
  b, m = .5 * (l1 + l2), .5 * (l1 - l2)
  q, qp = np.log10(e / eb / delta), np.log10(pivot / eb / delta)
  a, ap = m * delta * np.log(np.cosh(q)), m * delta * np.log(np.cosh(qp))
  return A * (e / pivot) ** b * 10 ** (a / ap)


def sbpl(e, A, l1, l2, eb, delta, pivot=100):
  """
  Smoothly broken power law spectrum
  :param e: energy (keV)
  :param A: amplitude (ph/cm2/keV/s)
  """
  b, m = .5 * (l2 + l1), .5 * (l2 - l1)
  q, qp = np.log10(e / eb) / delta, np.log10(pivot / eb) / delta
  a, ap = m * delta * np.log(np.cosh(q)), m * delta * np.log(np.cosh(qp))
  return A * (e / pivot) ** b * 10 ** (a - ap)


def closest_bkg_rate(sat_lat, bkg_list):
  """
  Find the closest bkg file for a satellite (in terms of latitude)
  Returns the count rate of this bkg file
  """
  sat_lat = np.rad2deg(sat_lat)
  if len(bkg_list) == 0:
    return 0.000001
  else:
    latitude_error = np.array([abs(bkg.dec - sat_lat) for bkg in bkg_list])
    return bkg_list[np.argmin(latitude_error)].cr



def calc_fluence(catalog, index, ergCut):
  """
  Return the number of photons per cm² for a given energy range, averaged over the duration of the sim : ncount/cm²/s
  """

  catalog.tofloat('flnc_spectrum_start')
  catalog.tofloat('flnc_spectrum_stop')
  catalog.tofloat('pflx_plaw_ampl')
  catalog.tofloat('pflx_plaw_index')
  catalog.tofloat('pflx_plaw_pivot')
  catalog.tofloat('pflx_comp_ampl')
  catalog.tofloat('pflx_comp_index')
  catalog.tofloat('pflx_comp_epeak')
  catalog.tofloat('pflx_comp_pivot')
  catalog.tofloat('pflx_band_ampl')
  catalog.tofloat('pflx_band_alpha')
  catalog.tofloat('pflx_band_beta')
  catalog.tofloat('pflx_band_epeak')
  catalog.tofloat('pflx_sbpl_ampl')
  catalog.tofloat('pflx_sbpl_indx1')
  catalog.tofloat('pflx_sbpl_indx2')
  catalog.tofloat('pflx_sbpl_brken')
  catalog.tofloat('pflx_sbpl_brksc')
  catalog.tofloat('pflx_sbpl_pivot')
  catalog.tofloat('flnc_plaw_ampl')
  catalog.tofloat('flnc_plaw_index')
  catalog.tofloat('flnc_plaw_pivot')
  catalog.tofloat('flnc_comp_ampl')
  catalog.tofloat('flnc_comp_index')
  catalog.tofloat('flnc_comp_epeak')
  catalog.tofloat('flnc_comp_pivot')
  catalog.tofloat('flnc_band_ampl')
  catalog.tofloat('flnc_band_alpha')
  catalog.tofloat('flnc_band_beta')
  catalog.tofloat('flnc_band_epeak')
  catalog.tofloat('flnc_sbpl_ampl')
  catalog.tofloat('flnc_sbpl_indx1')
  catalog.tofloat('flnc_sbpl_indx2')
  catalog.tofloat('flnc_sbpl_brken')
  catalog.tofloat('flnc_sbpl_brksc')
  catalog.tofloat('flnc_sbpl_pivot')

  model = catalog.flnc_best_fitting_model[index].strip()
  if model == "pflx_plaw":
    func = lambda x: plaw(x, catalog.pflx_plaw_ampl[index], catalog.pflx_plaw_index[index], catalog.pflx_plaw_pivot[index])
  elif model == "pflx_comp":
    func = lambda x: comp(x, catalog.pflx_comp_ampl[index], catalog.pflx_comp_index[index], catalog.pflx_comp_epeak[index],
                          catalog.pflx_comp_pivot[index])
  elif model == "pflx_band":
    func = lambda x: band(x, catalog.pflx_band_ampl[index], catalog.pflx_band_alpha[index], catalog.pflx_band_beta[index],
                          catalog.pflx_band_epeak[index])
  elif model == "pflx_sbpl":
    func = lambda x: sbpl(x, catalog.pflx_sbpl_ampl[index], catalog.pflx_sbpl_indx1[index], catalog.pflx_sbpl_indx2[index],
                          catalog.pflx_sbpl_brken[index], catalog.pflx_sbpl_brksc[index], catalog.pflx_sbpl_pivot[index])
  elif model == "flnc_plaw":
    func = lambda x: plaw(x, catalog.flnc_plaw_ampl[index], catalog.flnc_plaw_index[index], catalog.flnc_plaw_pivot[index])
  elif model == "flnc_comp":
    func = lambda x: comp(x, catalog.flnc_comp_ampl[index], catalog.flnc_comp_index[index], catalog.flnc_comp_epeak[index],
                          catalog.flnc_comp_pivot[index])
  elif model == "flnc_band":
    func = lambda x: band(x, catalog.flnc_band_ampl[index], catalog.flnc_band_alpha[index], catalog.flnc_band_beta[index],
                          catalog.flnc_band_epeak[index])
  elif model == "flnc_sbpl":
    func = lambda x: sbpl(x, catalog.flnc_sbpl_ampl[index], catalog.flnc_sbpl_indx1[index], catalog.flnc_sbpl_indx2[index],
                          catalog.flnc_sbpl_brken[index], catalog.flnc_sbpl_brksc[index], catalog.flnc_sbpl_pivot[index])
  else:
    print("Could not find best fit model for {} (indicated {}). Aborting this GRB.".format(catalog.name[index], model))
    return
  return quad(func, ergCut[0], ergCut[1])[0]


def eff_area_pola_func(theta, angle_lim, func_type="cos"):
  if func_type == "cos":
    if theta < np.deg2rad(angle_lim):
      ampl = 5.5
      ang_freq = 0.222
      phi0 = 0.76
      y_off_set = 2.5
      return np.absolute((ampl) * np.cos(theta * 2 * np.pi * ang_freq - phi0)) + y_off_set
    else:
      return 0
  elif func_type == "FoV":
    if theta < np.deg2rad(angle_lim):
      return 1
    else:
      return 0


def eff_area_spectro_func(theta, angle_lim, func_type="data"):
  if func_type == "data":
    if theta < np.deg2rad(angle_lim):
      angles = np.deg2rad(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 89, 91, 100, 110]))
      eff_area = np.array([137.4, 148.5, 158.4, 161.9, 157.4, 150.4, 133.5, 112.8, 87.5, 63.6, 64.7, 71.8, 77.3])
      interpo_ite = 1
      if theta > angles[-1]:
        return eff_area[-2] + (eff_area[-1] - eff_area[-2]) / (angles[-1] - angles[-2]) * (theta - angles[-2])
      else:
        while theta > angles[interpo_ite]:
          interpo_ite += 1
        return eff_area[interpo_ite - 1] + (eff_area[interpo_ite] - eff_area[interpo_ite - 1]) / (
                  angles[interpo_ite] - angles[interpo_ite - 1]) * (theta - angles[interpo_ite - 1])
    else:
      return 0
  elif func_type == "FoV":
    if theta < np.deg2rad(angle_lim):
      return 1
    else:
      return 0
