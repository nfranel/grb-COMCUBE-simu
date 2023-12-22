import numpy as np
import gzip
from scipy.integrate import quad
from time import time

#UTILISER ASTROPY;constant ?
m_elec = 9.1094e-31 #kg
c_light = 2.99792458e+8 #m/s
charge_elem = 1.6021e-19 #C
m_earth = 5.9722e24 #kg
R_earth = 6371 #km
G_const = 6.67430e-11 #m3/kg/s2
earth_rot_time = 86164 + 98e-3 + 903e-6 + 697e-9 #s

def printv(message, verbose):
  if verbose:
    print(message)


def horizonAngle(h, EarthRadius=R_earth, AtmosphereHeight=40):
  """
  Calculates the angle between the zenith and the horizon for a LEO satellite
  :param h: altitude of the satellite (km)
  :param EarthRadius: radius of the Earth (km), default=6371
  :param AtmosphereHeight: height of the atmosphere (km), default=40
  :returns: horizon angle (deg)
  """
  if h>=AtmosphereHeight:
    return 90 + np.rad2deg(np.arccos((EarthRadius + AtmosphereHeight) / (EarthRadius + h)))  # deg
  else:
    return 90


def orbital_period_calc(alt):
  """
  Calculates the orbital period of a satellite at a specific altitude
  :param alt : altitude of the orbit
  """
  return np.sqrt(4 * np.pi**2 / (G_const * m_earth) * ((R_earth + alt) * 1000)**3)


def earth_rotation_offset(time):
  """
  Calculates the offset in right ascension due to the earth rotation.
  The earth turns from W to E, so if the satellite orbits from W to E this correction has to be deducted from the
  calculated RA. If not it has to be added.
  :param time : time at which the rotational offset is calculated in seconds
  returns a correction angle in deg
  """
  return np.mod(360 * time / earth_rot_time, 360)


def treatCE(ener):
  """
  Function to sum the 2 energy deposits given by trafiles for a compton event
  """
  return np.array([float(ener[0]), float(ener[4])])


def treatPE(ener):
  """
  Function to sum the 2 energy deposits given by trafiles for a compton event
  """
  return float(ener)


def calculate_polar_angle(ener_sec, ener_tot):
  """
  Function to calculate the polar angle using the energy deposits
  This function is made so that the cos of the angle is >=-1 as it's not possible to take the arccos of a number <-1.
  By construction of cos_value, the value cannot exceed 1.
  """
  cos_value = 1 - m_elec * c_light ** 2 / charge_elem / 1000 * (1 / ener_sec - 1 / ener_tot)
  cos_value_filtered = np.where(cos_value < -1, -1, np.where(cos_value > 1, 1, cos_value))
  return np.rad2deg(np.arccos(cos_value_filtered))


def inwindow(E, ergcut):
  """
  Checks whether E is in the energy window defined by ergcut
  :param E: energy
  :param ergcut: (Emin, Emax)
  :returns: bool
  """
  # print(E)
  return E >= ergcut[0] and E <= ergcut[1]


def fname2decra(fname):
  """
  Infers dec and RA from file name with the shape :
    {prefix}_{sourcename}_sat{num_sat}_{num_sim}_{dec_world_frame}_{ra_world_frame}.inc{1/2}.id1.extracted.tra
  :param fname: *.tra or *.tra.gz filename
  :param polmark: str that identifies polarized files, default='inc1'
  :returns: dec, RA
  """
  data = fname.split("/")[-1] # to get rid of the first part of the prefix (and the potential "_" in it)
  data = data.split("_")
  return float(data[4]), float(".".join(data[5].split(".")[:2])), data[1], int(data[3]), int(data[2].split("sat")[1])


def fname2decratime(fname):
  """
  Infers dec and RA from file name with the shape :
    {prefix}_{sourcename}_sat{num_sat}_{num_sim}_{dec_world_frame}_{ra_world_frame}_{burst_time}.inc{1/2}.id1.extracted.tra
  :param fname: *.tra or *.tra.gz filename
  :param polmark: str that identifies polarized files, default='inc1'
  :returns: dec, RA
  """
  data = fname.split("/")[-1] # to get rid of the first part of the prefix (and the potential "_" in it)
  data = data.split("_")
  return float(data[4]), float(data[5]), float(".".join(data[6].split(".")[:2])), data[1], int(data[3]), int(data[2].split("sat")[1])


def save_log(filename, name, num_sim, num_sat, status, inc, ohm, omega, alt, random_time, sat_dec_wf, sat_ra_wf, grb_dec_wf, grb_ra_wf, grb_dec_st, grb_ra_sf):
  """
  Saves all the simulation informations into a log file.
  May be used to make sure everything works or to make some plots
  """
  with open(filename, "a") as f:
    f.write(f"{name} | {num_sim} | {num_sat} | {status} | {inc} | {ohm} | {omega} | {alt} | {random_time} | {sat_dec_wf} | {sat_ra_wf} | {grb_dec_wf} | {grb_ra_wf} | {grb_dec_st} | {grb_ra_sf}\n")


def read_grbpar(parfile):
  """

  """
  sat_info = []
  with open(parfile) as f:
    lines = f.read().split("\n")
  for line in lines:
    if line.startswith("@geometry"):
      geometry = line.split(" ")[1]
    elif line.startswith("@revancfgfile"):
      revan_file = line.split(" ")[1]
    elif line.startswith("@mimrecfile"):
      mimrec_file = line.split(" ")[1]
    elif line.startswith("@spectrafilepath"):
      spectra_path = line.split(" ")[1]
    elif line.startswith("@grbfile"):
      cat_file = line.split(" ")[1]
    elif line.startswith("@cosimasourcefile"):
      source_file = line.split(" ")[1]
    elif line.startswith("@prefix"):
      sim_prefix = line.split(" ")[1]
    elif line.startswith("@sttype"):
      sttype = line.split(" ")[1:]
    elif line.startswith("@simulationsperevent"):
      n_sim = int(line.split(" ")[1])
    elif line.startswith("@simtime"):
      simtime = line.split(" ")[1]
    elif line.startswith("@position"):
      position_allowed_sim = np.array(line.split(" ")[1:], dtype=float)
    elif line.startswith("@satellite"):
      dat = [float(e) for e in line.split(" ")[1:]]
      sat_info.append(dat)
  return geometry, revan_file, mimrec_file, spectra_path, cat_file, source_file, sim_prefix, sttype, n_sim, simtime, position_allowed_sim, sat_info


def read_bkgpar(parfile):
  with open(parfile, "r") as f:
    lines = f.read().split("\n")
  geom, revanf, mimrecf, source_base, spectra, simtime, latitudes, altitudes = None, None, None, None, None, None, None, None
  for line in lines:
    if line.startswith("@geometry"):
      geom = line.split(" ")[1]
    elif line.startswith("@revancfgfile"):
      revanf = line.split(" ")[1]
    elif line.startswith("@mimrecfile"):
      mimrecf = line.split(" ")[1]
    elif line.startswith("@cosimasourcefile"):
      source_base = line.split(" ")[1]
    elif line.startswith("@spectrafilepath"):
      spectra = line.split(" ")[1]
      if spectra.endswith("/"):
        spectra = spectra[:-1]
    elif line.startswith("@simtime"):
      simtime = float(line.split(" ")[1])
    elif line.startswith("@altitudes"):
      altitudes = list(map(float, line.split(" ")[1:]))
    elif line.startswith("@latitudes"):
      latitudes = list(map(int, line.split(" ")[1:]))
      latitudes = np.linspace(latitudes[0], latitudes[1], latitudes[2])
  return geom, revanf, mimrecf, source_base, spectra, simtime, latitudes, altitudes


def read_mupar(parfile):
  with open(parfile, "r") as f:
    lines = f.read().split("\n")
  geom, revanf, mimrecf, source_base, spectra, bandparam, poltime, unpoltime, decs, ras = None, None, None, None, None, None, None, None, None, None
  for line in lines:
    if line.startswith("@geometry"):
      geom = line.split(" ")[1]
    elif line.startswith("@revancfgfile"):
      revanf = line.split(" ")[1]
    elif line.startswith("@mimrecfile"):
      mimrecf = line.split(" ")[1]
    elif line.startswith("@cosimasourcefile"):
      source_base = line.split(" ")[1]
    elif line.startswith("@spectrafilepath"):
      spectra = line.split(" ")[1]
      if spectra.endswith("/"):
        spectra = spectra[:-1]
    elif line.startswith("@bandparam"):
      bandparam = list(map(float, line.split(" ")[1:]))
    elif line.startswith("@poltime"):
      poltime = float(line.split(" ")[1])
    elif line.startswith("@unpoltime"):
      unpoltime = float(line.split(" ")[1])
    elif line.startswith("@decposition"):
      decs = list(map(int, line.split(" ")[1:]))
      decs = np.linspace(decs[0], decs[1], decs[2])
    elif line.startswith("@raposition"):
      ras = list(map(int, line.split(" ")[1:]))
      # ras = np.linspace(ras[0], ras[1], ras[2])
  return geom, revanf, mimrecf, source_base, spectra, bandparam, poltime, unpoltime, decs, ras


def readfile(fname):
  """
  Reads a .tra or .tra.gz file and returns the extracted information if the event is in the energy range
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
  return data


def readevt(event, ergcut=None):
  """
  Reads an event and returns the information about this event if it's in the energy range
  :param event: str, event in a trafile
  :param ergcut: couple (Emin, Emax) or None, energy range in which events have to be to be processed, default=None(=no selection)
  :returns:     list of 3-uple of float
  """
  lines = event.split("\n")[1:-1]
  # print(lines)
  if lines[0] == "ET CO":
    # print("Extracting")
    # if lines[3] == "SQ 2":
    second_ener = float(lines[7].split(" ")[1])
    total_ener = second_ener + float(lines[7].split(" ")[5])
    if ergcut is None:
      time_interaction = float(lines[2].split(" ")[1])
      first_pos = np.array([float(lines[8].split(" ")[11]), float(lines[8].split(" ")[12]), float(lines[8].split(" ")[13])])
      second_pos = np.array([float(lines[8].split(" ")[1]), float(lines[8].split(" ")[2]), float(lines[8].split(" ")[3])])
      return [second_ener, total_ener, time_interaction, first_pos, second_pos]
    else:
      if inwindow(total_ener, ergcut):
        time_interaction = float(lines[2].split(" ")[1])
        first_pos = np.array([float(lines[8].split(" ")[11]), float(lines[8].split(" ")[12]), float(lines[8].split(" ")[13])])
        second_pos = np.array([float(lines[8].split(" ")[1]), float(lines[8].split(" ")[2]), float(lines[8].split(" ")[3])])
        return [second_ener, total_ener, time_interaction, first_pos, second_pos]
      else:
        return [None]
    # else:
    #   return [None]
  elif lines[0] == "ET PH":
    total_ener = float(lines[3].split(" ")[1])
    if ergcut is None:
      time_interaction = float(lines[2].split(" ")[1])
      pos = np.array([float(lines[4].split(" ")[1]), float(lines[4].split(" ")[2]), float(lines[4].split(" ")[3])])
      return [total_ener, time_interaction, pos]
    else:
      if inwindow(total_ener, ergcut):
        time_interaction = float(lines[2].split(" ")[1])
        pos = np.array([float(lines[4].split(" ")[1]), float(lines[4].split(" ")[2]), float(lines[4].split(" ")[3])])
        return [total_ener, time_interaction, pos]
      else:
        return [None]
  else:
    raise TypeError(f"An event has an unidentified type")


def angle(c, theta, phi, source_name, num_sim, num_sat):
  """
  Calculate the azimuthal Compton angle : Transforms the compton scattered gamma-ray vector (initialy in sat frame) into
  a new referential corresponding to the direction of the source. In that frame c0 and c1 are the coordinates of
  the vector in the plan orthogonal to the source direction. The x coordinate is the vector in the plane created by
  the zenith (z axis) of the instrument and the source direction and y is in the plane of the detector (zcoord=0)
  (so x is in the plane containing the zworld, source direction, and the axis yprime of the detector)
  The way the azimuthal scattering angle is calculated imply that the polarization vector is colinear with x
  Calculates the polar Compton angle
  :param c:     3-uple, Compton scattered gamma-ray vector
  :param theta: float,  source polar angle in sky in deg
  :param phi:   float,  source azimuthal angle in sky in deg
  :returns:     float,  angle in deg
  """
  if len(c) == 0:
    print(f"There is no compton event detected for source {source_name}, simulation {num_sim} and satellite {num_sat}")
    return np.array([]), np.array([])
  theta, phi = np.deg2rad(theta), np.deg2rad(phi)
  # Changing the direction of the vector to be in adequation with MEGAlib's functionning
  MEGAlib_direction = True
  if MEGAlib_direction:
    c = -c
  # Pluging in some MEGAlib magic
  c = c / np.reshape(np.linalg.norm(c, axis=1), (len(c), 1))
  # Rotation matrix around Z with an angle -phi
  mat1 = np.array([[np.cos(-phi), - np.sin(-phi), 0],
                   [np.sin(-phi), np.cos(-phi), 0],
                   [0, 0, 1]])
  # Rotation matrix around Y with an angle -theta
  mat2 = np.array([[np.cos(-theta), 0, np.sin(-theta)],
                   [0, 1, 0],
                   [- np.sin(-theta), 0, np.cos(-theta)]])
  # using matrix products to combine the matrix instead of doing it vector by vector
  c = np.matmul(c, np.transpose(mat1))
  c = np.matmul(c, np.transpose(mat2))
  if MEGAlib_direction:
    polar = 180 - np.rad2deg(np.arccos(c[:, 2]))
  else:
    polar = np.rad2deg(np.arccos(c[:, 2]))
  # Figure out a good arctan
  azim = np.rad2deg(np.arctan2(c[:, 1], c[:, 0]))
  return azim, polar


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


def set_bins(bin_mode, data=None):
  """

  """
  bins = ""
  if bin_mode == "fixed":
    bins = np.linspace(-180, 180, 21)
  elif bin_mode == "limited":
    bins = []
  elif bin_mode == "optimized":
    bins = []
  return bins


def err_calculation(pol, unpol, binwidth):
  """
  Calculation of the errorbar of the corrected polarigram according to megalib's way
  :param pol:      list,             bins for the polarized polarigram
  :param unpol:    list,             bins for the unpolarized polarigram
  :param binwidth: list,             bin widths
  """
  nbins = len(pol)
  mean_unpol = np.mean(unpol)

  uncertainty = (pol/unpol**2*mean_unpol*np.sqrt(unpol))**2 + (mean_unpol/unpol*np.sqrt(pol))**2
  # print(uncertainty)
  for ite_j in range(nbins):
    uncertainty += (pol / unpol / nbins * np.sqrt(unpol[ite_j])) ** 2
    # print(uncertainty)
  error = np.sqrt(uncertainty)
  return error/binwidth


def rescale_cr_to_GBM_pf(cr, GBM_mean_flux, GBM_peak_flux):
  """

  """
  flux_ratio = GBM_peak_flux / GBM_mean_flux
  return cr * flux_ratio


def SNR(S, B, C=0):
  """
  Calculates the signal to noise ratio of a GRB in a time bin
  :param S: number of counts in the source (background not included)
  :param B: expected number of background counts
  :param C: minimum number of counts in the source to consider the detection
  :returns: SNR (as defined in Sarah Antier's PhD thesis)
  """
  return S / np.sqrt(B + C)


def MDP(S, B, mu100, nsigma=4.29):
  """
  Calculates the minimum detectable polarization for a burst
  :param S: number of expected counts from the burst
  :param B: number of expected counts from the background
  :param mu100: modulation factor
  :param nsigma: significance of the result in number of sigmas, default=4.29 for 99% CL
  """
  return nsigma * np.sqrt(S + B) / (mu100 * S)


def closest_bkg_rate(sat_dec, sat_ra, bkg_list):
  """
  Find the closest bkg file for a satellite (in terms of latitude)
  Returns the count rate of this bkg file
  Warining : for now, only takes into account the dec of backgrounds, can be updated but the way the error is calculated may not be optimal as the surface of the sphere (polar coordinates) is not a plan.
  """
  if len(bkg_list) == 0:
    return 0.000001
  else:
    dec_error = np.array([(bkg.dec - sat_dec) ** 2 for bkg in bkg_list])
    # ra_error = np.array([(bkg.ra - sat_ra) ** 2 for bkg in bkg_list])
    ra_error = np.array([0 for bkg in bkg_list])
    total_error = np.sqrt(dec_error + ra_error)
    index = np.argmin(total_error)
    return [bkg_list[index].compton_cr, bkg_list[index].single_cr]


def affect_bkg(info_sat, burst_time, bkg_list):
  """

  """
  orbital_period = orbital_period_calc(info_sat[3])
  earth_ra_offset = earth_rotation_offset(burst_time)
  true_anomaly = true_anomaly_calc(burst_time, orbital_period)
  dec_sat_world_frame, ra_sat_world_frame = orbitalparam2decra(info_sat[0], info_sat[1], info_sat[2], nu=true_anomaly)  # deg
  ra_sat_world_frame -= earth_ra_offset
  count_rates = closest_bkg_rate(dec_sat_world_frame, ra_sat_world_frame, bkg_list)
  return dec_sat_world_frame, ra_sat_world_frame, count_rates[0], count_rates[1]


def closest_mufile(grb_dec_sf, grb_ra_sf, mu_list):
  """
  Find the closest bkg file for a satellite (in terms of latitude)
  Returns the count rate of this bkg file
  Warining : for now, only takes into account the dec of backgrounds, can be updated but the way the error is calculated may not be optimal as the surface of the sphere (polar coordinates) is not a plan.
  """
  if len(mu_list) == 0:
    return 0.000001, 0.000001, 0.000001, 0.000001
  else:
    dec_error = np.array([(mu.dec - grb_dec_sf) ** 2 for mu in mu_list])
    ra_error = np.array([(mu.ra - grb_ra_sf) ** 2 for mu in mu_list])
    total_error = np.sqrt(dec_error + ra_error)
    index = np.argmin(total_error)
    return mu_list[index].mu100, mu_list[index].mu100_err, mu_list[index].s_eff_compton, mu_list[index].s_eff_single


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
  print(ergCut)
  return quad(func, ergCut[0], ergCut[1])[0]


def duty_calc(inclination):
  print("inc value : ", inclination)
  precise = False
  if precise:
    if inclination <= 5:
      return 1.
    elif inclination <= 10:
      return 0.95
    elif inclination <= 15:
      return 0.90
    elif inclination <= 20:
      return 0.84
    elif inclination <= 25:
      return 0.75
    elif inclination <= 30:
      return 0.7
    else:
      return 0.65
  else:
    if inclination == 5 or inclination == 0:
      return 1.
    elif inclination == 45:
      return 0.67
    elif inclination == 83 or inclination == 82.5:
      return 0.5


def eff_area_compton_func(theta, angle_lim, func_type="cos", duty=1.):
  """
  Returns a value of the effective area for polarisation based on a cos function to account for the reception angle relative to the instrument's zenith
  This is an approximation as the cos function does not perfectly fit the data
  If func_type "FoV" computes instead the number of satellites viewing that part of the sky (no sensitivity considered)
  """
  theta, angle_lim = np.deg2rad(theta), np.deg2rad(angle_lim)
  if duty < 0 or duty > 1:
    print("Error estimating the duty time, incorrect value")
    return 0
  if func_type == "cos":
    if theta < angle_lim:
      # ampl = 5.5
      ampl = 1
      ang_freq = 0.222
      phi0 = 0.76
      y_off_set = 2.5
      return (np.absolute((ampl) * np.cos(theta * 2 * np.pi * ang_freq - phi0)) + y_off_set) * duty
    else:
      return 0
  elif func_type == "FoV":
    if theta < angle_lim:
      return 1 * duty
    else:
      return 0


def eff_area_single_func(theta, angle_lim, func_type="data", duty=True):
  """
  Returns a value of the effective area for spectrometry based on interpolation from values obtained from different reception angle relative to the instrument's zenith
  This is an approximation as the values used are obtained for monoenergetic simulations - grbs are not and sensitivity of the instrument depends on energy
  If func_type "FoV" computes instead the number of satellites viewing that part of the sky (no sensitivity considered)
  """
  if duty < 0 or duty > 1:
    print("Error estimating the duty time, incorrect value")
    return 0
  if func_type == "data":
    if theta < angle_lim:
      angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 89, 91, 100, 110])
      eff_area = np.array([137.4, 148.5, 158.4, 161.9, 157.4, 150.4, 133.5, 112.8, 87.5, 63.6, 64.7, 71.8, 77.3])
      interpo_ite = 1
      if theta > angles[-1]:
        return (eff_area[-2] + (eff_area[-1] - eff_area[-2]) / (angles[-1] - angles[-2]) * (theta - angles[-2])) * duty
      else:
        while theta > angles[interpo_ite]:
          interpo_ite += 1
        return (eff_area[interpo_ite - 1] + (eff_area[interpo_ite] - eff_area[interpo_ite - 1]) / (angles[interpo_ite] - angles[interpo_ite - 1]) * (theta - angles[interpo_ite - 1])) * duty
    else:
      return 0
  elif func_type == "FoV":
    if theta < angle_lim:
      return 1 * duty
    else:
      return 0


#######################################################################################################
# Functions to manipulate the referential and positions of sat and grbs                               #
#######################################################################################################


def grb_decra_worldf2satf(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf):
  """
  Converts dec,ra (declination, right ascension) world coordinates into satellite coordinate
  :param dec: declination (except it is 0 at north pole, 90° at equator and 180° at south pole)
  :param ra : Right ascension (0->360°)
  :param s: satellite from infos['satellites']
  :param unit: unit in which are given dec and ra, default="deg"
  :returns: theta_sat, phi_sat in rad
  """
  dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf = np.deg2rad(dec_grb_wf), np.deg2rad(ra_grb_wf), np.deg2rad(dec_sat_wf), np.deg2rad(ra_sat_wf)
  # source being the direction of the source in world coordinates
  source = [np.sin(dec_grb_wf)*np.cos(ra_grb_wf),
            np.sin(dec_grb_wf)*np.sin(ra_grb_wf),
            np.cos(dec_grb_wf)]
  # Referential for the satellite in world coordinates
  z_ref_sat = [np.sin(dec_sat_wf)*np.cos(ra_sat_wf),
               np.sin(dec_sat_wf)*np.sin(ra_sat_wf),
               np.cos(dec_sat_wf)]
  y_ref_sat = [-np.cos(dec_sat_wf)*np.cos(ra_sat_wf),
               -np.cos(dec_sat_wf)*np.sin(ra_sat_wf),
               np.sin(dec_sat_wf)]
  x_ref_sat = [-np.sin(ra_sat_wf),
               np.cos(ra_sat_wf),
               0]
  theta = np.arccos(np.dot(source, z_ref_sat))
  phi = np.mod(np.arctan2(np.dot(source, y_ref_sat), np.dot(source, x_ref_sat)), 2*np.pi)
  return np.rad2deg(theta), np.rad2deg(phi)


def grb_decrapol_worldf2satf(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf):
  """
  Converts dec,ra (declination, right ascension) world coordinates into satellite attitude parameters
  Polarization angle calculation rely on the fact that the polarization angle in is the plane generated by the direction of the source and the dec=0 direction, as it is the case in mamr.py.
  :param dec_grb_wf : declination (except it is 0 at north pole, 90° at equator and 180° at south pole)
  :param ra_grb_wf : Right ascension (0->360°)
  :param dec_sat_wf : satellite dec in world frame
  :param ra_sat_wf : satellite ra in world frame
  :returns: polarization, dec_grb, ra_grb, dec_pol, ra_pol in satellite frame in deg with MEGAlib's RelativeY convention
  """
  dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf = np.deg2rad(dec_grb_wf), np.deg2rad(ra_grb_wf), np.deg2rad(dec_sat_wf), np.deg2rad(ra_sat_wf)
  # source being the direction of the source
  source = [np.sin(dec_grb_wf)*np.cos(ra_grb_wf),
            np.sin(dec_grb_wf)*np.sin(ra_grb_wf),
            np.cos(dec_grb_wf)]
  # Referential for the satellite
  z_ref_sat = [np.sin(dec_sat_wf)*np.cos(ra_sat_wf),
               np.sin(dec_sat_wf)*np.sin(ra_sat_wf),
               np.cos(dec_sat_wf)]
  y_ref_sat = [-np.cos(dec_sat_wf)*np.cos(ra_sat_wf),
               -np.cos(dec_sat_wf)*np.sin(ra_sat_wf),
               np.sin(dec_sat_wf)]
  x_ref_sat = [-np.sin(ra_sat_wf),
               np.cos(ra_sat_wf),
               0]
  theta = np.arccos(np.dot(source, z_ref_sat))
  phi = np.mod(np.arctan2(np.dot(source, y_ref_sat), np.dot(source, x_ref_sat)), 2*np.pi)
  # Polarization
  dec_p, ra_p = np.mod(.5 * np.pi - dec_grb_wf, np.pi), ra_grb_wf + np.pi  # polarization direction in world coordinates (towards north or south pole)
  # pol_direction being the polarization vector in world coordinates
  pol_vec = [np.sin(dec_p) * np.cos(ra_p),
             np.sin(dec_p) * np.sin(ra_p),
             np.cos(dec_p)]
  thetap = np.arccos(np.dot(pol_vec, z_ref_sat))  # polarization direction in satellite coordinates
  phip = np.arctan2(np.dot(pol_vec, y_ref_sat), np.dot(pol_vec, x_ref_sat))  # idem
  pol_angle = np.arccos(np.dot(pol_vec, np.cross(source, y_ref_sat)))
  return np.rad2deg(pol_angle), np.rad2deg(theta), np.rad2deg(phi), np.rad2deg(thetap), np.rad2deg(phip)


def decrasat2world(dec, ra, s):
  """
  Converts dec,ra (declination, right ascension) satellite coordinates into world coordinate
  :param dec: declination (except it is 0 at instrument zenith and 90° at equator)
  :param ra : Right ascension (0->360°)
  :param s: satellite from infos['satellites']
  :param unit: unit in which are given dec and ra, default="deg"
  :returns: theta_world, phi_world in rad
  """
  dec, ra = np.deg2rad(dec), np.deg2rad(ra)
  decsat, rasat = np.deg2rad(s[0]), np.deg2rad(s[1])
  xworld = [-np.sin(rasat), -np.cos(decsat)*np.cos(rasat), np.sin(decsat)*np.cos(rasat)]
  yworld = [np.cos(rasat), -np.cos(decsat)*np.sin(rasat), np.sin(decsat)*np.sin(rasat)]
  zworld = [0, np.sin(decsat), np.cos(decsat)]
  source = [np.sin(dec)*np.cos(ra), np.sin(dec)*np.sin(ra), np.cos(dec)]
  theta = np.arccos(np.dot(source, zworld))
  phi = np.mod(np.arctan2(np.dot(source, yworld), np.dot(source, xworld)), 2*np.pi)
  return np.deg2rad(theta), np.deg2rad(phi)


def orbitalparam2decra(inclination, ohm, omega, nu=0):
  """
  Calculates the declination and right ascention of an object knowing its orbital parameters
  Returned results are in rad and the north direction is at 0° making the equator at 90°
  :param inclination : inclination of the orbit in deg
  :param ohm : longitude/ra of the ascending node in deg
  :param omega :
  :param nu : true anomalie at epoch t0 (both are equivalent there because of circular orbit)  in deg
  """
  # Puts the angle in rad and changes the omega to take into account the true anomaly
  inclination, ohm, omeganu = np.deg2rad(inclination), np.deg2rad(ohm), np.deg2rad(omega + nu)
  # normalized coordinates in the orbital frame:
  xsat = np.cos(omeganu) * np.cos(ohm) - np.sin(omeganu) * np.cos(inclination) * np.sin(ohm)
  ysat = np.cos(omeganu) * np.sin(ohm) + np.sin(omeganu) * np.cos(inclination) * np.cos(ohm)
  zsat = np.sin(inclination)*np.sin(omeganu)
  thetasat = np.arccos(zsat) #rad # taking the arccos makes the dec from 0 to 180° starting at the north pole
  phisat = np.arctan2(ysat, xsat) #rad
  return np.rad2deg(thetasat), np.rad2deg(phisat)


def true_anomaly_calc(time, period):
  """
  Calculates the true anomaly between 0 and 360° for a time time and an orbit of period period
  :param time : time at which the true anomaly is calculated
  :param period : period of the orbit
  Return the true anomaly in deg
  """
  return np.mod(360 * time / period, 360)


def cond_between(val, lim1, lim2):
  """
  Checks if a value is in an interval
  :param val : value to test
  :param lim1 : inferior limit
  :param lim2 : superior limit
  Returns True if the value is in the interval [lim1, lim2]
  """
  return val >= lim1 and val <= lim2


def verif_zone(theta, phi):
  """
  Function to verify wether a coordinate is in the exclusion area or not
  :param theta : latitude (-90 - 90°)
  :param phi : longitude (-180 - 180°)
  Returns True when the theta and phi are in an excusion area
  """
  ### SAA
  if cond_between(phi, -60, -10) and cond_between(theta, -45, -15):
    return True
  ### North pole anomaly
  elif cond_between(phi, -80, 70) and cond_between(theta, 50, 70):
    return True
  elif cond_between(phi, -180, -120) and cond_between(theta, 55, 65):
    return True
  elif cond_between(phi, 140, 180) and cond_between(theta, 60, 65):
    return True
  ### South pole anomaly
  elif cond_between(phi, -125, 20) and cond_between(theta, -78, -58):
    return True
  elif cond_between(phi, 0, 90) and cond_between(theta, -58, -47):
    return True
  else:
    return False


def verif_rad_belts(lat, long, alt):
  """

  """
  files = ["./bkg/exclusion/400km/AE8max_a400km_i90deg.out", "./bkg/exclusion/400km/AE8min_a400km_i90deg.out",
           "./bkg/exclusion/400km/AP8max_a400km_i90deg.out", "./bkg/exclusion/400km/AP8min_a400km_i90deg.out",
           "./bkg/exclusion/500km/AE8max_a500km_i90deg.out", "./bkg/exclusion/500km/AE8min_a500km_i90deg.out",
           "./bkg/exclusion/500km/AP8max_a500km_i90deg.out", "./bkg/exclusion/500km/AP8min_a500km_i90deg.out"]
  for file in files:
    file_alt = int(file.split("km/")[0].split("/")[-1])
    if alt == file_alt:
      if verif_zone_file(lat, long, file):
        return True
  return False


def verif_zone_file(lat, long, file):
  """
  :param theta : latitude (-90 - 90°)
  :param phi : longitude (-180 - 180°)
  """
  with open(file, "r") as f:
    lines = f.read().split("\n")
  exclusions = []
  for line in lines:
    exclusion = []
    if not line.startswith("#"):
      for val in line.split(" "):
        if val != '':
          exclusion.append(float(val))
    if len(exclusion) >= 1:
      exclusions.append(exclusion)
  new_exclusions = []
  for zone in exclusions:
    if len(zone) == 3:
      new_exclusions.append(zone)
    else:
      for ite in range(int(len(zone[1:])/2)):
        new_exclusions.append([zone[0], zone[1+ite*2], zone[2+ite*2]])
  new_exclusions = np.array(new_exclusions)
  exclude_lats = new_exclusions[:, 0]
  # print(exclude_lats)
  # print(exclusions)
  # print(new_exclusions)
  # from time import time
  # init = time()
  lat_index = []
  for ite in range(len(exclude_lats)):
    if lat >= exclude_lats[ite] - 0.5 and lat <= exclude_lats[ite] + 0.5:
      lat_index.append(ite)
  for index in lat_index:
    if cond_between(long, new_exclusions[index, 1], new_exclusions[index, 2]):
      return True
  return False


def random_GRB_dec_ra(dec_min, dec_max, ra_min, ra_max):
  """
  Take a random position for a GRB
  :param dec_min : minimum dec, if south pole -90 in deg
  :param dec_max : maximum dec, if north pole 90 in deg
  :param ra_min : minimum ra in deg
  :param ra_max : maximum ra in deg
  returns dec and ra in radiant. The north pole is set as dec = 0   in deg
  """
  dec_min, dec_max, ra_min, ra_max = np.deg2rad(dec_min), np.deg2rad(dec_max), np.deg2rad(ra_min), np.deg2rad(ra_max)
  dec = np.pi / 2 - np.arcsin(np.sin(dec_min) + np.random.rand() * (np.sin(dec_max) - np.sin(dec_min)))  # rad, 0 at north pole
  ra = ra_min + np.random.rand() * (ra_max - ra_min)  # rad, 0->2pi
  return np.rad2deg(dec), np.rad2deg(ra)

def decra2orbitalparam(thetasat, phisat):
  """
  Calculates the orbital parameters of an object knowing its dec and ra
  Only works for a value of omega set to pi/2
  Returned results are in rad
  :param inclination : inclination of the orbit
  :param ohm : longitude/ra of the ascending node
  :param omega : true anomalie at epoch t0 (both are equivalent there because of circular orbit)
  :param unit: unit in which are given dec and ra, default="deg"
  """
  thetasat, phisat = np.deg2rad(thetasat), np.deg2rad(phisat)
  inclination = np.arcsin(np.cos(thetasat)) #rad
  ohm = np.arctan2(-1, np.tan(phisat)) #rad
  omega = np.pi / 2
  return np.deg2rad(inclination), np.deg2rad(ohm), np.deg2rad(omega)


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


