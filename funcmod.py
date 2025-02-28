import numpy as np
import gzip
from scipy.integrate import quad, simpson, IntegrationWarning
import warnings
warnings.simplefilter("error", RuntimeWarning)

from scipy.stats import skewnorm
from time import time
import os
import subprocess
from apexpy import Apex
import traceback
import tracemalloc

import astropy.units
# Useful constants
keV_to_erg = 1 * astropy.units.keV
keV_to_erg = keV_to_erg.to_value("erg")
Gpc_to_cm = 1 * astropy.units.Gpc
Gpc_to_cm = Gpc_to_cm.to_value("cm")

# TODO USE ASTROPY CONSTANTS
m_elec = 9.1094e-31  # kg
c_light = 2.99792458e+8  # m/s
charge_elem = 1.6021e-19  # C
electron_rest_ener = m_elec * c_light ** 2 / charge_elem / 1000  # electron rest energy in keV
m_earth = 5.9722e24  # kg
R_earth = 6371  # km
G_const = 6.67430e-11  # m3/kg/s2
earth_rot_time = 86164 + 98e-3 + 903e-6 + 697e-9  # s


def printv(message, verbose):
  """
  Print message is verbose is True
  :param message: message to print
  :param verbose: whether or not the message is displayed (True or False)
  """
  if verbose:
    print(message)


def printcom(comment):
  print()
  if type(comment) is list:
    print("=========================================================================================================================")
    for com in comment:
      print(f"= {com}")
    print("=========================================================================================================================")
  elif type(comment) is str:
    print("=========================================================================================================================")
    print(f"= {comment}")
    print("=========================================================================================================================")
  else:
    raise TypeError("The type of the comment should be str or a list of str")


def endtask(taskname, timevar=None):
  if timevar is None:
    print(f"======      {taskname} finished      ======")
  else:
    print(f"======      {taskname} finished  -  processing time : {time() - timevar} seconds      ======")
    print()


def use_scipyquad(func, low_edge, high_edge, func_args=(), x_logscale=False):
  """
  Proceed to the quad integration using scipy quad and handle the integration warning by using simpson integration method from scipy if the warning is raised
  """
  if type(func_args) != tuple:
    raise TypeError(f"Function use_scipyquad takes func_args as tuple only, {type(func_args)} given")
  try:
    warnings.simplefilter("error", category=IntegrationWarning)
    int_spectrum, err = quad(func, low_edge, high_edge, args=func_args)
    return int_spectrum, err
  except IntegrationWarning:
    if x_logscale:
      int_x = np.logspace(np.log10(low_edge), np.log10(high_edge), 10000001)
    else:
      int_x = np.linspace(low_edge, high_edge, 10000001)
    int_y = func(int_x, *func_args)
    int_spectrum = simpson(int_y, x=int_x)
    return int_spectrum, None


def horizon_angle(h, earthradius=R_earth, atmosphereheight=40):
  """
  Calculates the angle between the zenith and the horizon for a LEO satellite
  :param h: altitude of the satellite (km)
  :param earthradius: radius of the Earth (km), default=6371
  :param atmosphereheight: height of the atmosphere (km), default=40
  :returns: horizon angle (deg)
  """
  if h >= atmosphereheight:
    return 90 + np.rad2deg(np.arccos((earthradius + atmosphereheight) / (earthradius + h)))  # deg
  else:
    return 90


def orbital_period_calc(alt):
  """
  Calculates the orbital period of a satellite at a specific altitude
  :param alt : altitude of the orbit
  :returns: duration of orbital period in [seconds]
  """
  return np.sqrt(4 * np.pi**2 / (G_const * m_earth) * ((R_earth + alt) * 1000)**3)


def earth_rotation_offset(time_val):
  """
  Calculates the offset in right ascension due to the earth rotation.
  The earth turns from W to E, so if the satellite orbits from W to E this correction has to be deducted from the
  calculated RA. If not it has to be added.
  :param time_val : time at which the rotational offset is calculated in seconds
  returns a correction angle in deg
  """
  return np.mod(360 * time_val / earth_rot_time, 360)


def treat_ce(event_ener):
  """
  Function to sum the 2 energy deposits given by trafiles for a compton event
  :param event_ener: list of information about the energy of an event given by a trafile
  """
  return np.array([float(event_ener[0]), float(event_ener[4])])


def treat_pe(event_ener):
  """
  Function to sum the 2 energy deposits given by trafiles for a compton event
  :param event_ener: list of information about the energy of an event given by a trafile
  """
  return float(event_ener)


def calculate_polar_angle(ener_sec, ener_tot, ener_sec_err=None, ener_tot_err=None):
  """
  Function to calculate the polar angle using the energy deposits of a compton event
    (Most of the time first interaction and final absorption)
  This function is made so that the cos of the angle is >=-1 as it's not possible to take the arccos of a number <-1.
  By construction of cos_value, the value cannot exceed 1.
  :param ener_sec: Energy of second deposit
  :param ener_tot: Total energy of deposits
  # base unit of mc**2 is Joule, dividing it by charge_elem and 1000 makes it in keV to correspond with E2 and Etot that are in keV
  """
  cos_value = 1 - electron_rest_ener * (1 / ener_sec - 1 / ener_tot)
  cos_value_filtered = np.where(cos_value < -1, -1, np.where(cos_value > 1, 1, cos_value))
  if ener_sec_err is not None and ener_tot_err is not None:
    ener_first_err = ener_tot_err - ener_sec_err
    cos_value_sqared_filtered = np.where(cos_value_filtered**2 == 1, np.nan, cos_value_filtered**2)
    err = electron_rest_ener / np.sqrt(1 - cos_value_sqared_filtered) * np.sqrt(ener_first_err**2 / ener_tot**4 + ((1/ener_sec**2 - 1/ener_tot**2) * ener_sec_err)**2)
  else:
    err = 0
  return np.rad2deg(np.arccos(cos_value_filtered)), np.rad2deg(err)


def inwindow(energy, ergcut):
  """
  Checks whether an energy is in the energy window defined by ergcut
  :param energy: energy to test
  :param ergcut: energy window (Emin, Emax)
  :returns: bool
  """
  return ergcut[0] <= energy <= ergcut[1]


def fname2decra(fname):
  """
  Infers dec and RA from file name with the shape :
    {prefix}_{sourcename}_sat{num_sat}_{num_sim}_{dec_world_frame}_{ra_world_frame}.inc{1/2}.id1.extracted.tra
  :param fname: *.tra or *.tra.gz filename
  :returns: dec, RA, sourcename, number of the sim, number of the sat
  """
  data = fname.split("/")[-1]  # to get rid of the first part of the prefix (and the potential "_" in it)
  data = data.split("_")
  return float(data[4]), float(".".join(data[5].split(".")[:2])), data[1], int(data[3]), int(data[2].split("sat")[1])


def fname2decratime(fname):
  """
  Infers dec and RA from file name with the shape :
    {prefix}_{sourcename}_sat{num_sat}_{num_sim}_{dec_world_frame}_{ra_world_frame}_{burst_time}.inc{1/2}.id1.extracted.tra
  :param fname: *.tra or *.tra.gz filename
  :returns: dec, RA, time at which the burst happened, sourcename, number of the sim, number of the sat
  """
  data = fname.split("/")[-1]  # to get rid of the first part of the prefix (and the potential "_" in it)
  data = data.split("_")
  return float(data[4]), float(data[5]), float(".".join(data[6].split(".")[:2])), data[1], int(data[3]), int(data[2].split("sat")[1])


def save_log(filename, name, num_grb, num_sim, num_sat, status, inc, ohm, omega, alt, random_time, sat_dec_wf, sat_ra_wf, grb_dec_wf, grb_ra_wf, grb_dec_st, grb_ra_sf):
  """
  Saves all the simulation information into a log file.
  May be used to make sure everything works or to make some plots
  :param filename: name of the log file to store information
  :param name: name of the source
  :param num_sim: number of the sime
  :param num_sat: number of the sat
  :param status: status of the simulation - Simulated - Ignored(horizon) - Ignored(off) - Ignored(faint)
  :param inc: inclination of the satellite's orbite
  :param ohm: ra of the ascending node of the satellite's orbite
  :param omega: argument of periapsis of the satellite's orbite
  :param alt: altitude of the satellite's orbite
  :param random_time: random time at which the source is simulated
  :param sat_dec_wf: satellite's dec in world frame
  :param sat_ra_wf: satellite's ra in world frame
  :param grb_dec_wf: source's dec in world frame
  :param grb_ra_wf: source's ra in world frame
  :param grb_dec_st: source's dec in sat frame
  :param grb_ra_sf: source's ra in sat frame
  """
  with open(filename, "a") as f:
    f.write(f"{name} | {num_grb} | {num_sim} | {num_sat} | {status} | {inc} | {ohm} | {omega} | {alt} | {random_time} | {sat_dec_wf} | {sat_ra_wf} | {grb_dec_wf} | {grb_ra_wf} | {grb_dec_st} | {grb_ra_sf}\n")


def extract_lc(fullname):
  """
  Opens a light curve file from a .dat file and returns 2 lists containing time and count
  :param fullname: path + name of the file to save the light curves
  """
  times = []
  counts = []
  with open(fullname, "r") as f:
    lines = f.read().split("\n")[3:-1]
    if float(lines[0].split(" ")[1]) != 0:
      raise ValueError("Error, light curve doesn't start at a time t=0")
    for line in lines:
      data = line.split(" ")
      times.append(data[1])
      counts.append(data[2])
  return np.array(times, dtype=float), np.array(counts, dtype=float)


def pflux_to_mflux_calculator(lc_name):
  """
  Returns the conversion value from pflux to mflux.
  It's based on a 1-second pflux as the Lpeak in the Yonetoku correlation is based on a 1-second timescale.
  """
  times, counts = extract_lc(f"./sources/Light_Curves/{lc_name}")
  delta_time = times[1:]-times[:-1]
  reduced_count = counts[:-1]
  # Mean number of count/second
  mean_count = np.sum(reduced_count) / times[-1]
  if times[-1] <= 1:
    # Case where the T90 <1s, mflux and pflux over 1s are then the same
    pflux_to_mflux = 1
  elif np.min(delta_time) >= 1:
    # No need to rebin, we just re-normalize the counts with the duration of the bin that is >1s
    reduced_count = reduced_count / delta_time
    pflux_to_mflux = mean_count / np.max(reduced_count)
  else:
    # Rebining needed, we define the number of rebins necessary and rebin
    rebining = int(1 / delta_time[0]) + 1
    newcount = []
    bin_ite = 0
    while bin_ite + rebining < len(reduced_count):
      newcount.append(np.sum(reduced_count[bin_ite:bin_ite+rebining]))
      bin_ite += rebining
    newcount.append(np.sum(reduced_count[bin_ite:]))
    pflux_to_mflux = mean_count / np.max(newcount)
  return pflux_to_mflux


def read_grbpar(parfile):
  """
  reads a source's parameter file to get useful information for the analysis
  :param parfile: path/name of the parameter file
  """
  sat_info = []
  geometry, revan_file, mimrec_file, sim_mode, spectra_path, cat_file, source_file, sim_prefix, sttype, n_sim, simtime, position_allowed_sim = None, None, None, None, None, None, None, None, None, None, None, None
  with open(parfile) as f:
    lines = f.read().split("\n")
  for line in lines:
    if line.startswith("@geometry"):
      geometry = line.split(" ")[1]
    elif line.startswith("@revancfgfile"):
      revan_file = line.split(" ")[1]
    elif line.startswith("@mimrecfile"):
      mimrec_file = line.split(" ")[1]
    elif line.startswith("@simmode"):
      sim_mode = line.split(" ")[1]
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
  return geometry, revan_file, mimrec_file, sim_mode, spectra_path, cat_file, source_file, sim_prefix, sttype, n_sim, simtime, position_allowed_sim, sat_info


def read_bkgpar(parfile):
  """
  reads a background parameter file to get useful information for the analysis
  :param parfile: path/name of the parameter file
  """
  geom, revanf, mimrecf, source_base, spectra, simtime, latitudes, altitudes = None, None, None, None, None, None, None, None
  with open(parfile, "r") as f:
    lines = f.read().split("\n")
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
      values = line.split(" ")[1:]
      if values[0].startswith("[") and values[-1].endswith("]"):
        values[0] = values[0][1:]
        values[-1] = values[-1][:-1]
        try:
          latitudes = np.array(list(map(int, values)))
        except ValueError:
          raise ValueError("Problem while reading the background parameter file, latitude list should be int separated by a space and between brackets")
      elif len(values) == 3:
        try:
          latitudes = list(map(int, values))
          latitudes = np.linspace(latitudes[0], latitudes[1], latitudes[2])
        except ValueError:
          raise ValueError("Problem while reading the background parameter file, latitude range should be 3 int separated by a space")
      else:
        raise ValueError("Problem while reading the background parameter file, latitude should be a range or a list of value as described in the header")
  return geom, revanf, mimrecf, source_base, spectra, simtime, latitudes, altitudes


def read_mupar(parfile):
  """
  reads a mu100 parameter file to get useful information for the analysis
  :param parfile: path/name of the parameter file
  """
  geom, revanf, mimrecf, source_base, spectra, bandparam, poltime, unpoltime, decs, ras = None, None, None, None, None, None, None, None, None, None
  with open(parfile, "r") as f:
    lines = f.read().split("\n")
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
    elif line.startswith("@raposition"):
      ras = list(map(int, line.split(" ")[1:]))
  return geom, revanf, mimrecf, source_base, spectra, bandparam, poltime, unpoltime, decs, ras


def readfile(fname):
  """
  Reads a .tra or .tra.gz file and returns the information for an event, delimited by "SE" in the .tra file
  :param fname: str, name of .tra file
  :returns: information on the event
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
  Reads the information of an event given by readfile and returns this information in a list if it's in the energy range
  :param event: str, event in a trafile
  :param ergcut: couple (Emin, Emax) or None, energy range in which events have to be to be processed, default=None(=no selection)
  :returns:
    list of 3-uple of float if this is a single event (energy, time, position)
    list of 5-uple of float if this is a compton event (first deposit, total energy, time, 1st position, 2nd position)
  """
  lines = event.split("\n")[1:-1]
  # Treating compton events
  if lines[0] == "ET CO":
    second_ener = float(lines[7].split(" ")[1])
    total_ener = second_ener + float(lines[7].split(" ")[5])
    second_ener_err = float(lines[7].split(" ")[2])
    total_ener_err = second_ener_err + float(lines[7].split(" ")[6])
    if ergcut is None:
      time_interaction = float(lines[2].split(" ")[1])
      first_pos = np.array([float(lines[8].split(" ")[11]), float(lines[8].split(" ")[12]), float(lines[8].split(" ")[13])])
      second_pos = np.array([float(lines[8].split(" ")[1]), float(lines[8].split(" ")[2]), float(lines[8].split(" ")[3])])
      first_pos_err = np.array([float(lines[8].split(" ")[16]), float(lines[8].split(" ")[17]), float(lines[8].split(" ")[18])])
      second_pos_err = np.array([float(lines[8].split(" ")[6]), float(lines[8].split(" ")[7]), float(lines[8].split(" ")[8])])
      return [second_ener, total_ener, time_interaction, first_pos, second_pos, second_ener_err, total_ener_err, first_pos_err, second_pos_err]
    else:
      if inwindow(total_ener, ergcut):
        time_interaction = float(lines[2].split(" ")[1])
        first_pos = np.array([float(lines[8].split(" ")[11]), float(lines[8].split(" ")[12]), float(lines[8].split(" ")[13])])
        second_pos = np.array([float(lines[8].split(" ")[1]), float(lines[8].split(" ")[2]), float(lines[8].split(" ")[3])])
        first_pos_err = np.array([float(lines[8].split(" ")[16]), float(lines[8].split(" ")[17]), float(lines[8].split(" ")[18])])
        second_pos_err = np.array([float(lines[8].split(" ")[6]), float(lines[8].split(" ")[7]), float(lines[8].split(" ")[8])])
        return [second_ener, total_ener, time_interaction, first_pos, second_pos, second_ener_err, total_ener_err, first_pos_err, second_pos_err]
      else:
        return [None]
  # Treating single events
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


def save_value(file, value):
  if value is None:
    file.write(f"None\n")
  else:
    if type(value) is np.ndarray or type(value) is list:
      if len(value) == 0:
        file.write(f"\n")
      else:
        for ite in range(len(value) - 1):
          file.write(f"{value[ite]}|")
        file.write(f"{value[-1]}\n")
    elif type(value) in [int, float, np.int64, np.int32, np.float32, np.float64]:
    # elif type(value) is int or type(value) is float or type(value) is np.float64 or type(value) is np.float32:
      file.write(f"{value}\n")
    else:
      raise TypeError(f"Uncorrect type for value saved : {value}, type {type(value)}")


def save_grb_data(data_file, filename, sat_info_list, bkg_data, mu_data, ergcut, armcut, geometry, force=False):
  # tracemalloc.start()  # Start memory monitoring
  #
  # # get memory statistics
  # current, peak = tracemalloc.get_traced_memory()
  # print("\nInit")
  # print(f"Current memory use : {current / 1024:.2f} Ko")
  # print(f"Peak use : {peak / 1024:.2f} Ko")

  array_dtype = np.float32
  sim_dir, fname = filename.split(f"/extracted-{ergcut[0]}-{ergcut[1]}/")

  file_exist = os.path.exists(filename)
  if file_exist and not force:
    print("Checking for already extracted files", end="\r")
  else:
    if file_exist:
      print("Extracted file exists, re-writing is forced", end="\r")
    else:
      print("Extracted files does not exist : Extraction in progress", end="\r")

    #################################################################################################################
    #        Extracting position information from data_file name and the sat_info, plus the background and mu100 information
    #################################################################################################################
    dec_world_frame, ra_world_frame, burst_time, source_name, num_sim, num_sat = fname2decratime(data_file)
    # Error on estimating the GRB direction (obtained with Alexey's estimations)
    dec_wf_error, ra_wf_error = 1, 1
    sat_info = sat_info_list[num_sat]
    # sat_dec_wf, sat_ra_wf = sat_info_2_decra(sat_info, burst_time)
    if sat_info is not None:
      # # get memory statistics
      # current, peak = tracemalloc.get_traced_memory()
      # print("\nBefore affecting")
      # print(f"Current memory use : {current / 1024:.2f} Ko")
      # print(f"Peak use : {peak / 1024:.2f} Ko")
      sat_dec_wf, sat_ra_wf, sat_alt, compton_b_rate, single_b_rate, b_idx = affect_bkg(sat_info, burst_time, bkg_data)
      # # get memory statistics
      # current, peak = tracemalloc.get_traced_memory()
      # print("\nAfter affecting")
      # print(f"Current memory use : {current / 1024:.2f} Ko")
      # print(f"Peak use : {peak / 1024:.2f} Ko")

    else:
      raise ValueError("Satellite information given is None. Please give satellite information for the analyse to work.")
    # Error on estimating the satellite pointing direction (takes into account the pointing itself and the effect of the satellite not being exactly in the right position - very minor effect)
    dec_sat_wf_error, ra_sat_wf_error = 0.5, 0.5
    expected_pa, grb_dec_sat_frame, grb_ra_sat_frame, grb_dec_sf_err, grb_ra_sf_err = grb_decrapol_worldf2satf(dec_world_frame, ra_world_frame, sat_dec_wf, sat_ra_wf, dec_grb_wf_err=dec_wf_error,
                                                                                                               ra_grb_wf_err=ra_wf_error, dec_sat_wf_err=dec_sat_wf_error, ra_sat_wf_err=ra_sat_wf_error)[:5]

    mu100_ref, mu100_err_ref, s_eff_compton_ref, s_eff_single_ref = closest_mufile(grb_dec_sat_frame, grb_ra_sat_frame, mu_data)

    # # get memory statistics
    # current, peak = tracemalloc.get_traced_memory()
    # print("\nBefore file reading")
    # print(f"Current memory use : {current / 1024:.2f} Ko")
    # print(f"Peak use : {peak / 1024:.2f} Ko")

    #################################################################################################################
    #        Readding file and saving values
    #################################################################################################################
    # Extracting the data from first file
    data_pol = readfile(data_file)
    compton_second = []
    compton_ener = []
    compton_time = []
    compton_firstpos = []
    compton_secpos = []
    single_ener = []
    single_time = []
    single_pos = []
    compton_second_err = []
    compton_ener_err = []
    compton_firstpos_err = []
    compton_secpos_err = []
    for event in data_pol:
      reading = readevt(event, ergcut)
      if len(reading) == 9:
        compton_second.append(reading[0])
        compton_ener.append(reading[1])
        compton_time.append(reading[2])
        compton_firstpos.append(reading[3])
        compton_secpos.append(reading[4])
        compton_second_err.append(reading[5])
        compton_ener_err.append(reading[6])
        compton_firstpos_err.append(reading[7])
        compton_secpos_err.append(reading[8])
      elif len(reading) == 3:
        single_ener.append(reading[0])
        single_time.append(reading[1])
        single_pos.append(reading[2])
    # Free the variable
    del data_pol

    compton_ener = np.array(compton_ener, dtype=array_dtype)
    compton_second = np.array(compton_second, dtype=array_dtype)
    single_ener = np.array(single_ener, dtype=array_dtype)
    compton_firstpos = np.array(compton_firstpos, dtype=array_dtype)
    compton_secpos = np.array(compton_secpos, dtype=array_dtype)
    single_pos = np.array(single_pos, dtype=array_dtype)
    compton_time = np.array(compton_time, dtype=array_dtype)
    single_time = np.array(single_time, dtype=array_dtype)
    compton_ener_err = np.array(compton_ener_err, dtype=array_dtype)
    compton_second_err = np.array(compton_second_err, dtype=array_dtype)
    compton_firstpos_err = np.array(compton_firstpos_err, dtype=array_dtype)
    compton_secpos_err = np.array(compton_secpos_err, dtype=array_dtype)
    scat_vec_err = np.sqrt(compton_secpos_err**2 + compton_firstpos_err**2)

    # ok ! grb_decrapol_worldf2satf tested and linked with all the other parts
    # ok ! calculate_polar_angle    tested     linked
    # ok ! angle                    tested     linked

    # ok ! Use dec ra errors
    # tbd if needed ! Use polar_from_energy error
    # ! Use pol errors

    # ! Erreurs extraites mais pas à enregistrer ? juste traiter et enregistrer les erreurs une fois l'erreur sur les angles obtenue (pol par exemple)
    # ! Estimer l'importance de l'erreur de polar from direction, pas convaincu que c'est utile
    # ! Erreurs pas prise en compte pour les backgrounds (energie et position) puisque pas d'intéret dans la détermination de la direction de la source et d'une quelconque polarisation (considéré unpolarized)
    # ! Recuperer erreur des fichiers avec readevt, erreur seulement sur les ev compton, ev single ne presentent pas d erreur sur l energie et la position donnee
    # ! erreur avec polar from energy
    # ! erreur probablement assez complexe avec angle
    # ! Semble peu utile de faire le tri sur l ARM et l energie en prenant en compte les erreurs, voir l ampleur de l erreur mais probablement trop faible pour vraiment etre interessant
    # ! surtout que le tri sur l energie se fait lors de la recuperation des valeurs donc encore plus problematique
    #
    # ! Ensuite retravailler les mu100
    # ! La valeur de mu_100_err va changer, mais en plus il faudrait prendre en compte l erreur dans le calcul de la MDP et dans le mu100 de la constellation

    #################################################################################################################
    #                     Filling the fields
    #################################################################################################################
    # Calculating the polar angle with energy values and compton azim and polar scattering angles from the kinematics
    # polar and position angle stored in deg
    polar_from_energy, polar_from_energy_err = calculate_polar_angle(compton_second, compton_ener, ener_sec_err=compton_second_err, ener_tot_err=compton_ener_err)
    pol, polar_from_position, pol_err = angle(compton_secpos - compton_firstpos, grb_dec_sat_frame, grb_ra_sat_frame, source_name, num_sim, num_sat, scatter_vector_err=scat_vec_err, grb_dec_sf_err=grb_dec_sf_err, grb_ra_sf_err=grb_ra_sf_err)

    # Calculating the arm and extracting the indexes of correct arm events (arm in deg)
    arm_pol = np.array(polar_from_position - polar_from_energy, dtype=array_dtype)
    accepted_arm_pol = np.where(np.abs(arm_pol) <= armcut, True, False)
    # Restriction of the values according to arm cut
    compton_ener = compton_ener[accepted_arm_pol]
    compton_second = compton_second[accepted_arm_pol]
    compton_firstpos = compton_firstpos[accepted_arm_pol]
    compton_secpos = compton_secpos[accepted_arm_pol]
    compton_time = compton_time[accepted_arm_pol]
    polar_from_energy = polar_from_energy[accepted_arm_pol]
    polar_from_position = np.array(polar_from_position[accepted_arm_pol], dtype=array_dtype)
    pol = np.array(pol[accepted_arm_pol], dtype=array_dtype)

    # # get memory statistics
    # current, peak = tracemalloc.get_traced_memory()
    # print("\nAfter file reading / before detector search")
    # print(f"Current memory use : {current / 1024:.2f} Ko")
    # print(f"Peak use : {peak / 1024:.2f} Ko")

    #################################################################################################################
    #     Finding the detector of interaction for each event
    #################################################################################################################
    compton_first_detector, compton_sec_detector, single_detector = find_detector(compton_firstpos, compton_secpos, single_pos, geometry)

    # # get memory statistics
    # current, peak = tracemalloc.get_traced_memory()
    # print("\nAfter detector search")
    # print(f"Current memory use : {current / 1024:.2f} Ko")
    # print(f"Peak use : {peak / 1024:.2f} Ko")

    # Saving information
    with open(filename, "w") as f:
      f.write(f"Extracted file of simfile : {data_file} with ergcut : {ergcut[0]}-{ergcut[1]} and armcut : {armcut}\n")
      f.write(f"{dec_world_frame}|{ra_world_frame}|{burst_time}|{source_name}|{num_sim}\n")
      # Specific to satellite
      save_value(f, b_idx)
      save_value(f, sat_dec_wf)
      save_value(f, sat_ra_wf)
      save_value(f, sat_alt)
      save_value(f, num_sat)
      save_value(f, compton_b_rate)
      save_value(f, single_b_rate)
      # Information from mu files
      save_value(f, mu100_ref)
      save_value(f, mu100_err_ref)
      save_value(f, s_eff_compton_ref)
      save_value(f, s_eff_single_ref)
      # GRB position and polarisation
      save_value(f, grb_dec_sat_frame)
      save_value(f, grb_ra_sat_frame)
      save_value(f, expected_pa)
      # Value arrays
      save_value(f, compton_ener)
      save_value(f, compton_second)
      save_value(f, single_ener)
      save_value(f, compton_time)
      save_value(f, single_time)
      save_value(f, pol)
      save_value(f, polar_from_position)
      save_value(f, polar_from_energy)
      save_value(f, arm_pol)
      save_value(f, compton_first_detector)
      save_value(f, compton_sec_detector)
      save_value(f, single_detector)

  #   # get memory statistics
  #   current, peak = tracemalloc.get_traced_memory()
  #   print("\nAfter saving")
  #   print(f"Current memory use : {current / 1024:.2f} Ko")
  #   print(f"Peak use : {peak / 1024:.2f} Ko")
  #   tracemalloc.stop()
  # stop


def angle(scatter_vector, grb_dec_sf, grb_ra_sf, source_name, num_sim, num_sat, scatter_vector_err=None, grb_dec_sf_err=None, grb_ra_sf_err=None):  # TODO : limits on variables
  """
  Calculate the azimuthal and polar Compton angles : Transforms the compton scattered gamma-ray vector
  (initialy in sat frame) into a new referential corresponding to the direction of the source.
    From [xsat, ysat, zsat] to [xsource, ysource, zsource]
    In that new frame scatter_vector[0] and scatter_vector[1] are the coordinates of the vector in the plan orthogonal
    to the source direction.
  xsource is the vector in the plane created by the zenith (zsat axis) of the instrument and the source direction zsource
  ysource is constructed by taking it orthogonal to xsource and zsource. This also makes it in the plane of the detector.
    (also xsource is in the plane containing the zworld, the source direction, and the axis ysat of the detector)
  The way the azimuthal scattering angle is calculated imply that the polarization vector is colinear with x
  :param scatter_vector:  array of 3-uple, Compton scattered gamma-ray vector
  :param grb_dec_sf:      float,  source polar angle seen by satellite [deg]
  :param grb_ra_sf:       float,  source azimuthal angle seen by satellite [deg]
  :param source_name:     name of the source
  :param num_sim:         number of the simulation
  :param num_sat:         number of the satellite
  :returns:     2 array, polar and azimuthal compton scattering angles [deg]
  """
  if len(scatter_vector) == 0:
    print(f"There is no compton event detected for source {source_name}, simulation {num_sim} and satellite {num_sat}")
    return np.array([]), np.array([]), np.array([])
  grb_dec_sf, grb_ra_sf = np.deg2rad(grb_dec_sf), np.deg2rad(grb_ra_sf)

  # Changing the direction of the vector to be in adequation with MEGAlib's functionning
  # Megalib changes the direction of the vector in its source code so the same is did here for some coherence
  MEGAlib_direction = True
  if MEGAlib_direction:
    scatter_vector = -scatter_vector
  init_vector = scatter_vector

  # Pluging in some MEGAlib magic :
  # Making the norm of the vector 1 and reshaping it so that numpy operations are done properly
  scatter_vector = scatter_vector / np.reshape(np.linalg.norm(scatter_vector, axis=1), (len(scatter_vector), 1))
  # Rotation matrix around Z with an angle -phi
  mat1 = np.array([[np.cos(-grb_ra_sf), - np.sin(-grb_ra_sf), 0],
                   [np.sin(-grb_ra_sf), np.cos(-grb_ra_sf), 0],
                   [0, 0, 1]])
  # Rotation matrix around Y with an angle -theta
  mat2 = np.array([[np.cos(-grb_dec_sf), 0, np.sin(-grb_dec_sf)],
                   [0, 1, 0],
                   [- np.sin(-grb_dec_sf), 0, np.cos(-grb_dec_sf)]])
  # Using matrix products to combine the matrix instead of doing it vector by vector
  # Rotations to put the scatter_vector into a frame where z in the source direction and x the polarization vector direction.
  scatter_vector = np.matmul(scatter_vector, np.transpose(mat1))
  scatter_vector = np.matmul(scatter_vector, np.transpose(mat2))
  try:
    temp_polar = np.arccos(scatter_vector[:, 2])
  except:
    for value in scatter_vector[:, 2]:
      if np.abs(value) >=1 :
        print(value)
    print(len(scatter_vector[:, 2]))
    stop
  if MEGAlib_direction:
    polar = 180 - np.rad2deg(temp_polar)
  else:
    polar = np.rad2deg(temp_polar)
  # Figure out a good arctan
  azim = np.rad2deg(np.arctan2(scatter_vector[:, 1], scatter_vector[:, 0]))

  if scatter_vector_err is not None and grb_dec_sf_err is not None and grb_ra_sf_err is not None:
    grb_dec_sf_err, grb_ra_sf_err = np.deg2rad(grb_dec_sf_err), np.deg2rad(grb_ra_sf_err)

    def angle_u(init_scat_vector, pg):
      fx, fy = init_scat_vector[:, 0], init_scat_vector[:, 1]
      return fx * np.sin(-pg) + fy * np.cos(-pg)

    def angle_v(init_scat_vector, tg, pg):
      fx, fy, fz = init_scat_vector[:, 0], init_scat_vector[:, 1], init_scat_vector[:, 2]
      return fx * np.cos(-pg) * np.cos(-tg) - fy * np.sin(-pg) * np.cos(-tg) + fz * np.sin(-tg)

    def angleder_u5(init_scat_vector, pg):
      fx, fy = init_scat_vector[:, 0], init_scat_vector[:, 1]
      return - fx * np.cos(-pg) + fy * np.sin(-pg)

    def angleder_v4(init_scat_vector, tg, pg):
      fx, fy, fz = init_scat_vector[:, 0], init_scat_vector[:, 1], init_scat_vector[:, 2]
      return fx * np.cos(-pg) * np.sin(-tg) - fy * np.sin(-pg) * np.sin(-tg) - fz * np.cos(-tg)

    def angleder_v5(init_scat_vector, tg, pg):
      fx, fy = init_scat_vector[:, 0], init_scat_vector[:, 1]
      return fx * np.sin(-pg) * np.cos(-tg) + fy * np.cos(-pg) * np.cos(-tg)

    val_u = angle_u(init_vector, grb_ra_sf)
    val_v = angle_v(init_vector, grb_dec_sf, grb_ra_sf)

    azim_err = np.where(val_v == 0, np.nan, (np.sqrt(((np.sin(-grb_ra_sf) * val_v - np.cos(-grb_ra_sf) * np.cos(-grb_dec_sf) * val_u) * scatter_vector_err[:, 0])**2 +
                       ((np.cos(-grb_ra_sf) * val_v + np.sin(-grb_ra_sf) * np.cos(-grb_dec_sf) * val_u) * scatter_vector_err[:, 1])**2 +
                       (np.sin(-grb_dec_sf) * val_u * scatter_vector_err[:, 2])**2 +
                       (angleder_v4(init_vector, grb_dec_sf, grb_ra_sf) * val_u * grb_dec_sf_err)**2 +
                       ((angleder_u5(init_vector, grb_ra_sf) * val_v - angleder_v5(init_vector, grb_dec_sf, grb_ra_sf) * val_u) * grb_ra_sf_err)**2) /
                (val_v ** 2 * (1 + (val_u / val_v) ** 2))))
  else:
    azim_err = 0
  return azim, polar, azim_err


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
  Create bins for polarigrams
  :param bin_mode: How the bins are created
    fixed : 21 equal bins between -180 and 180
    limited : Each bin has at list 9 events TODO
    optimized : bins are created so that the fit is optimized TODO + find how to have this optimization
  :returns:   array with the bins' values
  """
  bins = ""
  if bin_mode == "fixed":
    bins = np.linspace(-180, 180, 21, dtype=np.float32)
  elif bin_mode == "limited":
    bins = []
  elif bin_mode == "optimized":
    bins = []
  return bins


def err_calculation(polhist, unpolhist, binwidth, polhist_err, unpolhist_err):
  """
  Calculation of the errorbar of the corrected polarigram according to megalib's way
  Obtained by propagating the error of pol, unpol and mean with equation pol/unpol*mean
  :param polhist:      list,   bins for the polarized polarigram
  :param unpolhist:    list,   bins for the unpolarized polarigram
  :param binwidth: list,   bin widths
  """
  nbins = len(polhist)
  mean_unpol = np.mean(unpolhist)

  # uncertainty = (pol/unpol**2*mean_unpol*np.sqrt(unpol))**2 + (mean_unpol/unpol*np.sqrt(pol))**2
  # for ite_j in range(nbins):
  #   uncertainty += (pol / unpol / nbins * np.sqrt(unpol[ite_j])) ** 2
  uncertainty = (mean_unpol / unpolhist) ** 2 * (polhist + polhist_err) + (polhist * mean_unpol / unpolhist**2)**2 * (unpolhist + unpolhist_err) + (polhist / unpolhist) ** 2 * mean_unpol / nbins
  error = np.sqrt(uncertainty)
  return error/binwidth


def make_error_histogram(array, error_array, bins, hardlim=(True, False)):
  """
  Hardlim is used to keep values that would be out of the physical range.
  For instance, hardlim on 0 for mdp, because mdp > 0.
  In that situation if an errored value < 0 is found (to be put in bin 0 that was initialy in bin 1) then the value is not taken out of bin 1 into bin 0 as it's not physical to find it there
  Returns the sup error and inf error on bins
  """
  array_sup = array + error_array
  array_inf = array - error_array

  ind = np.digitize(array, bins)
  indsup = np.digitize(array_sup, bins)
  indinf = np.digitize(array_inf, bins)

  hist = np.histogram(array, bins=bins)[0]
  nbins = len(hist)
  hist_inf = np.zeros((nbins + 2))
  hist_sup = np.zeros((nbins + 2))
  for ite in range(len(array)):
    if ind[ite] - indsup[ite] != 0:
      if not (hardlim[1] and indsup[ite] == nbins and ind[ite] == nbins-1):
        hist_sup[ind[ite] + 1:indsup[ite] + 1] += 1
        hist_sup[ind[ite]] -= 1
      # else:
      #   print("Hardlim on final bin, value not taken away from the bin")
    if ind[ite] - indinf[ite] != 0:
      if not (hardlim[0] and indinf[ite] == 0 and ind[ite] == 1):
        hist_inf[indinf[ite]:ind[ite]] += 1
        hist_inf[ind[ite]] -= 1
      # else:
      #   print("Hardlim on initial bin, value not taken away from the bin")
  hist_inf = hist_inf[1:-1]
  hist_sup = hist_sup[1:-1]

  err_sup = np.max(np.vstack([hist_sup + hist_inf, hist_sup, hist_inf, np.zeros(len(hist_inf))]), axis=0)
  err_inf = np.min(np.vstack([hist_sup + hist_inf, hist_sup, hist_inf, np.zeros(len(hist_inf))]), axis=0)

  return err_inf, err_sup


def pol_unpol_hist_err(pol, unpol, pol_err, unpol_err, bins):
  """
  The method for estimated the error in each bin is not exact but as there is a overestimation of the error value by taking bin - maxerr <= bin <= bin + maxerr with maxerr = max(abs(errinf), abs(errsup))
  intead of bin - errinf <= bin <= bin + errsup

  """
  polp = np.where(pol + pol_err > 180, pol + pol_err - 360, pol + pol_err)
  polm = np.where(pol - pol_err < -180, 360 - (pol - pol_err), pol - pol_err)
  hist_pol = np.histogram(pol, bins)[0]

  hist_polp = np.histogram(polp, bins)[0] - hist_pol
  hist_polm = np.histogram(polm, bins)[0] - hist_pol

  histpolerr = np.where(hist_polp * hist_polm >= 0, np.abs(hist_polp + hist_polm), np.where(np.abs(hist_polp) > np.abs(hist_polm), np.abs(hist_polp), np.abs(hist_polm)))

  unpolp = np.where(unpol + unpol_err > 180, unpol + unpol_err - 360, unpol + unpol_err)
  unpolm = np.where(unpol - unpol_err < -180, 360 - (unpol - unpol_err), unpol - unpol_err)
  hist_unpol = np.histogram(unpol, bins)[0]

  hist_unpolp = np.histogram(unpolp, bins)[0] - hist_unpol
  hist_unpolm = np.histogram(unpolm, bins)[0] - hist_unpol

  histunpolerr = np.where(hist_unpolp * hist_unpolm >= 0, np.abs(hist_unpolp + hist_unpolm), np.where(np.abs(hist_unpolp) > np.abs(hist_unpolm), np.abs(hist_unpolp), np.abs(hist_unpolm)))
  return histpolerr, histunpolerr


def rescale_cr_to_GBM_pf(cr, GBM_mean_flux, GBM_peak_flux):
  """
  Rescales the count rate for a simulation made using a mean flux for the source to obtain an estimation of the count
  rate that should be obtained during the peak of the burst
  :param cr: count rate for a simulation of a GRB with a mean flux
  :param GBM_mean_flux: mean flux given by GBM for a given GRB
  :param GBM_peak_flux: peak flux given by GBM for this same GRB
  :returns: float, count rate at peak
  """
  flux_ratio = GBM_peak_flux / GBM_mean_flux
  return cr * flux_ratio


def calc_snr(S, B, C=0):
  """
  Calculates the signal to noise ratio of a GRB in a time bin. Returns 0 if the value is negative.
  :param S: number of counts in the source (background not included)
  :param B: expected number of background counts
  :param C: minimum number of counts in the source to consider the detection
  :returns: SNR (as defined in Sarah Antier's PhD thesis)
  """
  try:
    S, B = np.where(S <= 0, 0, S), np.where(S <= 0, 1, B)
    snr = S / np.sqrt(B + C)
    snr_err = np.sqrt(S/(B + C) + S**2 * B/(4*(B+C)**3))
  except RuntimeWarning:
    print("S", S)
    print("B", B)
    print("C", C)
    print("snr", S / np.sqrt(B + C))
    print("snr_err", np.sqrt(S/(B + C) + S**2 * B/(4*(B+C)**3)))
  return snr, snr_err


def calc_mdp(S, B, mu100, nsigma=4.29, mu100_err=None):
  """
  Calculates the minimum detectable polarization for a burst
  Error is calculated with error propagation
  :param S: number of expected counts from the burst
  :param B: number of expected counts from the background
  :param mu100: modulation factor
  :param nsigma: significance of the result in number of sigmas, default=4.29 for 99% CL
  """
  if S == 0:
    mdp = np.inf
    mdp_err = 0
  else:
    if mu100_err is not None:
      mdp_err = np.sqrt((nsigma / (mu100 * S**2) * (S/2 + B))**2 * S + (nsigma / (2 * mu100 * S))**2 * B + (nsigma * (S + B) * mu100_err / (S * mu100**2))**2) / np.sqrt(S+B)
    else:
      mdp_err = 0
    mdp = nsigma * np.sqrt(S + B) / (mu100 * S)

  return mdp, mdp_err


def closest_bkg_info(sat_dec, sat_ra, sat_alt, bkg_list):  # TODO : limits on variables
  """
  Find the closest bkg file for a satellite (in terms of latitude, may be updated for longitude too)
  Returns the count rate of this bkg file
  Warning : for now, only takes into account the dec of backgrounds, can be updated but the way the error is calculated
  may not be optimal as the surface of the sphere (polar coordinates) is not a plan.
  :param sat_dec: declination of the satellite [deg] [0 - 180]
  :param sat_ra: right ascension of the satellite [deg] [0 - 360]
    :param sat_alt: altitude of the satellite [km]

  :param bkg_list: list of all the background files
  :returns: compton and single event count rates of the closest background file
  """
  if len(bkg_list) == 0:
    return 0.000001
  else:
    bkg_selec = []
    for bkg in bkg_list:
      if bkg.alt == sat_alt:
        bkg_selec.append(bkg)
    if bkg_selec == []:
      raise FileNotFoundError("No background file were loaded for the given altitude.")
    dec_error = np.array([(bkg.dec - sat_dec) ** 2 for bkg in bkg_selec])
    # ra_error = np.array([(bkg.ra - sat_ra) ** 2 for bkg in bkg_selec])
    ra_error = np.array([0 for bkg in bkg_selec])
    total_error = np.sqrt(dec_error + ra_error)
    index = np.argmin(total_error)
    return [bkg_selec[index].compton_cr, bkg_selec[index].single_cr, index]


def geo_to_mag(dec_wf, ra_wf, altitude):  # TODO : limits on variables
  """
  Converts the geographic declination and right ascension into geomagnetic declination and right ascension
  :param dec_wf: geographic declination in world frame [0 to 180°]
  :param ra_wf: geographic right ascension in world frame [0 to 360°]
  :param altitude: altitude of the point
  :returns: geomagnetic declination and right ascension
  """
  apex15 = Apex(date=2025)
  mag_lat, mag_lon = apex15.convert(90 - dec_wf, ra_wf, 'geo', 'apex', height=altitude)
  return 90 - mag_lat, mag_lon


def affect_bkg(info_sat, burst_time, bkg_list):
  """
  Uses orbital parameters of a satellite to obtain its dec and ra in world frame and to get the expected count rates
  for compton and single events at this position
  dec is calculated so that it is from 0 to 180°
  ra is calculated so that it is from 0 to 360°
  :param info_sat: information about the satellite orbit
  :param burst_time: time at which the burst occured
  :param bkg_list: list of the background files to extract the correct count rates
  :returns  dec_sat_world_frame, ra_sat_world_frame, compton_cr, single_cr
  """
  orbital_period = orbital_period_calc(info_sat[3])
  earth_ra_offset = earth_rotation_offset(burst_time)
  true_anomaly = true_anomaly_calc(burst_time, orbital_period)
  dec_sat_world_frame, ra_sat_world_frame = orbitalparam2decra(info_sat[0], info_sat[1], info_sat[2], nu=true_anomaly)
  ra_sat_world_frame = np.mod(ra_sat_world_frame - earth_ra_offset, 360)
  mag_dec_sat_world_frame, mag_ra_sat_world_frame = geo_to_mag(dec_sat_world_frame, ra_sat_world_frame, info_sat[3])
  bkg_info = closest_bkg_info(mag_dec_sat_world_frame, mag_ra_sat_world_frame, info_sat[3], bkg_list)
  return dec_sat_world_frame, ra_sat_world_frame, info_sat[3], bkg_info[0], bkg_info[1], bkg_info[2]


def closest_mufile(grb_dec_sf, grb_ra_sf, mu_list):  # TODO : limits on variables
  """
  Find the mu100 file closest to a certain direction of detection
  Warning : for now, only takes into account the dec of backgrounds, can be updated but the way the error is calculated
  may not be optimal as the surface of the sphere (polar coordinates) is not a plan.
  :param grb_dec_sf:  declination of the source in satellite frame [deg] [0 - 180]
  :param grb_ra_sf:   right ascension of the source in satellite frame [deg] [0 - 360]
  :param mu_list:     list of all the mu100 files
  :returns:   mu100, mu100_err, s_eff_compton, s_eff_single
  """
  if len(mu_list) == 0:
    return 0.000001, 0.000001, 0.000001, 0.000001
  else:
    dec_error = np.array([(mu.dec - grb_dec_sf) ** 2 for mu in mu_list])
    ra_error = np.array([(mu.ra - grb_ra_sf) ** 2 for mu in mu_list])
    total_error = np.sqrt(dec_error + ra_error)
    index = np.argmin(total_error)
    return mu_list[index].mu100, mu_list[index].mu100_err, mu_list[index].s_eff_compton, mu_list[index].s_eff_single


def det_counter(det_idx_array):
  return np.array([[np.count_nonzero(det_idx_array == 1), np.count_nonzero(det_idx_array == 2), np.count_nonzero(det_idx_array == 3), np.count_nonzero(det_idx_array == 4), np.count_nonzero(det_idx_array == 5)],
                   [np.count_nonzero(det_idx_array == 6), np.count_nonzero(det_idx_array == 7), np.count_nonzero(det_idx_array == 8), np.count_nonzero(det_idx_array == 9), np.count_nonzero(det_idx_array == 10)],
                   [np.count_nonzero(det_idx_array == 11), np.count_nonzero(det_idx_array == 12), np.count_nonzero(det_idx_array == 13), np.count_nonzero(det_idx_array == 14), np.count_nonzero(det_idx_array == 15)],
                   [np.count_nonzero(det_idx_array == 16), np.count_nonzero(det_idx_array == 17), np.count_nonzero(det_idx_array == 18), np.count_nonzero(det_idx_array == 19), np.count_nonzero(det_idx_array == 20)]])


def det_counter_by_type(det_idx_array):
  return np.count_nonzero(np.isin(det_idx_array, [1, 2, 6, 7, 11, 12, 16, 17])), np.count_nonzero(np.isin(det_idx_array, [3, 4, 8, 9, 13, 14, 18, 19])), np.count_nonzero(np.isin(det_idx_array, [5, 10, 15, 20])), len(det_idx_array)


def calc_flux_gbm(catalog, index, ergcut, cat_is_df=False):
  """
  Calculates the fluence per unit time of a given source using an energy cut and its spectrum
  :param catalog: GBM catalog containing sources' information
  :param index: index of the source in the catalog
  :param ergcut: energy window over which the fluence is calculated
  :returns: the number of photons per cm² for a given energy range, averaged over the duration of the sim : ncount/cm²/s
  """
  if cat_is_df:
    used_df = catalog
  else:
    used_df = catalog.df
  model = used_df.flnc_best_fitting_model[index]
  if model == "flnc_band":
    func = band
    func_args = (used_df.flnc_band_ampl[index], used_df.flnc_band_alpha[index], used_df.flnc_band_beta[index], used_df.flnc_band_epeak[index])
  elif model == "flnc_comp":
    func = comp
    func_args = (used_df.flnc_comp_ampl[index], used_df.flnc_comp_index[index], used_df.flnc_comp_epeak[index], used_df.flnc_comp_pivot[index])
  elif model == "flnc_sbpl":
    func = sbpl
    func_args = (used_df.flnc_sbpl_ampl[index], used_df.flnc_sbpl_indx1[index], used_df.flnc_sbpl_indx2[index], used_df.flnc_sbpl_brken[index], used_df.flnc_sbpl_brksc[index], used_df.flnc_sbpl_pivot[index])
  elif model == "flnc_plaw":
    func = plaw
    func_args = (used_df.flnc_plaw_ampl[index], used_df.flnc_plaw_index[index], used_df.flnc_plaw_pivot[index])
  else:
    print("Could not find best fit model for {} (indicated {}). Aborting this GRB.".format(used_df.name[index], model))
    return
  return use_scipyquad(func, ergcut[0], ergcut[1], func_args=func_args, x_logscale=True)[0]


def duty_calc(inclination):  # TODO : limits on variables CHANGE IT WITH THE NEW
  """
  Function to estimate a duty cycle according to inclination
  :param inclination: inclination of the orbit
  :returns: the duty cycle
  """
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


# def eff_area_func(dec_wf, ra_wf, dec_sat_wf, ra_sat_wf, sat_alt, mu100_list):  # TODO : limits on variables
def eff_area_func(dec_wf, ra_wf, info_sat, mu100_list):  # TODO : limits on variables
  """
  Returns a value of the effective area for single event, compton event or 1 if the satellite is in sight for a direction dec_wt, ra_wf
  The value is obtained from mu100 files
  :param dec_wf: dec for which the function is used
  :param ra_wf: ra for which the function is used
  :param info_sat: orbital information about the satellite
  :param mu100_list: Data contained in the mu100 files
  :returns:
  """
  dec_verif(dec_wf)
  ra_verif(ra_wf)

  # extracting satellite information
  sat_alt = info_sat[3]
  # Turning the orbital parameters into dec and ra in world frame, burst_time taken = 0 for convenience
  dec_sat_wf, ra_sat_wf = sat_info_2_decra(info_sat, burst_time=0)

  if not verif_rad_belts(dec_sat_wf, ra_sat_wf, sat_alt):
    dec_sf, ra_sf = grb_decra_worldf2satf(dec_wf, ra_wf, dec_sat_wf, ra_sat_wf)

    angle_lim = horizon_angle(sat_alt)
    if dec_sf < angle_lim:
      seff_compton, seff_single = closest_mufile(dec_sf, ra_sf, mu100_list)[-2:]  # TODO test !

      # ampl = 1
      # ang_freq = 0.222
      # phi0 = 0.76
      # y_off_set = 2.5
      # seff_compton = np.absolute(ampl * np.cos(dec_sf * 2 * np.pi * ang_freq - phi0)) + y_off_set
      #
      #
      # angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 89, 91, 100, 110])
      # eff_area = np.array([137.4, 148.5, 158.4, 161.9, 157.4, 150.4, 133.5, 112.8, 87.5, 63.6, 64.7, 71.8, 77.3])
      # interpo_ite = 1
      # if dec_sf > angles[-1]:
      #   seff_single = eff_area[-2] + (eff_area[-1] - eff_area[-2]) / (angles[-1] - angles[-2]) * (dec_sf - angles[-2])
      # else:
      #   while dec_sf > angles[interpo_ite]:
      #     interpo_ite += 1
      #   seff_single = eff_area[interpo_ite - 1] + (eff_area[interpo_ite] - eff_area[interpo_ite - 1]) / (
      #             angles[interpo_ite] - angles[interpo_ite - 1]) * (dec_sf - angles[interpo_ite - 1])

      return seff_compton, seff_single, 1
    else:
      return 0, 0, 0
  else:
    return 0, 0, 0


#######################################################################################################
# Functions to manipulate the referential and positions of sat and grbs                               #
#######################################################################################################
def grb_decra_worldf2satf(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf):  # TODO : limits on variables
  """
  Converts dec_grb_wf, ra_grb_wf (declination, right ascension) world coordinates into satellite coordinate
  :param dec_grb_wf: declination of the source in world frame (0° at north pole) [deg]
  :param ra_grb_wf : Right ascension of the source in world frame (0->360°) [deg]
  :param dec_sat_wf: declination of the satellite in world frame (0° at north pole) [deg]
  :param ra_sat_wf : Right ascension of the satellite in world frame (0->360°) [deg]
  :returns: dec_grb_sf, ra_grb_sf [deg]
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
  dec_grb_sf = np.arccos(np.dot(source, z_ref_sat))
  ra_grb_sf = np.mod(np.arctan2(np.dot(source, y_ref_sat), np.dot(source, x_ref_sat)), 2*np.pi)
  return np.rad2deg(dec_grb_sf), np.rad2deg(ra_grb_sf)


def grb_decrapol_worldf2satf(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf, dec_grb_wf_err=None, ra_grb_wf_err=None, dec_sat_wf_err=None, ra_sat_wf_err=None):
  """
  Converts dec_grb_wf, ra_grb_wf (declination, right ascension) world coordinates into satellite coordinate
  Polarization angle calculation rely on the fact that the polarization angle is in the plane generated by the direction of the source and the dec=0 direction.
  The polarization angle, dec and ra are defined according to MEGAlib's RelativeY convention :
    polarization vector in the plan defined by north pole direction, source direction (and also ysat base vector hence the "RelativeY convention")
  :param dec_grb_wf : declination (except it is 0 at north pole, 90° at equator and 180° at south pole) [deg]
  :param ra_grb_wf : Right ascension (0->360°) [deg]
  :param dec_sat_wf : satellite dec in world frame [deg]
  :param ra_sat_wf : satellite ra in world frame [deg]
  :returns: pol_angle, dec_grb_sf, ra_grb_sf, dec_pol_sf, ra_pol_sf [deg]
  CARREFUL : not tested for variables being numpy arrays, validated only with floats
  """
  # Variable domain verification verif on ra_max is done without function to make things easier in the param file
  dec_verif(dec_grb_wf)
  dec_verif(dec_sat_wf)
  ra_verif(ra_grb_wf)
  ra_verif(ra_sat_wf)
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
  dec_grb_sf = np.arccos(np.dot(source, z_ref_sat))
  ra_grb_sf = np.mod(np.arctan2(np.dot(source, y_ref_sat), np.dot(source, x_ref_sat)), 2*np.pi)

  # ERROR ESTIMATION
  # TESTED WITH ANOTHER METHOD, COMPATIBLE RESULTS
  if dec_grb_wf_err is not None and ra_grb_wf_err is not None and dec_sat_wf_err is not None and ra_sat_wf_err is not None:
    dec_grb_wf_err, ra_grb_wf_err, dec_sat_wf_err, ra_sat_wf_err = np.deg2rad(dec_grb_wf_err), np.deg2rad(ra_grb_wf_err), np.deg2rad(dec_sat_wf_err), np.deg2rad(ra_sat_wf_err)

    #########################################################################################################################################################
    # dec err calculation functions
    #########################################################################################################################################################
    def utheta(tg, pg, ts, ps):
      return np.sin(tg) * np.cos(pg) * np.sin(ts) * np.cos(ps) + np.sin(tg) * np.sin(pg) * np.sin(ts) * np.sin(ps) + np.cos(tg) * np.cos(ts)

    def dertheta1(tg, pg, ts, ps):
      return np.cos(tg) * np.cos(pg) * np.sin(ts) * np.cos(ps) + np.cos(tg) * np.sin(pg) * np.sin(ts) * np.sin(ps) - np.sin(tg) * np.cos(ts)

    def dertheta2(tg, pg, ts, ps):
      return - np.sin(tg) * np.sin(pg) * np.sin(ts) * np.cos(ps) + np.sin(tg) * np.cos(pg) * np.sin(ts) * np.sin(ps)

    def dertheta3(tg, pg, ts, ps):
      return np.sin(tg) * np.cos(pg) * np.cos(ts) * np.cos(ps) + np.sin(tg) * np.sin(pg) * np.cos(ts) * np.sin(ps) - np.cos(tg) * np.sin(ts)

    def dertheta4(tg, pg, ts, ps):
      return - np.sin(tg) * np.cos(pg) * np.sin(ts) * np.sin(ps) + np.sin(tg) * np.sin(pg) * np.sin(ts) * np.cos(ps)

    #########################################################################################################################################################
    # ra err calculation functions
    #########################################################################################################################################################
    def uphi(tg, pg, ts, ps):
      return - np.sin(tg) * np.cos(pg) * np.cos(ts) * np.cos(ps) - np.sin(tg) * np.sin(pg) * np.cos(ts) * np.sin(ps) + np.cos(tg) * np.sin(ts)

    def vphi(tg, pg, ps):
      return - np.sin(tg) * np.cos(pg) * np.sin(ps) + np.sin(tg) * np.sin(pg) * np.cos(ps)

    def derphi_u1(tg, pg, ts, ps):
      return - np.cos(tg) * np.cos(pg) * np.cos(ts) * np.cos(ps) - np.cos(tg) * np.sin(pg) * np.cos(ts) * np.sin(ps) - np.sin(tg) * np.sin(ts)

    def derphi_u2(tg, pg, ts, ps):
      return np.sin(tg) * np.sin(pg) * np.cos(ts) * np.cos(ps) - np.sin(tg) * np.cos(pg) * np.cos(ts) * np.sin(ps)

    def derphi_u3(tg, pg, ts, ps):
      return np.sin(tg) * np.cos(pg) * np.sin(ts) * np.cos(ps) + np.sin(tg) * np.sin(pg) * np.sin(ts) * np.sin(ps) + np.cos(tg) * np.cos(ts)

    def derphi_u4(tg, pg, ts, ps):
      return np.sin(tg) * np.cos(pg) * np.cos(ts) * np.sin(ps) - np.sin(tg) * np.sin(pg) * np.cos(ts) * np.cos(ps)

    def derphi_v1(tg, pg, ps):
      return - np.cos(tg) * np.cos(pg) * np.sin(ps) + np.cos(tg) * np.sin(pg) * np.cos(ps)

    def derphi_v2(tg, pg, ps):
      return np.sin(tg) * np.sin(pg) * np.sin(ps) + np.sin(tg) * np.cos(pg) * np.cos(ps)

    def derphi_v4(tg, pg, ps):
      return - np.sin(tg) * np.cos(pg) * np.cos(ps) - np.sin(tg) * np.sin(pg) * np.sin(ps)

    #########################################################################################################################################################
    # error calculation
    #########################################################################################################################################################
    val_utheta = utheta(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf)
    val_vphi = vphi(dec_grb_wf, ra_grb_wf, ra_sat_wf)
    val_uphi = uphi(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf)
    dec_grb_sf_err = np.where(val_utheta**2 == 1, np.nan, np.sqrt((dertheta1(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf) * dec_grb_wf_err)**2 +
                             (dertheta2(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf) * ra_grb_wf_err)**2 +
                             (dertheta3(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf) * dec_sat_wf_err)**2 +
                             (dertheta4(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf) * ra_sat_wf_err)**2) / np.sqrt(1 - val_utheta**2))

    ra_grb_sf_err = np.where(val_vphi == 0, np.nan, (np.sqrt(((derphi_u1(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf) * val_vphi - derphi_v1(dec_grb_wf, ra_grb_wf, ra_sat_wf) * val_uphi) * dec_grb_wf_err)**2 +
                            ((derphi_u2(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf) * val_vphi - derphi_v2(dec_grb_wf, ra_grb_wf, ra_sat_wf) * val_uphi) * ra_grb_wf_err)**2 +
                            (derphi_u3(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf) * val_vphi * dec_sat_wf_err)**2 +
                            ((derphi_u4(dec_grb_wf, ra_grb_wf, dec_sat_wf, ra_sat_wf) * val_vphi - derphi_v4(dec_grb_wf, ra_grb_wf, ra_sat_wf) * val_uphi) * ra_sat_wf_err)**2) /
                     (val_vphi**2 * (1 + (val_uphi/val_vphi)**2))))

    # print(f"grb_dec_err : {np.rad2deg(dec_grb_sf_err):.4f} grb_ra_err : {np.rad2deg(ra_grb_sf_err):.4f}")
  else:
    dec_grb_sf_err, ra_grb_sf_err = 0, 0

  # Polarization
  dec_p, ra_p = np.mod(.5 * np.pi - dec_grb_wf, np.pi), ra_grb_wf + np.pi
  # pol_direction being the polarization vector in world coordinates
  pol_vec = [np.sin(dec_p) * np.cos(ra_p),
             np.sin(dec_p) * np.sin(ra_p),
             np.cos(dec_p)]
  dec_pol_sf = np.arccos(np.dot(pol_vec, z_ref_sat))
  ra_pol_sf = np.arctan2(np.dot(pol_vec, y_ref_sat), np.dot(pol_vec, x_ref_sat))
  pol_angle = np.arccos(np.dot(pol_vec, np.cross(source, y_ref_sat)))
  polstr = f"{np.sin(dec_pol_sf) * np.cos(ra_pol_sf)} {np.sin(dec_pol_sf) * np.sin(ra_pol_sf)} {np.cos(dec_pol_sf)}"
  return np.rad2deg(pol_angle), np.rad2deg(dec_grb_sf), np.rad2deg(ra_grb_sf), np.rad2deg(dec_grb_sf_err), np.rad2deg(ra_grb_sf_err), np.rad2deg(dec_pol_sf), np.rad2deg(ra_pol_sf), polstr


def decrasat2world(dec_grb_sf, ra_grb_sf, dec_sat_wf, ra_sat_wf):  # TODO : limits on variables
  """
  Converts dec_grb_sf, ra_grb_sf (declination, right ascension) satellite coordinates into world coordinate dec_grb_wf, ra_grb_wf
  :param dec_grb_sf: grb declination in sat frame (0° at instrument zenith) [deg]
  :param ra_grb_sf : grb right ascension in sat frame (0->360°) [deg]
  :param dec_sat_wf: sat declination in world frame (0° at instrument zenith) [deg]
  :param ra_sat_wf : sat right ascension in world frame (0->360°) [deg]
  :returns: dec_grb_wf, ra_grb_wf [deg]
  """
  dec_grb_sf, ra_grb_sf = np.deg2rad(dec_grb_sf), np.deg2rad(ra_grb_sf)
  dec_sat_wf, ra_sat_wf = np.deg2rad(dec_sat_wf), np.deg2rad(ra_sat_wf)
  xworld = [-np.sin(ra_sat_wf),
            -np.cos(dec_sat_wf)*np.cos(ra_sat_wf),
            np.sin(dec_sat_wf)*np.cos(ra_sat_wf)]
  yworld = [np.cos(ra_sat_wf),
            -np.cos(dec_sat_wf)*np.sin(ra_sat_wf),
            np.sin(dec_sat_wf)*np.sin(ra_sat_wf)]
  zworld = [0,
            np.sin(dec_sat_wf),
            np.cos(dec_sat_wf)]
  source = [np.sin(dec_grb_sf)*np.cos(ra_grb_sf),
            np.sin(dec_grb_sf)*np.sin(ra_grb_sf),
            np.cos(dec_grb_sf)]
  dec_grb_wf = np.arccos(np.dot(source, zworld))
  ra_grb_sf = np.mod(np.arctan2(np.dot(source, yworld), np.dot(source, xworld)), 2*np.pi)
  return np.deg2rad(dec_grb_wf), np.deg2rad(ra_grb_sf)


def orbitalparam2decra(inclination, ohm, omega, nu=0):  # TODO : limits on variables
  """
  Calculates the declination and right ascention of an object knowing its orbital parameters
  Returned results are in deg and the north direction is at 0° making the equator at 90°
  :param inclination : inclination of the orbit [deg]
  :param ohm : longitude/ra of the ascending node of the orbit [deg]
  :param omega : argument of periapsis of the orbit [deg]
  :param nu : true anomalie at epoch t0 [deg]
  :returns: dec_sat_wf, ra_sat_wf [deg]
  """
  # Variable domain verification verif on ra_max is done without function to make things easier in the param file
  # TODO
  # Puts the angle in rad and changes the omega to take into account the true anomaly
  inclination, ohm, omeganu = np.deg2rad(inclination), np.deg2rad(ohm), np.deg2rad(omega + nu)
  # normalized coordinates in the orbital frame:
  xsat = np.cos(omeganu) * np.cos(ohm) - np.sin(omeganu) * np.cos(inclination) * np.sin(ohm)
  ysat = np.cos(omeganu) * np.sin(ohm) + np.sin(omeganu) * np.cos(inclination) * np.cos(ohm)
  zsat = np.sin(inclination) * np.sin(omeganu)
  dec_sat_wf = np.arccos(zsat)
  ra_sat_wf = np.arctan2(ysat, xsat)
  return np.rad2deg(dec_sat_wf), np.rad2deg(ra_sat_wf)


def sat_info_2_decra(info_sat, burst_time):
  """
  Uses orbital parameters of a satellite in the for of "info_sat" to obtain its dec and ra in world frame
  dec is calculated so that it is from 0 to 180°
  ra is calculated so that it is from 0 to 360°
  :param info_sat: information about the satellite orbit
    [inclination, RA of ascending node, argument of periapsis, altitude]
  :param burst_time: time at which the burst occured
  :returns  dec_sat_world_frame, ra_sat_world_frame [deg] [deg]
  """
  orbital_period = orbital_period_calc(info_sat[3])
  earth_ra_offset = earth_rotation_offset(burst_time)
  true_anomaly = true_anomaly_calc(burst_time, orbital_period)
  dec_sat_world_frame, ra_sat_world_frame = orbitalparam2decra(info_sat[0], info_sat[1], info_sat[2], nu=true_anomaly)
  ra_sat_world_frame = np.mod(ra_sat_world_frame - earth_ra_offset, 360)
  return dec_sat_world_frame, ra_sat_world_frame


def decra2orbitalparam(dec_sat_wf, ra_sat_wf):  # TODO : limits on variables
  """
  Calculates the orbital parameters of an object knowing its dec and ra
    Only works for a value of omega set to pi/2
  Returned results are in rad
  :param dec_sat_wf : satellite's dec [deg]
  :param ra_sat_wf : satellite's ra [deg]
  :returns: inclination, ohm, omega [deg]
  """
  dec_sat_wf, ra_sat_wf = np.deg2rad(dec_sat_wf), np.deg2rad(ra_sat_wf)
  inclination = np.arcsin(np.cos(dec_sat_wf))
  ohm = np.arctan2(-1, np.tan(ra_sat_wf))
  omega = np.pi / 2
  return np.deg2rad(inclination), np.deg2rad(ohm), np.deg2rad(omega)


def true_anomaly_calc(time_val, period):
  """
  Calculates the true anomaly between 0 and 360° for a time "time_val" and an orbit of period "period"
  :param time_val : time at which the true anomaly is calculated
  :param period : period of the orbit
  Return the true anomaly [deg]
  """
  return np.mod(360 * time_val / period, 360)


def cond_between(val, lim1, lim2):
  """
  Checks if a value is in an interval
  :param val : value to test
  :param lim1 : inferior limit
  :param lim2 : superior limit
  Returns True if the value is in the interval [lim1, lim2]
  """
  return lim1 <= val <= lim2


def verif_zone(lat, lon):  # TODO : limits on variables
  """
  Function to verify whether a coordinate is in the exclusion area or not
  :param lat: latitude (-90 - 90°)
  :param lon: longitude (-180 - 180°)
  :returns: True when the theta and phi are in an exclusion area
  """
  # SAA
  if cond_between(lon, -60, -10) and cond_between(lat, -45, -15):
    return True
  # North pole anomaly
  elif cond_between(lon, -80, 70) and cond_between(lat, 50, 70):
    return True
  elif cond_between(lon, -180, -120) and cond_between(lat, 55, 65):
    return True
  elif cond_between(lon, 140, 180) and cond_between(lat, 60, 65):
    return True
  # South pole anomaly
  elif cond_between(lon, -125, 20) and cond_between(lat, -78, -58):
    return True
  elif cond_between(lon, 0, 90) and cond_between(lat, -58, -47):
    return True
  else:
    return False


def verif_zone_file(lat, long, file):
  """
  Function to verify whether a coordinate is in the exclusion area or not, using an exclusion file
  :param lat: latitude (-90 - 90°) [deg]
  :param long: longitude (-180 - 180°) [deg]
  :param file: exclusion file
  :returns: True when the latitude and longitude are in an exclusion area
  """
  lat_verif(lat)
  lon_verif(long)
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
  lat_index = []
  for ite in range(len(exclude_lats)):
    if exclude_lats[ite] - 0.5 <= lat <= exclude_lats[ite] + 0.5:
      lat_index.append(ite)
  for index in lat_index:
    if cond_between(long, new_exclusions[index, 1], new_exclusions[index, 2]):
      return True
  return False


def ra2lon(ra):
  """
  Change a coordinate in ra to its longitude
  :param ra: earth right ascension [0 - 360]
  :returns: longitude  [-180 - 180]
  """
  ra_verif(ra)
  if ra <= 180:
    return ra
  else:
    return ra - 360


def verif_rad_belts(dec, ra, alt):
  """
  Function to verify whether a coordinate is in the exclusion area of several exclusion files at a certain altitude
  The exclusion files represent the Earth's radiation belts
  :param dec: earth declination (0 - 180°) [deg]
  :param ra: earth ra (0 - 360°) [deg]
  :param alt: altitude of the verification
  :returns: True when the latitude and longitude are in an exclusion area
  """
  dec_verif(dec)
  ra_verif(ra)
  files = ["./bkg/exclusion/400km/AE8max_400km.out", "./bkg/exclusion/400km/AP8min_400km.out",
           "./bkg/exclusion/500km/AE8max_500km.out", "./bkg/exclusion/500km/AP8min_500km.out"]
  # files = ["./bkg/exclusion/400km/AP8min_400km.out",
  #          "./bkg/exclusion/500km/AP8min_500km.out"]
  for file in files:
    file_alt = int(file.split("km/")[0].split("/")[-1])
    if alt == file_alt:
      if verif_zone_file(90 - dec, ra2lon(ra), file):
        return True
  return False


def random_grb_dec_ra(lat_min, lat_max, lon_min, lon_max):
  """
  Take a random position for a GRB in an area defined by min/max dec and ra
  :param lat_min : minimum latitude [deg] [-90, 90]
  :param lat_max : maximum latitude [deg] [-90, 90]
  :param lon_min : minimum longitude [deg] [-180, 180[
  :param lon_max : maximum longitude [deg] [-180, 180[
  :returns: dec and ra. [0, 180] & [0, 360[
  """
  # Variable domain verification verif on ra_max is done without function to make things easier in the param file
  lat_verif(lat_min)
  lat_verif(lat_max)
  lon_verif(lon_max)
  if not -180 <= lon_min <= 180:
    raise ValueError(f"Longitude has a wrong value : -180 <= {lon_min} <= 180")
  lat_min, lat_max, lon_min, lon_max = np.deg2rad(lat_min), np.deg2rad(lat_max), np.deg2rad(lon_min), np.deg2rad(lon_max)
  dec = np.pi / 2 - np.arcsin(np.sin(lat_min) + np.random.rand() * (np.sin(lat_max) - np.sin(lat_min)))
  ra = np.mod(lon_min + np.random.rand() * (lon_max - lon_min), 2 * np.pi)
  dec, ra = np.rad2deg(dec), np.rad2deg(ra)
  # correcting some issue with the rounding of ra
  if round(ra, 4) == 360.0:
    ra = 0.0
  return dec, ra


def format_detector(det_str):
  """

  """
  unit, det = det_str.split(" ")
  if unit == "InstrumentU_1_1":
    det_id = 0
  elif unit == "InstrumentU_1_2":
    det_id = 5
  elif unit == "InstrumentU_2_1":
    det_id = 10
  elif unit == "InstrumentU_2_2":
    det_id = 15
  else:
    raise ValueError("The unit name doesn't match")

  if det == "SideDetX":
    det_id += 1
  elif det == "SideDetY":
    det_id += 2
  elif det == "Layer_1":
    det_id += 3
  elif det == "Layer_2":
    det_id += 4
  elif det == "Calor":
    det_id += 5
  else:
    raise ValueError("The detector name doesn't match")

  return det_id


def execute_finder(file, events, geometry, cpp_routine="find_detector"):
  """
  Executes the "find_detector" c++ routine that find the detector of interaction of different position of interaction
  stored in a file
  :param file: file name used to create the files
  :param events: array containing the 3 coordinate of multiple events
  :param geometry: geometry to use
  :param cpp_routine: name of the c++ routine
  :returns: an array containing a list [Instrument unit of the interaction, detector where interaction happened]
  """
  with open(f"{file}.txt", "w") as data_file:
    for event in events:
      data_file.write(f"{event[0]} {event[1]} {event[2]}\n")
  subprocess.call(f"{cpp_routine} -g {geometry} -f {file}", shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
  with open(f"{file}save.txt", "r") as save_file:
    lines = save_file.read().split("\n")[:-1]
    positions = list(map(format_detector, lines))
  return np.array(positions, dtype=np.int8)


def find_detector(pos_first_compton, pos_sec_compton, pos_single, geometry):
  """
  Execute the position finder for different arrays pos_first_compton, pos_sec_compton, pos_single
  :param pos_first_compton: array containing the position of the first compton interaction
  :param pos_sec_compton: array containing the position of the second compton interaction
  :param pos_single: array containing the position of the single event interaction
  :param sat_num: Number of the satellite that made the detection
  :param geometry: geometry to use
  :returns: 3 arrays containing a list [Instrument unit of the interaction, detector where interaction happened]
  """
  pid = os.getpid()
  file_fc = f"temp_pos_fc_{pid}"
  file_sc = f"temp_pos_sc_{pid}"
  file_s = f"temp_pos_s_{pid}"
  if len(pos_first_compton) >= 1:
    det_first_compton = execute_finder(file_fc, pos_first_compton, geometry)
    subprocess.call(f"rm {file_fc}*", shell=True)
  else:
    det_first_compton = np.array([])
  if len(pos_sec_compton) >= 1:
    det_sec_compton = execute_finder(file_sc, pos_sec_compton, geometry)
    subprocess.call(f"rm {file_sc}*", shell=True)
  else:
    det_sec_compton = np.array([])
  if len(pos_single) >= 1:
    det_single = execute_finder(file_s, pos_single, geometry)
    subprocess.call(f"rm {file_s}*", shell=True)
  else:
    det_single = np.array([])
  return det_first_compton, det_sec_compton, det_single


# Variable domain verification verif on ra_max is done without function to make things easier in the param file
def dec_verif(dec):
  """
  Raises an error if dec is not in the domain [0, 180]
  :param dec:
  """
  if not 0 <= dec <= 180:
    raise ValueError(f"Declination has a wrong value : 0 <= {dec} <= 180")


def ra_verif(ra):
  """
  Raises an error if ra is not in the domain [0, 360[
  :param ra:
  """
  if not 0 <= ra < 360:
    raise ValueError(f"Right ascension has a wrong value : 0 <= {ra} < 360")


def lat_verif(lat):
  """
  Raises an error if lat is not in the domain [-90, 90]
  :param lat:
  """
  if not -90 <= lat <= 90:
    raise ValueError(f"verif_rad_belts : Latitude has a wrong value : -90 <= {lat} <= 90")


def lon_verif(lon):
  """
  Raises an error if lon is not in the domain ]-180, 180]
  :param lon:
  """
  if not -180 < lon <= 180:
    raise ValueError(f"Longitude has a wrong value : -180 < {lon} <= 180")

# TODO verif for inclination , etc

#######################################################################################################
# Functions to create spectra                                                                         #
#######################################################################################################
def plaw(e, ampl, index_l, pivot=100):
  """
  Power-law spectrum (ph/cm2/keV/s)
  :param e: energy (keV)
  :param ampl: amplitude (ph/cm2/keV/s)
  :param index_l: spectral index
  :param pivot: pivot energy (keV), depends only on the instrument, default=100 keV for Fermi/GBM
  :returns: ph/cm2/keV/s
  """
  return ampl * (e / pivot) ** index_l


def comp(e, ampl, index_l, ep, pivot=100):
  """
  Comptonized spectrum (ph/cm2/keV/s)
  :param e: energy (keV)
  :param ampl: amplitude (ph/cm2/keV/s)
  :param index_l: spectral index
  :param ep: peak energy (keV)
  :param pivot: pivot energy (keV), depends only on the instrument, default=100 keV for Fermi/GBM
  :returns: ph/cm2/keV/s
  """
  return ampl * (e / pivot) ** index_l * np.exp(-(index_l + 2) * e / ep)


def glog(e, ampl, ec, s):
  """
  log10-gaussian spectrum model (ph/cm2/keV/s)
  :param e: energy (keV)
  :param ampl: amplitude (ph/cm2/keV/s)
  :param ec: central energy (keV)
  :param s: distribution width
  :returns: ph/cm2/keV/s
  """
  return ampl / np.sqrt(2 * np.pi * s) * np.exp(-.5 * (np.log10(e / ec) / s) ** 2)


def band(e, ampl, alpha, beta, ep, pivot=100):
  """
  Band spectrum (ph/cm2/keV/s)
  :param e: energy (keV)
  :param ampl: amplitude (ph/cm2/keV/s)
  :param alpha: low-energy spectral index
  :param beta: high-energy spectral index
  :param ep: peak energy (keV)
  :param pivot: pivot energy (keV), depends only on the instrument, default=100 keV for Fermi/GBM
  :returns: ph/cm2/keV/s
  """
  c = (alpha - beta) * ep / (alpha + 2)
  if e > c:
    return ampl * (e / pivot) ** beta * np.exp(beta - alpha) * (c / pivot) ** (alpha - beta)
  else:
    return ampl * (e / pivot) ** alpha * np.exp(-(alpha + 2) * e / ep)


def sbpl_sa(e, ampl, l1, l2, eb, delta, pivot=100):
  """
  Smoothly broken power law spectrum (ph/cm2/keV/s)
  :param e: energy (keV)
  :param ampl: amplitude (ph/cm2/keV/s)
  :param l1: first powerlaw index
  :param l2: second powerlaw index
  :param eb: break energy [keV]
  :param delta: break scale [keV]
  :param pivot: pivot energy [keV]
  """
  b, m = .5 * (l1 + l2), .5 * (l1 - l2)
  q, qp = np.log10(e / eb / delta), np.log10(pivot / eb / delta)
  a, ap = m * delta * np.log(np.cosh(q)), m * delta * np.log(np.cosh(qp))
  return ampl * (e / pivot) ** b * 10 ** (a / ap)


def sbpl(e, ampl, l1, l2, eb, delta, pivot=100):
  """
  Smoothly broken power law spectrum (ph/cm2/keV/s)
  :param e: energy (keV)
  :param ampl: amplitude (ph/cm2/keV/s)
  :param l1: first powerlaw index
  :param l2: second powerlaw index
  :param eb: break energy [keV]
  :param delta: break scale [keV]
  :param pivot: pivot energy [keV]
  """
  b, m = .5 * (l2 + l1), .5 * (l2 - l1)
  q, qp = np.log10(e / eb) / delta, np.log10(pivot / eb) / delta
  a, ap = m * delta * np.log(np.cosh(q)), m * delta * np.log(np.cosh(qp))
  return ampl * (e / pivot) ** b * 10 ** (a - ap)













# To put in a separate class for triggers
def compatibility_test(val1, bin_w1, val2, bin_w2):
  """
  Checks if the 2 intervals are compatible
  :param val1: Center of interval 1
  :param bin_w1: Amplitude of the interval 1 (=1bin)
  :param val2: Center of interval 2
  :param bin_w2: Amplitude of the interval 2 (=1bin)
  """
  min1, max1 = val1 - bin_w1, val1 + bin_w1
  min2, max2 = val2 - bin_w2, val2 + bin_w2
  return not (max2 < min1 or min2 > max1)
