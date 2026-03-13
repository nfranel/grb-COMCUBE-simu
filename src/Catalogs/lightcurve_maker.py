# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  lightcurve_maker.py
# Contains a main and various functions to create the light curves of GBM GRBs using the Fermi gbm data tool
# ================================================================

from decimal import DivisionByZero

from gbm.data import TTE, Cspec
from gbm.binning.unbinned import bin_by_time
import matplotlib.pyplot as plt
from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
import gbm
from gbm.finder import TriggerFtp

import numpy as np
import subprocess
import os
from src.Catalogs.catalog import Catalog

def bin_selector(lc, tstart, tstop, minedges, maxedges):
  """
  Extracts the count rate values for the bins whose time range falls between tstart and tstop (both inclusive).
  :param lc: gbm.data.primitives.TimeBins or gbm.background.background.BackgroundRates
      or np.ndarray, light curve or background data from which rates are extracted
  :param tstart: float, start time of the selection window [s]
  :param tstop: float, stop time of the selection window [s]
  :param minedges: np.ndarray, lower edges of all time bins [s]
  :param maxedges: np.ndarray, upper edges of all time bins [s]
  :returns: np.ndarray, count rates for the selected bins
  :raises TypeError: if lc is not one of the expected data types
  """
  if type(lc) is gbm.data.primitives.TimeBins or type(lc) is gbm.background.background.BackgroundRates:
    rates = lc.rates
  elif type(lc) is np.ndarray:
    rates = lc
  else:
    raise TypeError("Wrong data type given for lc")

  if len(minedges[minedges <= tstart]) == 0:
    startindex = 0
  else:
    startindex = len(minedges[minedges <= tstart]) - 1
  if len(maxedges[maxedges >= tstop]) == 0:
    return rates[startindex:]
  else:
    stopindex = - len(maxedges[maxedges >= tstop]) + 1
  return rates[startindex:stopindex]


def substract_bkg(lc_rates, bkg_rates):
  """
  Subtracts background count rates from light curve count rates and clips the result to zero to avoid negative values.
  :param lc_rates: np.ndarray, raw count rates from the light curve [count/s]
  :param bkg_rates: np.ndarray or float, background count rates to subtract [count/s]
  :returns: np.ndarray, background-subtracted count rates, clipped to >= 0
  """
  rates = np.array(lc_rates - bkg_rates)
  return np.where(rates >= 0, rates, 0)


def rm_files(tte_list, directory):
  """
  Removes a list of downloaded TTE or CSPEC files from disk.
  :param tte_list: list, list of str filenames (not full paths) to remove
  :param directory: str, path to the directory containing the files to remove
  """
  if not directory.endswith("/"):
    directory += "/"
  for file in tte_list:
    subprocess.call(f"rm -f {directory}{file}", shell=True)


def save_LC(rates, centroids, fullname):
  """
  Writes a background-subtracted light curve to a cosima-compatible .dat file
  in LinLin interpolation format, with time in seconds and count rate per bin.
  :param rates: np.ndarray, background-subtracted count rates for each bin [count/s]
  :param centroids: np.ndarray, centroid times of the bins [s]
  :param fullname: str, full path (directory + filename) of the output file
  :raises TypeError: if rates or centroids are not numpy arrays
  :raises ValueError: if centroids are not in strictly increasing order, or if any rate value is negative
  """
  # Checking the types are the ones expected
  if type(rates) is not np.ndarray or type(centroids) is not np.ndarray:
    raise TypeError("rates and centroids should be numpy arrays")
  for value_ite in range(len(centroids) - 1):
    if centroids[value_ite + 1] <= centroids[value_ite]:
      raise ValueError(f"The x values for {fullname} are not in increasing order, correction needed")

  with open(fullname, "w") as f:
    f.write("# Light curve file, first column is time, second is count rate\n")
    f.write("\n")
    f.write("IP LinLin\n")
    centroids -= centroids[0]
    for ite in range(len(rates)):
      if rates[ite] < 0:
        raise ValueError("Error : one of the light curve bin has a negative number of counts")
      f.write(f"DP {centroids[ite]} {rates[ite]}\n")
    f.write("EN")


def make_tte_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, p_to_m_flux, bin_size=0.1, ener_range=(10, 1000), show=False, directory="../Data/sources/", saving=True):
  """
  Downloads TTE (Time-Tagged Event) data for a GRB from the Fermi GBM archive,
  bins it in time, fits and subtracts the background, optionally corrects the
  peak-to-mean flux ratio, and saves the resulting light curve. Falls back to
  make_cspec_lc() if TTE files are unavailable or the time range is insufficient.
  :param name: str, GRB name in the format "GRByymmddxxx" (e.g. "GRB080916009")
  :param start_t90: float, start time of the T90 interval relative to trigger [s]
  :param end_t90: float, end time of the T90 interval relative to trigger [s]
  :param time_range: tuple, (t_start, t_stop) full time range for background fitting [s]
  :param bkg_range: list of two tuples, [(t_low_start, t_low_stop), (t_high_start, t_high_stop)]
      defining the pre- and post-burst background windows [s]
  :param lc_detector_mask: str, binary mask string of length equal to the number of NaI detectors; "1" marks detectors to include
  :param p_to_m_flux: float, target peak-to-mean flux ratio used for light curve correction; np.nan to skip correction, 1 to produce a flat light curve
  :param bin_size: float, time bin size [s], default=0.1
  :param ener_range: tuple, (E_min, E_max) energy range for the light curve extraction [keV], default=(10, 1000)
  :param show: bool, if True display diagnostic plots during processing, default=False
  :param directory: str, path to the directory where TTE files are downloaded and light curve files are saved, default="../Data/sources/"
  :param saving: bool, if True save the light curve plot and .dat file to disk, default=True
  :returns: int or str, 0 on success, or the GRB name string if no data files were found
  """
  #####################################################################################################################
  # Loading tte files and extracting some information
  #####################################################################################################################
  if not directory.endswith("/"):
    directory += "/"
  # Files to load :
  # initialize the Trigger data finder with a trigger number
  trig_finder = TriggerFtp(name.split("GRB")[1])
  files = trig_finder.ls_tte()
  if files == []:
    cfiles = trig_finder.ls_cspec()
    if cfiles != []:
      return make_cspec_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, p_to_m_flux, ener_range=ener_range, show=show, directory=directory, saving=saving)
    else:
      return name
  else:
    nai_files = files[2:]
    trig_finder.get_tte(directory)
    ttes = []
    for file_ite in range(len(nai_files)):
      if lc_detector_mask[file_ite] == "1":
        ttes.append(TTE.open(f"{directory}{nai_files[file_ite]}"))
    t_low_rangemax = max([tte.time_range[0] for tte in ttes])
    t_high_rangemin = min([tte.time_range[1] for tte in ttes])

    #####################################################################################################################
    # Checking if the lc needs to used cspec files and running make_cspec_lc if so
    #####################################################################################################################
    if t_low_rangemax > bkg_range[0][0] or t_high_rangemin < bkg_range[1][1]:
      rm_files(files, directory)
      return make_cspec_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, p_to_m_flux, ener_range=ener_range, show=show, directory=directory, saving=saving)
    else:
      tte_total = ttes[0].merge(ttes)

      ###################################################################################################################
      # Creating the light curve objets after merging
      ###################################################################################################################
      pha = tte_total.to_phaii(bin_by_time, bin_size, time_ref=start_t90)
      lc = pha.to_lightcurve(time_range=time_range, energy_range=ener_range)
      lc_select = lc.slice(start_t90, end_t90)

      # Verification of the centroid values
      for value_ite in range(len(lc_select.centroids) - 1):
        if lc_select.centroids[value_ite + 1] <= lc_select.centroids[value_ite]:
          raise ValueError("The centroids list has to be corrected, tte correction is not implemented")
      ###################################################################################################################
      # Creating background
      ###################################################################################################################
      try:
        backfitter = BackgroundFitter.from_phaii(pha, Polynomial, bkg_range)
        try:
          backfitter.fit(order=1)
          bkgd_model = backfitter.interpolate_bins(lc.lo_edges, lc.hi_edges)
          lc_bkgd = bkgd_model.integrate_energy(ener_range[0], ener_range[1])
        except (RuntimeError, np.linalg.LinAlgError):
          low_mean = np.mean(lc.rates[np.where(lc.centroids < bkg_range[0][1], True, False)])
          high_mean = np.mean(lc.rates[np.where(lc.centroids > bkg_range[1][0], True, False)])
          lc_bkgd = (high_mean - low_mean) / (bkg_range[1][0] - bkg_range[0][1]) * (lc.centroids - bkg_range[0][1]) + low_mean
      except np.linalg.LinAlgError:
        bkg_range = [(bkg_range[0][0] - 5, bkg_range[0][1]), (bkg_range[1][0], bkg_range[1][1] + 5)]
        if t_low_rangemax < bkg_range[0][0] and t_high_rangemin > bkg_range[1][1]:
          backfitter = BackgroundFitter.from_phaii(pha, Polynomial, bkg_range)
          try:
            backfitter.fit(order=1)
            bkgd_model = backfitter.interpolate_bins(lc.lo_edges, lc.hi_edges)
            lc_bkgd = bkgd_model.integrate_energy(ener_range[0], ener_range[1])
          except (RuntimeError, np.linalg.LinAlgError):
            # rm_files(files, directory)
            low_mean = np.mean(lc.rates[np.where(lc.centroids < bkg_range[0][1], True, False)])
            high_mean = np.mean(lc.rates[np.where(lc.centroids > bkg_range[1][0], True, False)])
            lc_bkgd = (high_mean - low_mean) / (bkg_range[1][0] - bkg_range[0][1]) * (lc.centroids - bkg_range[0][1]) + low_mean
        else:
          raise ValueError("Need to find another value for the background")
      except RuntimeWarning:
        raise DivisionByZero("Division error probably coming from BackgroundFitter")

      ###################################################################################################################
      # Correcting the rates and selecting the good bins
      ###################################################################################################################
      rates_bkg_select_total = bin_selector(lc_bkgd, start_t90, end_t90, lc.lo_edges, lc.hi_edges)
      substracted_rates = substract_bkg(lc_select.rates, rates_bkg_select_total)

      if np.isnan(p_to_m_flux):
        counts_corr = substracted_rates
      elif p_to_m_flux == 1:
        counts_corr = np.zeros(len(substracted_rates)) + np.max(substracted_rates)
      else:
        corr = (np.mean(substracted_rates) - p_to_m_flux * np.max(substracted_rates)) / (p_to_m_flux - 1)
        if corr >= 0:
          counts_corr = (substracted_rates + np.random.poisson(corr, len(substracted_rates)))
        else:
          counts_corr = (substracted_rates - np.random.poisson(-corr, len(substracted_rates)))
          if np.min(counts_corr) < 0:
            counts_corr -= np.min(counts_corr)
        counts_corr = counts_corr / np.max(counts_corr) * np.max(substracted_rates)

      if show:
        print(f"ratio peak to mean : {p_to_m_flux}")
        print("LC ratio peak to mean corrected : ", np.mean(counts_corr) / np.max(counts_corr))
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plt.suptitle(f"Light curve correction - {name}")
        ax.step(lc_select.centroids, substracted_rates, where="post", label="Reconstructed LC")
        ax.step(lc_select.centroids, counts_corr, where="post", label="Corrected LC")
        ax.set(xlabel="Time (s)", ylabel="Number of counts")
        ax.legend()
        plt.tight_layout()
        plt.show()

      ###################################################################################################################
      # Ploting if requested and saving the figure and light curves
      ###################################################################################################################
      if show:
        fig_full, ax_full = plt.subplots(figsize=(10, 6))
        ax_full.step(lc.centroids, lc.rates)
        if type(lc_bkgd) is gbm.data.primitives.TimeBins or type(lc_bkgd) is gbm.background.background.BackgroundRates:
          bkg_plot_rates = lc_bkgd.rates
        elif type(lc_bkgd) is np.ndarray:
          bkg_plot_rates = lc_bkgd
        else:
          print(type(lc_bkgd))
        ax_full.step(lc.centroids, bkg_plot_rates, color="red", label="Background fitted")
        ax_full.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve {name} with tte with background")
        ax_full.axvline(start_t90, color="black", label="T90 start and stop")
        ax_full.axvline(end_t90, color="black")
        ax_full.legend()
        plt.tight_layout()
        plt.show()

      fig, ax = plt.subplots(figsize=(10, 6))
      ax.step(lc_select.centroids, counts_corr)
      ax.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve {name} with tte")
      ax.axvline(start_t90, color="black")
      ax.axvline(end_t90, color="black")
      plt.tight_layout()
      if show:
        plt.show()
      else:
        plt.close(fig)
      if saving:
        fig.savefig(f"../Data/sources/LC_plots_GBM/LightCurve_{name}.png")
        save_LC(counts_corr, lc_select.centroids, f"../Data/sources/GBM_Light_Curves/LightCurve_{name}.dat")

      ###################################################################################################################
      # removing the files
      ###################################################################################################################
      rm_files(files, directory)
      return 0


def make_cspec_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, p_to_m_flux, ener_range=(10, 1000), show=False, directory="../Data/sources/", saving=True):
  """
  Downloads CSPEC (Continuous Spectroscopy) data for a GRB from the Fermi GBM
  archive, combines all active detectors, fits and subtracts the background,
  optionally corrects the peak-to-mean flux ratio, and saves the resulting
  light curve. Used as a fallback when TTE data are unavailable or cover an
  insufficient time range.
  :param name: str, GRB name in the format "GRByymmddxxx" (e.g. "GRB080916009")
  :param start_t90: float, start time of the T90 interval relative to trigger [s]
  :param end_t90: float, end time of the T90 interval relative to trigger [s]
  :param time_range: tuple, (t_start, t_stop) full time range for background fitting [s]
  :param bkg_range: list of two tuples, [(t_low_start, t_low_stop), (t_high_start, t_high_stop)]
      defining the pre- and post-burst background windows [s]
  :param lc_detector_mask: str, binary mask string of length equal to the number of NaI detectors; "1" marks detectors to include
  :param p_to_m_flux: float, target peak-to-mean flux ratio used for light curve correction; np.nan to skip correction, 1 to produce a flat light curve
  :param ener_range: tuple, (E_min, E_max) energy range for the light curve extraction [keV], default=(10, 1000)
  :param show: bool, if True display diagnostic plots during processing, default=False
  :param directory: str, path to the directory where CSPEC files are downloaded and light curve files are saved, default="../Data/sources/"
  :param saving: bool, if True save the light curve plot and .dat file to disk, default=True
  :returns: int, 0 on success
  """
  #####################################################################################################################
  # Loading tte files and extracting some information
  #####################################################################################################################
  if not directory.endswith("/"):
    directory += "/"
  # Files to load :
  # initialize the Trigger data finder with a trigger number
  trig_finder = TriggerFtp(name.split("GRB")[1])
  files = trig_finder.ls_cspec()
  nai_files = files[2:]
  trig_finder.get_cspec(directory)
  cspecs = []
  for file_ite in range(len(nai_files)):
    if lc_detector_mask[file_ite] == "1":
      cspecs.append(Cspec.open(f"{directory}{nai_files[file_ite]}"))

  #####################################################################################################################
  # Creating the light curve objets
  #####################################################################################################################
  lc_list = [cspec.to_lightcurve(time_range=time_range, energy_range=ener_range) for cspec in cspecs]

  lc_select_list = [lc.slice(start_t90, end_t90) for lc in lc_list]

  source_rates = np.sum(np.vstack(np.array([lc.rates for lc in lc_list])), axis=0)
  source_rates_select_list = np.array([lc.rates for lc in lc_select_list])
  source_rates_select = np.sum(np.vstack(source_rates_select_list), axis=0)

  # Verification of the centroid values
  correct_centroids = False
  for value_ite in range(len(lc_select_list[0].centroids) - 1):
    if lc_select_list[0].centroids[value_ite + 1] <= lc_select_list[0].centroids[value_ite]:
      if lc_select_list[0].centroids[value_ite - 1] > 0:
        correct_centroids = True
        print(f"The centroids list will be corrected for {name}")
      else:
        raise ValueError("The centroids list has to be corrected for a situation not implemented")

  #####################################################################################################################
  # Creating background
  #####################################################################################################################
  if correct_centroids:
    temp_selec_cent = lc_select_list[0].centroids
    temp_cent = lc_list[0].centroids
    for value_ite in range(len(lc_select_list[0].centroids) - 1):
      if lc_select_list[0].centroids[value_ite + 1] <= lc_select_list[0].centroids[value_ite]: # no need to check again if value is > 0
        temp_selec_cent[value_ite + 1] = 2 * lc_select_list[0].centroids[value_ite] - lc_select_list[0].centroids[value_ite - 1]
    for value_ite in range(len(lc_list[0].centroids) - 1):
      if lc_list[0].centroids[value_ite + 1] <= lc_list[0].centroids[value_ite]:
        if lc_list[0].centroids[value_ite - 1] > 0:
          temp_cent[value_ite + 1] = 2 * lc_list[0].centroids[value_ite] - lc_list[0].centroids[value_ite - 1]
        else:
          raise ValueError("The full centroids list has to be corrected for a situation not implemented")
    low_mean = np.mean(source_rates[np.where(temp_cent < bkg_range[0][1], True, False)])
    high_mean = np.mean(source_rates[np.where(temp_cent > bkg_range[1][0], True, False)])
    bkgd_rates = (high_mean - low_mean) / (bkg_range[1][0] - bkg_range[0][1]) * (temp_cent - bkg_range[0][1]) + low_mean
    used_centroids = temp_selec_cent
  else:
    try:
      backfitter_list = [BackgroundFitter.from_phaii(cspec, Polynomial, bkg_range) for cspec in cspecs]
      try:
        for backfitter in backfitter_list:
          backfitter.fit(order=1)
        bkgd_model_list = [backfitter.interpolate_bins(lc_list[0].lo_edges, lc_list[0].hi_edges) for backfitter in backfitter_list]
        bkgd_lc_list = [bkgd_model.integrate_energy(ener_range[0], ener_range[1]) for bkgd_model in bkgd_model_list]

        bkgd_rates = np.sum(np.vstack(np.array([lc.rates for lc in bkgd_lc_list])), axis=0)
      except (RuntimeError, np.linalg.LinAlgError):
        low_mean = np.mean(source_rates[np.where(lc_list[0].centroids < bkg_range[0][1], True, False)])
        high_mean = np.mean(source_rates[np.where(lc_list[0].centroids > bkg_range[1][0], True, False)])
        bkgd_rates = (high_mean - low_mean) / (bkg_range[1][0] - bkg_range[0][1]) * (lc_list[0].centroids - bkg_range[0][1]) + low_mean
      used_centroids = lc_select_list[0].centroids
    except RuntimeWarning:
      low_mean = np.mean(source_rates[np.where(lc_list[0].centroids < bkg_range[0][1], True, False)])
      high_mean = np.mean(source_rates[np.where(lc_list[0].centroids > bkg_range[1][0], True, False)])
      bkgd_rates = (high_mean - low_mean) / (bkg_range[1][0] - bkg_range[0][1]) * (lc_list[0].centroids - bkg_range[0][1]) + low_mean
      used_centroids = lc_select_list[0].centroids
      # raise DivisionByZero("Division error probably coming from BackgroundFitter")

  #####################################################################################################################
  # Correcting and combining the rates and selecting the good bins
  #####################################################################################################################
  bkgd_rates_select = bin_selector(bkgd_rates, start_t90, end_t90, lc_list[0].lo_edges, lc_list[0].hi_edges)
  substracted_rates = substract_bkg(source_rates_select, bkgd_rates_select)

  if np.isnan(p_to_m_flux):
    counts_corr = substracted_rates
  elif p_to_m_flux == 1:
    counts_corr = np.zeros(len(substracted_rates)) + np.max(substracted_rates)
  else:
    corr = (np.mean(substracted_rates) - p_to_m_flux * np.max(substracted_rates)) / (p_to_m_flux - 1)
    if corr >= 0:
      counts_corr = (substracted_rates + np.random.poisson(corr, len(substracted_rates)))
    else:
      counts_corr = (substracted_rates - np.random.poisson(-corr, len(substracted_rates)))
      if np.min(counts_corr) < 0:
        counts_corr -= np.min(counts_corr)
    counts_corr = counts_corr / np.max(counts_corr) * np.max(substracted_rates)

  if show:
    print(f"ratio peak to mean : {p_to_m_flux}")
    print("LC ratio peak to mean corrected : ", np.mean(counts_corr) / np.max(counts_corr))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plt.suptitle(f"Light curve correction - {name}")
    ax.step(used_centroids, substracted_rates, where="post", label="Reconstructed LC")
    ax.step(used_centroids, counts_corr, where="post", label="Corrected LC")
    ax.set(xlabel="Time (s)", ylabel="Number of counts")
    ax.legend()
    plt.tight_layout()
    plt.show()

  #####################################################################################################################
  # Creating background
  #####################################################################################################################
  if show:
    fig_full, ax_full = plt.subplots(figsize=(10, 6))
    ax_full.step(lc_list[0].centroids, source_rates)
    ax_full.step(lc_list[0].centroids, bkgd_rates, color="red", label="Background fitted")
    ax_full.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve {name} with cspec with background")
    ax_full.axvline(start_t90, color="black", label="T90 start and stop")
    ax_full.axvline(end_t90, color="black")
    ax_full.legend()
    plt.tight_layout()
    plt.show()

  fig, ax = plt.subplots(figsize=(10, 6))
  ax.step(used_centroids, counts_corr)
  ax.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve {name} with cspec")
  ax.axvline(start_t90, color="black")
  ax.axvline(end_t90, color="black")
  plt.tight_layout()
  if show:
    plt.show()
  else:
    plt.close(fig)
  if saving:
    fig.savefig(f"../Data/sources/LC_plots_GBM/LightCurve_{name}.png")
    save_LC(counts_corr, used_centroids, f"../Data/sources/GBM_Light_Curves/LightCurve_{name}.dat")

  #####################################################################################################################
  # removing the files
  #####################################################################################################################
  rm_files(files, directory)
  return 0


def create_lc(cat, ite_grb, bin_size="auto", ener_range=(10, 1000), show=False, directory="../Data/sources/", saving=True):
  """
  Orchestrates the creation of a light curve for a single GRB entry in a Catalog
  by extracting all required metadata from the catalog DataFrame and delegating
  to make_tte_lc(). The bin size can be set automatically based on the GRB duration.
  :param cat: Catalog, catalog object whose DataFrame contains the GRB metadata
  :param ite_grb: int, index of the GRB in cat.df to process
  :param bin_size: float or str, time bin size [s], or "auto" to compute it from the T90 duration using a log-linear scaling, default="auto"
  :param ener_range: tuple, (E_min, E_max) energy range for the light curve extraction [keV], default=(10, 1000)
  :param show: bool, if True display diagnostic plots during processing, default=False
  :param directory: str, path to the directory where data files are downloaded and light curve files are saved, default="../Data/sources/"
  :param saving: bool, if True save the light curve plot and .dat file to disk, default=True
  :returns: int or str, 0 on success, or the GRB name string if no data files were found (propagated from make_tte_lc)
  """
  GRBname = cat.df.name.values[ite_grb]
  t90 = float(cat.df.t90.values[ite_grb])
  start_t90 = float(cat.df.t90_start.values[ite_grb])
  end_t90 = start_t90 + t90
  time_integ_lower_energy = float(cat.df.duration_energy_low.values[ite_grb])
  time_integ_higher_energy = float(cat.df.duration_energy_high.values[ite_grb])
  bk_time_low_start = float(cat.df.back_interval_low_start.values[ite_grb])
  bk_time_low_stop = float(cat.df.back_interval_low_stop.values[ite_grb])
  bk_time_high_start = float(cat.df.back_interval_high_start.values[ite_grb])
  bk_time_high_stop = float(cat.df.back_interval_high_stop.values[ite_grb])
  lc_detector_mask = cat.df.bcat_detector_mask.values[ite_grb]
  spec_detector_mask = cat.df.scat_detector_mask.values[ite_grb]
  flu_integ_start_time = float(cat.df.flnc_spectrum_start.values[ite_grb])
  flu_integ_stop_time = float(cat.df.flnc_spectrum_stop.values[ite_grb])
  # print("verif :", GRBname, t90, start_t90, end_t90, time_integ_lower_energy, time_integ_higher_energy, bk_time_low_start,
  #       bk_time_low_stop, bk_time_high_start, bk_time_high_stop, lc_detector_mask, spec_detector_mask, flu_integ_start_time, flu_integ_stop_time)
  p_to_m_flux = cat.df.mean_flux.values[ite_grb] / cat.df.peak_flux.values[ite_grb]
  if bin_size == "auto":
    bin_number = round(10**(0.5*np.log10(t90) + 1.5), 0)
    bin_size = t90 / bin_number
    if bin_size < 0.01:
      # Keeps only 1 significative figure
      bin_size = float("%.1g" % bin_size)
    else:
      # Keeps only 2 significative figures
      bin_size = float("%.2g" % bin_size)
    bin_size = min(bin_size, 1)

  bkg_range = [(bk_time_low_start, bk_time_low_stop), (bk_time_high_start, bk_time_high_stop)]
  time_range = (bk_time_low_start, bk_time_high_stop)

  print(f"Running {GRBname}, ite : {ite_grb}")
  # print("==== 1 ====")
  return make_tte_lc(GRBname, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, p_to_m_flux, bin_size=bin_size, ener_range=ener_range, show=show, directory=directory, saving=saving)

if not os.path.exists("../Data/sources/GBM_Light_Curves"):
  os.mkdir("../Data/sources/GBM_Light_Curves")
if not os.path.exists("../Data/sources/LC_plots_GBM"):
  os.mkdir("../Data/sources/LC_plots_GBM")

gbm_cat = Catalog("../Data/CatData/allGBM.txt", [4, '\n', 5, '|', 4000], "../Data/CatData/rest_frame_properties.txt")


# MAIN to create the lightcurves
if __name__ == "__main__":
  import matplotlib as mpl
  mpl.use("Qt5Agg")

  # Saving both light curve plots and light curve files
  tosave = True
  # Showing the plots or not
  toshow = False
  for grb_ite in range(len(gbm_cat)):
    create_lc(gbm_cat, grb_ite, bin_size="auto", ener_range=(10, 1000), show=toshow, directory="../Data/sources/", saving=tosave)
