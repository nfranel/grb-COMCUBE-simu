from decimal import DivisionByZero

from gbm.data import TTE, Cspec, GbmDetectorCollection
from gbm.binning.unbinned import bin_by_time
from gbm.plot import Lightcurve, Spectrum
import matplotlib.pyplot as plt
from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
import gbm
from gbm.finder import TriggerFtp

import numpy as np
import subprocess
from catalog import Catalog

# Warnings gestion
import warnings
warnings.simplefilter("error", category=RuntimeWarning)


def bin_selector(lc, tstart, tstop, minedges, maxedges):
  """
  Select the bins between tstart and tend (both included)
  :param lc: GBM lightcurve like data
  :param tstart: begining of the time selection
  :param tstop: ending of the time selection
  :param minedges: low edges of the bins
  :param maxedges: high edges of the bins
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
  returns the count rates for the light curve with substracted background
  :param lc_rates: GBM lightcurve count rates
  :param bkg_rates: Background count rates
  """
  rates = np.array(lc_rates - bkg_rates)
  return np.where(rates >= 0, rates, 0)


def rm_files(tte_list, directory):
  """
  Removes a downloaded tte file
  :param tte_list: list of the tte file names
  :param directory: path to where the tte files were downloaded
  """
  if not directory.endswith("/"):
    directory += "/"
  for file in tte_list:
    subprocess.call(f"rm -f {directory}{file}", shell=True)


def save_LC(rates, centroids, fullname):
  """
  Writes the light curve in a .dat file
  :param rates: Count rates for each bin of the curve
  :param centroids: Centroids of the different bins
  :param fullname: path + name of the file to save the light curves
  """
  # Checking the types are the ones expected
  if type(rates) is not np.ndarray or type(centroids) is not np.ndarray:
    raise TypeError("rates and centroids should be numpy arrays")
  # Verification of the values
  for value_ite in range(len(centroids) - 1):
    if centroids[value_ite + 1] <= centroids[value_ite]:
      print(f" ERROR while creating {fullname} : x values are not in increasing order !")
  with open(fullname, "w") as f:
    f.write("# Light curve file, first column is time, second is count rate\n")
    f.write("\n")
    f.write("IP LinLin\n")
    centroids -= centroids[0]
    for ite in range(len(rates)):
      f.write(f"DP {centroids[ite]} {rates[ite]}\n")
    f.write("EN")


def make_tte_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, bin_size=0.1, ener_range=(10, 1000), show=False, directory="./sources/"):
  """

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
  print("==== 2 ====")
  if files == []:
    cfiles = trig_finder.ls_cspec()
    if cfiles != []:
      return make_cspec_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, ener_range=ener_range, show=show, directory=directory)
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
    print("==== 3 ====")

    #####################################################################################################################
    # Checking if the lc needs to used cspec files and running make_cspec_lc if so
    #####################################################################################################################
    if t_low_rangemax > bkg_range[0][0] or t_high_rangemin < bkg_range[1][1]:
      print("==== 41 ====")
      rm_files(files, directory)
      return make_cspec_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, ener_range=ener_range, show=show, directory=directory)
    else:
      print("==== 42 ====")
      tte_total = ttes[0].merge(ttes)

      ###################################################################################################################
      # Creating the light curve objets after merging
      ###################################################################################################################
      pha = tte_total.to_phaii(bin_by_time, bin_size, time_ref=start_t90)
      lc = pha.to_lightcurve(time_range=time_range, energy_range=ener_range)
      lc_select = lc.slice(start_t90, end_t90)

      ###################################################################################################################
      # Creating background
      ###################################################################################################################
      try:
        print("==== 43 ====")
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
        print("==== 44 ====")
        bkg_range = [(bkg_range[0][0] - 5, bkg_range[0][1]), (bkg_range[1][0], bkg_range[1][1] + 5)]
        if t_low_rangemax < bkg_range[0][0] and t_high_rangemin > bkg_range[1][1]:
          backfitter = BackgroundFitter.from_phaii(pha, Polynomial, bkg_range)
          try:
            print("==== 45 ====")
            backfitter.fit(order=1)
            bkgd_model = backfitter.interpolate_bins(lc.lo_edges, lc.hi_edges)
            lc_bkgd = bkgd_model.integrate_energy(ener_range[0], ener_range[1])
          except (RuntimeError, np.linalg.LinAlgError):
            print("==== 46 ====")
            # rm_files(files, directory)
            low_mean = np.mean(lc.rates[np.where(lc.centroids < bkg_range[0][1], True, False)])
            high_mean = np.mean(lc.rates[np.where(lc.centroids > bkg_range[1][0], True, False)])
            lc_bkgd = (high_mean - low_mean) / (bkg_range[1][0] - bkg_range[0][1]) * (lc.centroids - bkg_range[0][1]) + low_mean
        else:
          raise ValueError("Need to find another value for the background")
      except RuntimeWarning:
        raise DivisionByZero("Division error probably coming from BackgroundFitter")
      print("==== 5 ====")

      ###################################################################################################################
      # Correcting the rates and selecting the good bins
      ###################################################################################################################
      rates_bkg_select_total = bin_selector(lc_bkgd, start_t90, end_t90, lc.lo_edges, lc.hi_edges)
      substracted_rates = substract_bkg(lc_select.rates, rates_bkg_select_total)
      print("==== 6 ====")

      ###################################################################################################################
      # Ploting if requested and saving the figure and light curves
      ###################################################################################################################
      fig, ax = plt.subplots(figsize=(10, 6))
      ax.step(lc_select.centroids, substracted_rates)
      ax.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve {name} with tte")
      ax.axvline(start_t90, color="black")
      ax.axvline(end_t90, color="black")
      if show:
        plt.show()
      else:
        plt.close(fig)
      fig.savefig(f"sources/LC_plots/LightCurve_{name}.png")
      save_LC(substracted_rates, lc_select.centroids, f"sources/Light_Curves/LightCurve_{name}.dat")
      print("==== 7 ====")

      ###################################################################################################################
      # removing the files
      ###################################################################################################################
      rm_files(files, directory)
      return 0

def make_cspec_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, ener_range=(10, 1000), show=False, directory="./sources/"):
  """

  """
  print("==== 111 ====")
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
  print("==== 112 ====")

  #####################################################################################################################
  # Creating the light curve objets
  #####################################################################################################################
  lc_list = [cspec.to_lightcurve(time_range=time_range, energy_range=ener_range) for cspec in cspecs]
  print("==== 1131 ====")

  lc_select_list = [lc.slice(start_t90, end_t90) for lc in lc_list]
  print("==== 1132 ====")

  source_rates = np.sum(np.vstack(np.array([lc.rates for lc in lc_list])), axis=0)
  print("==== 1133 ====")
  source_rates_select_list = np.array([lc.rates for lc in lc_select_list])
  print("==== 1134 ====")
  source_rates_select = np.sum(np.vstack(source_rates_select_list), axis=0)
  print("==== 1135 ====")

  #####################################################################################################################
  # Creating background
  #####################################################################################################################
  backfitter_list = [BackgroundFitter.from_phaii(cspec, Polynomial, bkg_range) for cspec in cspecs]
  print(backfitter_list[0].parameters)
  print("==== 1136 ====")
  try:
    print("==== 114 ====")
    for backfitter in backfitter_list:
      backfitter.fit(order=1)
    bkgd_model_list = [backfitter.interpolate_bins(lc_list[0].lo_edges, lc_list[0].hi_edges) for backfitter in backfitter_list]
    bkgd_lc_list = [bkgd_model.integrate_energy(ener_range[0], ener_range[1]) for bkgd_model in bkgd_model_list]

    bkgd_rates = np.sum(np.vstack(np.array([lc.rates for lc in bkgd_lc_list])), axis=0)
  except (RuntimeError, np.linalg.LinAlgError):
    print("==== 115 ====")
    low_mean = np.mean(source_rates[np.where(lc_list[0].centroids < bkg_range[0][1], True, False)])
    high_mean = np.mean(source_rates[np.where(lc_list[0].centroids > bkg_range[1][0], True, False)])
    bkgd_rates = (high_mean - low_mean) / (bkg_range[1][0] - bkg_range[0][1]) * (lc_list[0].centroids - bkg_range[0][1]) + low_mean
  print(bkgd_rates)

  #####################################################################################################################
  # Correcting and combining the rates and selecting the good bins
  #####################################################################################################################
  bkgd_rates_select = bin_selector(bkgd_rates, start_t90, end_t90, lc_list[0].lo_edges, lc_list[0].hi_edges)
  substracted_rates = substract_bkg(source_rates_select, bkgd_rates_select)
  print("==== 116 ====")

  #####################################################################################################################
  # Creating background
  #####################################################################################################################
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.step(lc_select_list[0].centroids, substracted_rates)
  ax.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve {name} with cspec")
  ax.axvline(start_t90, color="black")
  ax.axvline(end_t90, color="black")
  if show:
    plt.show()
  else:
    plt.close(fig)
  fig.savefig(f"sources/LC_plots/LightCurve_{name}.png")
  save_LC(substracted_rates, lc_select_list[0].centroids, f"sources/Light_Curves/LightCurve_{name}.dat")
  print("==== 117 ====")

  #####################################################################################################################
  # removing the files
  #####################################################################################################################
  rm_files(files, directory)
  return 0


def create_lc(cat, ite_grb, bin_size="auto", ener_range=(10, 1000), show=False, directory="./sources/"):
  """

  """
  GRBname = cat.df.name[ite_grb]
  t90 = float(cat.df.t90[ite_grb])
  start_t90 = float(cat.df.t90_start[ite_grb])
  end_t90 = start_t90 + t90
  time_integ_lower_energy = float(cat.df.duration_energy_low[ite_grb])
  time_integ_higher_energy = float(cat.df.duration_energy_high[ite_grb])
  bk_time_low_start = float(cat.df.back_interval_low_start[ite_grb])
  bk_time_low_stop = float(cat.df.back_interval_low_stop[ite_grb])
  bk_time_high_start = float(cat.df.back_interval_high_start[ite_grb])
  bk_time_high_stop = float(cat.df.back_interval_high_stop[ite_grb])
  lc_detector_mask = cat.df.bcat_detector_mask[ite_grb]
  spec_detector_mask = cat.df.scat_detector_mask[ite_grb]
  flu_integ_start_time = float(cat.df.flnc_spectrum_start[ite_grb])
  flu_integ_stop_time = float(cat.df.flnc_spectrum_stop[ite_grb])
  # print("verif :", GRBname, t90, start_t90, end_t90, time_integ_lower_energy, time_integ_higher_energy, bk_time_low_start,
  #       bk_time_low_stop, bk_time_high_start, bk_time_high_stop, lc_detector_mask, spec_detector_mask, flu_integ_start_time, flu_integ_stop_time)

  if bin_size == "auto":
    # a and b in 10**(a * log(T90) + b) obtained by fitting these values with lx the T90 and ly the desired bins
    # lx = [0.02, 0.04, 0.1, 0.2, 0.4, 1, 10, 100]
    # ly = [4, 6, 10, 20, 30, 50, 100, 300]
    bin_number = round(10**(0.5*np.log10(t90) + 1.5), 0)
    bin_size = t90 / bin_number
    if bin_size < 0.01:
      # Keeps only 1 significative figure
      bin_size = float("%.1g" % bin_size)
    # elif bin_size < 1:
    else:
      # Keeps only 2 significative figures
      bin_size = float("%.2g" % bin_size)
    bin_size = min(bin_size, 1)

  bkg_range = [(bk_time_low_start, bk_time_low_stop), (bk_time_high_start, bk_time_high_stop)]
  time_range = (bk_time_low_start, bk_time_high_stop)

  print(f"Running {GRBname}, ite : {ite_grb}")
  print("==== 1 ====")
  return make_tte_lc(GRBname, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, bin_size=bin_size, ener_range=ener_range, show=show, directory=directory)


gbm_cat = Catalog("./GBM/allGBM.txt", [4, '\n', 5, '|', 4000], "GBM/rest_frame_properties.txt")
create_lc(gbm_cat, 17, bin_size="auto", ener_range=(10, 1000), show=False, directory="./sources/")

# for grb_ite in range(len(gbm_cat)):
#   create_lc(gbm_cat, grb_ite, bin_size="auto", ener_range=(10, 1000), show=False, directory="./sources/")
