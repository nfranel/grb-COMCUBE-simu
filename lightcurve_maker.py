from gbm.data import TTE, Cspec, GbmDetectorCollection
from gbm.binning.unbinned import bin_by_time
from gbm.plot import Lightcurve, Spectrum
import matplotlib.pyplot as plt
from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
from gbm.background.unbinned import NaivePoisson
import gbm
from gbm.finder import TriggerFtp

import numpy as np
from catalog import Catalog
import subprocess
import time
import multiprocessing as mp
from itertools import repeat


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


def rm_ttes(tte_list, directory):
  """
  Removes a downloaded tte file
  :param tte_list: list of the tte file names
  :param directory: path to where the tte files were downloaded
  """
  if not directory.endswith("/"):
    directory += "/"
  for file in tte_list:
    subprocess.call(f"rm -f {directory}{file}", shell=True)


# Values from catalog :
cat_all = Catalog("GBM/allGBM.txt", [4, '\n', 5, '|', 4000])
# cat_towork = Catalog("GBM/updatedallGBM.txt", [4, '\n', 5, '|', 2000])
# print("cat loaded")
# ite = 0
# name = cat_all.name[ite]
# t90 = float(cat_all.t90[ite])
# start_t90 = float(cat_all.t90_start[ite])
# end_t90 = start_t90 + t90
# time_integ_lower_energy = float(cat_all.duration_energy_low[ite])
# time_integ_higher_energy = float(cat_all.duration_energy_high[ite])
# bk_time_low_start = float(cat_all.back_interval_low_start[ite])
# bk_time_low_stop = float(cat_all.back_interval_low_stop[ite])
# bk_time_high_start = float(cat_all.back_interval_high_start[ite])
# bk_time_high_stop = float(cat_all.back_interval_high_stop[ite])
# lc_detector_mask = cat_all.bcat_detector_mask[ite]
# spec_detector_mask = cat_all.scat_detector_mask[ite]
# flu_integ_start_time = float(cat_all.flnc_spectrum_start[ite])
# flu_integ_stop_time = float(cat_all.flnc_spectrum_stop[ite])
#
# fluence = float(cat_all.fluence[ite])
#
# bkg_range = [(bk_time_low_start, bk_time_low_stop), (bk_time_high_start, bk_time_high_stop)]
# time_range = None
# bin_size=0.064
# bin_size=0.25
# ener_range=(50, 300)
# ener_range=(10, 1000)
# show=True
# directory="../fermi_lc/"


def save_LC(rates, centroids, fullname):
  """
  Writes the light curve in a .dat file
  :param rates: Count rates for each bin of the curve
  :param centroids: Centroids of the different bins
  :param fullname: path + name of the file to save the light curves
  """
  with open(fullname, "w") as f:
    f.write("# Light curve file, first column is time, second is count rate\n")
    f.write("\n")
    f.write("IP LinLin\n")
    if type(rates) is not np.ndarray or type(centroids) is not np.ndarray:
      raise TypeError("rates and centroids should be numpy arrays")
    centroids -= centroids[0]
    for ite in range(len(rates)):
      f.write(f"DP {centroids[ite]}  {rates[ite]}\n")
    f.write("EN")


def make_tte_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, bin_size=0.0625, ener_range=(10, 1000), show=False, directory="./sources/"):
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

    #####################################################################################################################
    # Checking if the lc needs to used cspec files and running make_cspec_lc if so
    #####################################################################################################################
    if t_low_rangemax > bkg_range[0][0] or t_high_rangemin < bkg_range[1][1]:
      for file in files:
        subprocess.call(f"rm -f {directory}{file}", shell=True)
      return make_cspec_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, ener_range=ener_range, show=show, directory=directory)
    else:
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
        backfitter = BackgroundFitter.from_phaii(pha, Polynomial, bkg_range)
        try:
          backfitter.fit(order=1)
        except (RuntimeError, np.linalg.LinAlgError):
          for file in files:
            subprocess.call(f"rm -f {directory}{file}", shell=True)
          return name
      except np.linalg.LinAlgError:
        bkg_range = [(bkg_range[0][0] - 5, bkg_range[0][1]), (bkg_range[1][0], bkg_range[1][1] + 5)]
        if t_low_rangemax < bkg_range[0][0] and t_high_rangemin > bkg_range[1][1]:
          backfitter = BackgroundFitter.from_phaii(pha, Polynomial, bkg_range)
          try:
            backfitter.fit(order=1)
          except (RuntimeError, np.linalg.LinAlgError):
            for file in files:
              subprocess.call(f"rm -f {directory}{file}", shell=True)
            return name

        else:
          raise ValueError("Need to find another value for the background")
      # print(np.mean(backfitter.statistic / backfitter.dof))
      bkgd_model = backfitter.interpolate_bins(lc.lo_edges, lc.hi_edges)
      lc_bkgd = bkgd_model.integrate_energy(ener_range[0], ener_range[1])

      ###################################################################################################################
      # Correcting the rates and selecting the good bins
      ###################################################################################################################
      rates_bkg_select_total = bin_selector(lc_bkgd, start_t90, end_t90, lc_bkgd.tstart, lc_bkgd.tstop)
      substracted_rates = substract_bkg(lc_select.rates, rates_bkg_select_total)

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

      ###################################################################################################################
      # removing the files
      ###################################################################################################################
      for file in files:
        subprocess.call(f"rm -f {directory}{file}", shell=True)
      return 0

def make_cspec_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, ener_range=(10, 1000), show=False, directory="./sources/"):
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

  #####################################################################################################################
  # Creating background
  #####################################################################################################################
  backfitter_list = [BackgroundFitter.from_phaii(cspec, Polynomial, bkg_range) for cspec in cspecs]
  try:
    for backfitter in backfitter_list:
      backfitter.fit(order=1)
  except (RuntimeError, np.linalg.LinAlgError):
    for file in files:
      subprocess.call(f"rm -f {directory}{file}", shell=True)
    return name
  # print([np.mean(backfitter.statistic / backfitter.dof) for backfitter in backfitter_list])
  bkgd_model_list = [backfitter.interpolate_bins(lc_list[0].lo_edges, lc_list[0].hi_edges) for backfitter in backfitter_list]
  bkgd_lc_list = [bkgd_model.integrate_energy(ener_range[0], ener_range[1]) for bkgd_model in bkgd_model_list]

  #####################################################################################################################
  # Correcting and combining the rates and selecting the good bins
  #####################################################################################################################
  source_rates_select_list = np.array([lc.rates for lc in lc_select_list])
  # source_rates = np.sum(np.vstack(np.array([lc.rates for lc in lc_list])), axis=0)
  bkgd_rates = np.sum(np.vstack(np.array([lc.rates for lc in bkgd_lc_list])), axis=0)

  source_rates_select = np.sum(np.vstack(source_rates_select_list), axis=0)
  bkgd_rates_select = bin_selector(bkgd_rates, start_t90, end_t90, lc_list[0].lo_edges, lc_list[0].hi_edges)

  substracted_rates = substract_bkg(source_rates_select, bkgd_rates_select)

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

  #####################################################################################################################
  # removing the files
  #####################################################################################################################
  for file in files:
    subprocess.call(f"rm -f {directory}{file}", shell=True)
  return 0


def create_lc(cat, GRB_ite, bin_size="auto", ener_range=(10, 1000), show=False, directory="./sources/"):
  """

  """
  GRBname = cat.name[GRB_ite]
  t90 = float(cat.t90[GRB_ite])
  start_t90 = float(cat.t90_start[GRB_ite])
  end_t90 = start_t90 + t90
  time_integ_lower_energy = float(cat.duration_energy_low[GRB_ite])
  time_integ_higher_energy = float(cat.duration_energy_high[GRB_ite])
  bk_time_low_start = float(cat.back_interval_low_start[GRB_ite])
  bk_time_low_stop = float(cat.back_interval_low_stop[GRB_ite])
  bk_time_high_start = float(cat.back_interval_high_start[GRB_ite])
  bk_time_high_stop = float(cat.back_interval_high_stop[GRB_ite])
  lc_detector_mask = cat.bcat_detector_mask[GRB_ite]
  spec_detector_mask = cat.scat_detector_mask[GRB_ite]
  flu_integ_start_time = float(cat.flnc_spectrum_start[GRB_ite])
  flu_integ_stop_time = float(cat.flnc_spectrum_stop[GRB_ite])

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

  print(f"Running {GRBname}, ite : {GRB_ite}")
  return make_tte_lc(GRBname, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, bin_size=bin_size, ener_range=ener_range, show=show, directory=directory)
  # tte_works = make_tte_lc(GRBname, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, bin_size=bin_size, ener_range=ener_range, show=show, directory=directory)
  # if tte_works == 0:
  #   return 0
  # elif tte_works == 1:
  #   return
  # else:
  #   return tte_works

unfit = []
for ite in range(768, 769):
# for ite in range(589, 591):
  ret = create_lc(cat_all, ite, bin_size="auto", show=False)
  if ret != 0:
    unfit.append((cat_all.name[ite], ite))

# with mp.Pool() as pool:
#   ret = pool.starmap(create_lc, zip(repeat(cat_all), range(15, 18)))

# for ite in range(10):
#   print(cat_all.name[ite], cat_all.t90[ite], cat_all.t90_start[ite], bkg_min[ite], cat_all.back_interval_low_start[ite], cat_all.back_interval_high_stop[ite])

#
# create_lc(cat_all, 0, bin_size=1.024, show=False)



