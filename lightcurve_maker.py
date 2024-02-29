from gbm.data import TTE, Cspec, GbmDetectorCollection
from gbm.binning.unbinned import bin_by_time
from gbm.plot import Lightcurve, Spectrum
import matplotlib.pyplot as plt
from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
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
      rm_files(files, directory)
      return make_cspec_lc(name, start_t90, end_t90, time_range, bkg_range, lc_detector_mask, ener_range=ener_range, show=show, directory=directory)
    else:
      tte_total = ttes[0].merge(ttes)

      ###################################################################################################################
      # Creating the light curve objets after merging
      ###################################################################################################################
      pha = tte_total.to_phaii(bin_by_time, bin_size, time_ref=start_t90)
      lc = pha.to_lightcurve(time_range=time_range, energy_range=ener_range)
      lc_select = lc.slice(start_t90, end_t90)

      # fig, ax = plt.subplots(figsize=(10, 6))
      # ax.step(lc.centroids, lc.rates)
      # ax.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve check {name} with tte")
      # ax.axvline(start_t90, color="black")
      # ax.axvline(end_t90, color="black")
      # ax.axvline(bkg_range[0][0], color="red")
      # ax.axvline(bkg_range[0][1], color="red")
      # ax.axvline(bkg_range[1][0], color="red")
      # ax.axvline(bkg_range[1][1], color="red")
      # low_mean = np.mean(lc.rates[np.where(lc.centroids < bkg_range[0][1], True, False)])
      # high_mean = np.mean(lc.rates[np.where(lc.centroids > bkg_range[1][0], True, False)])
      # bkg_rate = (high_mean - low_mean) / (bkg_range[1][0] - bkg_range[0][1]) * (lc.centroids - bkg_range[0][1]) + low_mean
      # ax.step(lc.centroids, bkg_rate, color="green")
      # ax.axhline(low_mean, color="blue")
      # ax.axhline(high_mean, color="red")
      # for val_ite, val in enumerate(lc.rates):
      #   if lc.centroids[val_ite] < start_t90:
      #     ratio = val / low_mean
      #   else:
      #     ratio = val / high_mean
      #   if ratio < 0.5:
      #     ax.axvline(lc.centroids[val_ite], color="green")
      # if show:
      #   plt.show()

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

      ###################################################################################################################
      # Correcting the rates and selecting the good bins
      ###################################################################################################################
      rates_bkg_select_total = bin_selector(lc_bkgd, start_t90, end_t90, lc.lo_edges, lc.hi_edges)
      substracted_rates = substract_bkg(lc_select.rates, rates_bkg_select_total)

      ###################################################################################################################
      # Ploting if requested and saving the figure and light curves
      ###################################################################################################################
      fig, ax = plt.subplots(figsize=(10, 6))
      ax.step(lc.centroids, lc.rates)
      ax.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve check {name} with tte")
      ax.axvline(start_t90, color="black")
      ax.axvline(end_t90, color="black")
      ax.axvline(bkg_range[0][0], color="red")
      ax.axvline(bkg_range[0][1], color="red")
      ax.axvline(bkg_range[1][0], color="red")
      ax.axvline(bkg_range[1][1], color="red")
      if type(lc_bkgd) is gbm.data.primitives.TimeBins or type(lc) is gbm.background.background.BackgroundRates:
        ax.step(lc.centroids, lc_bkgd.rates, color="green")
      elif type(lc_bkgd) is np.ndarray:
        ax.step(lc.centroids, lc_bkgd, color="green")
      if show:
        plt.show()

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
      rm_files(files, directory)
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

  source_rates = np.sum(np.vstack(np.array([lc.rates for lc in lc_list])), axis=0)
  source_rates_select_list = np.array([lc.rates for lc in lc_select_list])
  source_rates_select = np.sum(np.vstack(source_rates_select_list), axis=0)

  # fig, ax = plt.subplots(figsize=(10, 6))
  # ax.step(lc_list[0].centroids, source_rates)
  # ax.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve check {name} with cspec")
  # ax.axvline(start_t90, color="black")
  # ax.axvline(end_t90, color="black")
  # ax.axvline(bkg_range[0][0], color="red")
  # ax.axvline(bkg_range[0][1], color="red")
  # ax.axvline(bkg_range[1][0], color="red")
  # ax.axvline(bkg_range[1][1], color="red")
  # low_mean = np.mean(source_rates[np.where(lc_list[0].centroids < bkg_range[0][1], True, False)])
  # high_mean = np.mean(source_rates[np.where(lc_list[0].centroids > bkg_range[1][0], True, False)])
  # bkg_rate = (high_mean - low_mean) / (bkg_range[1][0] - bkg_range[0][1]) * (lc_list[0].centroids - bkg_range[0][1]) + low_mean
  # ax.step(lc_list[0].centroids, bkg_rate, color="green")
  # ax.axhline(low_mean, color="blue")
  # ax.axhline(high_mean, color="red")
  # for val_ite, val in enumerate(source_rates):
  #   if lc_list[0].centroids[val_ite] < start_t90:
  #     ratio = val / low_mean
  #   else:
  #     ratio = val / high_mean
  #   if ratio < 0.5:
  #     ax.axvline(lc_list[0].centroids[val_ite], color="green")
  # if show:
  #   plt.show()

  #####################################################################################################################
  # Creating background
  #####################################################################################################################
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

  #####################################################################################################################
  # Correcting and combining the rates and selecting the good bins
  #####################################################################################################################
  bkgd_rates_select = bin_selector(bkgd_rates, start_t90, end_t90, lc_list[0].lo_edges, lc_list[0].hi_edges)
  substracted_rates = substract_bkg(source_rates_select, bkgd_rates_select)

  #####################################################################################################################
  # Creating background
  #####################################################################################################################
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.step(lc_list[0].centroids, source_rates)
  ax.set(xlabel="Time(s)", ylabel="Count rate (count/s)", title=f"Light curve check {name} with cspec")
  ax.axvline(start_t90, color="black")
  ax.axvline(end_t90, color="black")
  ax.axvline(bkg_range[0][0], color="red")
  ax.axvline(bkg_range[0][1], color="red")
  ax.axvline(bkg_range[1][0], color="red")
  ax.axvline(bkg_range[1][1], color="red")
  ax.step(lc_list[0].centroids, bkgd_rates, color="green")
  if show:
    plt.show()

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
  rm_files(files, directory)
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

# unfit = []
# for ite in range(0, len(cat_all.name)):
# # for ite in range(589, 591):
#   ret = create_lc(cat_all, ite, bin_size="auto", show=False)
#   if ret != 0:
#     unfit.append((cat_all.name[ite], ite))
# print(ret)

# lc_list = subprocess.getoutput("ls ./sources/Light_Curves/").split("\n")
# lc_names = [name.split(".dat")[0].split("LightCurve_")[1] for name in lc_list]
# names = cat_all.name
#
# not_ready_name = []
# not_ready_ite = []
# for cat_ite, name in enumerate(names):
#   if name not in lc_names:
#     not_ready_name.append(name)
#     not_ready_ite.append(cat_ite)

failed = ['GRB080804456', 'GRB090514734', 'GRB091024380', 'GRB101015558', 'GRB110928180', 'GRB120713226', 'GRB120719146', 'GRB120727681', 'GRB120728434', 'GRB120728934', 'GRB120801920', 'GRB120811649', 'GRB120819048', 'GRB120820585', 'GRB120831901', 'GRB120908873', 'GRB120908938', 'GRB120915000', 'GRB120921877', 'GRB120922939', 'GRB121027038', 'GRB121029350', 'GRB121116459', 'GRB121117018', 'GRB121123421', 'GRB121125356', 'GRB121125469', 'GRB130515056', 'GRB130925173', 'GRB131006367']
failed_ite = [17, 200, 327, 549, 768, 943, 947, 948, 949, 950, 952, 955, 960, 961, 968, 971, 972, 978, 986, 987, 1000, 1002, 1009, 1010, 1016, 1019, 1020, 1114, 1203, 1209]

ret_list = []
for ite in failed_ite:
  ret = create_lc(cat_all, ite, bin_size="auto", show=True)
  ret_list.append(ret)

len(ret_list)

