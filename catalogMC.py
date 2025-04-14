import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import subprocess
from itertools import repeat
from time import time

from fontTools.varLib.plot import stops
from matplotlib.pyplot import suptitle

from catalog import Catalog
from funcmod import extract_lc, calc_flux_gbm, use_scipyquad, pflux_to_mflux_calculator
from funcsample import *
from scipy.stats import norm, chisquare
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl

# mpl.use("Qt5Agg")
mpl.use('Agg')


def categorize_pierson_chi2(pierson_chi2_array, mode="fine"):
  sup18 = np.sum(np.where(pierson_chi2_array >= 18, 1, 0))
  sup50 = np.sum(np.where(pierson_chi2_array >= 50, 1, 0))
  sup300 = np.sum(np.where(pierson_chi2_array >= 300, 1, 0))
  sup1000 = np.sum(np.where(pierson_chi2_array >= 1000, 1, 0))
  c0to6 = np.sum(np.where(pierson_chi2_array >= 0, np.where(pierson_chi2_array < 6, 1, 0), 0))
  c0to12 = np.sum(np.where(pierson_chi2_array >= 0, np.where(pierson_chi2_array < 12, 1, 0), 0))
  c0to18 = np.sum(np.where(pierson_chi2_array >= 0, np.where(pierson_chi2_array < 18, 1, 0), 0))
  c0to50 = np.sum(np.where(pierson_chi2_array >= 0, np.where(pierson_chi2_array < 50, 1, 0), 0))
  c6to12 = np.sum(np.where(pierson_chi2_array >= 6, np.where(pierson_chi2_array < 12, 1, 0), 0))
  c12to18 = np.sum(np.where(pierson_chi2_array >= 12, np.where(pierson_chi2_array < 18, 1, 0), 0))
  c12to24 = np.sum(np.where(pierson_chi2_array >= 12, np.where(pierson_chi2_array < 24, 1, 0), 0))
  c18to50 = np.sum(np.where(pierson_chi2_array >= 18, np.where(pierson_chi2_array < 50, 1, 0), 0))
  c24to50 = np.sum(np.where(pierson_chi2_array >= 24, np.where(pierson_chi2_array < 50, 1, 0), 0))
  c50to300 = np.sum(np.where(pierson_chi2_array >= 50, np.where(pierson_chi2_array < 300, 1, 0), 0))
  c300to1000 = np.sum(np.where(pierson_chi2_array >= 300, np.where(pierson_chi2_array < 1000, 1, 0), 0))
  if mode == "fine":
    c1 = f"0 - 6  |  {c0to6} sims"
    c2 = f"6 - 12  |  {c6to12} sims"
    c3 = f"12 - 18  |  {c12to18} sims"
    c4 = f">= 18  |  {sup18} sims"
    categorized = np.where(pierson_chi2_array >= 18, c4, np.where(pierson_chi2_array >= 12, c3, np.where(pierson_chi2_array >= 6, c2, c1)))
    hue_order = [c1, c2, c3, c4]

  elif mode == "medium_fine":
    c1 = f"0 - 12  |  {c0to12} sims"
    c2 = f"12 - 24  |  {c12to24} sims"
    c3 = f"24 - 50  |  {c24to50} sims"
    c4 = f">= 50  |  {sup50} sims"
    categorized = np.where(pierson_chi2_array >= 50, c4, np.where(pierson_chi2_array >= 24, c3, np.where(pierson_chi2_array >= 12, c2, c1)))
    hue_order = [c1, c2, c3, c4]

  elif mode == "medium_coarse":
    c1 = f"0 - 18  |  {c0to18} sims"
    c2 = f"18 - 50  |  {c18to50} sims"
    c3 = f"50 - 300  |  {c50to300} sims"
    c4 = f">= 300  |  {sup300} sims"
    categorized = np.where(pierson_chi2_array >= 300, c4, np.where(pierson_chi2_array >= 50, c3, np.where(pierson_chi2_array >= 18, c2, c1)))
    hue_order = [c1, c2, c3, c4]

  elif mode == "coarse":
    c1 = f"0 - 50  |  {c0to50} sims"
    c2 = f"50 - 300  |  {c50to300} sims"
    c3 = f"300 - 1000  |  {c300to1000} sims"
    c4 = f">= 1000  |  {sup1000} sims"
    categorized = np.where(pierson_chi2_array >= 1000, c4, np.where(pierson_chi2_array >= 300, c3, np.where(pierson_chi2_array >= 50, c2, c1)))
    hue_order = [c1, c2, c3, c4]

  else:
    raise ValueError("Wrong name for mode")
  return categorized, hue_order


def get_df(select_col, csvfile="./Sampled/longred_lum_discreet/longfit_red.csv"):
  result_df = pd.read_csv(csvfile)
  return result_df[select_col]


def MC_explo_pairplot(fileused, legend_mode, grbtype):
  if grbtype == "long":
    extract_cols = ["long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "long_ind1_lum", "long_ind2_lum", "long_lb", "pierson_chi2"]
    select_cols = ["long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "long_ind1_lum", "long_ind2_lum", "long_lb"]
  elif grbtype == "short":
    extract_cols = ["short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "short_ind1_lum", "short_ind2_lum", "short_lb", "pierson_chi2"]
    select_cols = ["short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "short_ind1_lum", "short_ind2_lum", "short_lb"]

  df_selec = get_df(extract_cols, csvfile=fileused)
  # df_selec = df_selec[df_selec.pierson_chi2 > -20]

  pierson_chi2_categories, order_hue = categorize_pierson_chi2(df_selec['pierson_chi2'].values, mode=legend_mode)

  # rainbow_palette = sns.color_palette("rainbow", len(order_hue))  # Nombre de catégories
  # palette = {cat: rainbow_palette[i] for i, cat in enumerate(order_hue)}

  df_selec['pierson_chi2_category'] = pierson_chi2_categories

  sns.pairplot(df_selec.sort_values(by="pierson_chi2", ascending=False), hue="pierson_chi2_category", vars=select_cols, corner=False, plot_kws={'s': 20}, palette="rainbow_r")


class MCCatalog:
  """

  """
  def __init__(self, gbm_file="GBM/allGBM.txt", sttype=None, rf_file="GBM/rest_frame_properties.txt", mode="catalog"):
    """

    """
    if sttype is None:
      sttype = [4, '\n', 5, '|', 4000]

    # Computation variables
    self.zmin = 0
    self.zmax = 10
    self.epmin = 1e0
    self.epmax = 1e5
    self.lmin = 1e49  # erg/s
    self.lmax = 3e54
    self.n_year = 10
    gbmduty = 0.587
    self.gbm_weight = 1 / gbmduty
    self.sample_weight = 1
    self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Extracting GBM data and spliting in short and long GRBs for verifications
    self.gbm_cat = Catalog(gbm_file, sttype, rf_file)
    self.df_short = self.gbm_cat.df[self.gbm_cat.df.t90 <= 2].reset_index(drop=True)
    self.df_long = self.gbm_cat.df[self.gbm_cat.df.t90 > 2].reset_index(drop=True)
    self.ergcut = (10, 1000)

    self.gbm_l_mflux = []
    self.gbm_l_pflux = []
    self.gbm_l_flnc = []
    self.gbm_l_mp_ratio = []
    for ite_l in range(len(self.df_long)):
      model = self.df_long.flnc_best_fitting_model[ite_l]
      p_model = self.df_long.pflx_best_fitting_model[ite_l]
      if type(p_model) is str:
        self.gbm_l_pflux.append(self.df_long[f"{p_model}_phtflux"][ite_l])
        self.gbm_l_mp_ratio.append(self.df_long[f"{p_model}_phtflux"][ite_l] / self.df_long[f"{model}_phtflux"][ite_l])
      self.gbm_l_mflux.append(self.df_long[f"{model}_phtflux"][ite_l])
      self.gbm_l_flnc.append(calc_flux_gbm(self.df_long, ite_l, self.ergcut, cat_is_df=True) * self.df_long.t90[ite_l])

    self.gbm_s_mflux = []
    self.gbm_s_pflux = []
    self.gbm_s_flnc = []
    self.gbm_s_mp_ratio = []
    for ite_s in range(len(self.df_short)):
      model = self.df_short.flnc_best_fitting_model[ite_s]
      p_model = self.df_short.pflx_best_fitting_model[ite_s]
      if type(p_model) is str:
        self.gbm_s_pflux.append(self.df_short[f"{p_model}_phtflux"][ite_s])
        self.gbm_s_mp_ratio.append(self.df_short[f"{p_model}_phtflux"][ite_s] / self.df_short[f"{model}_phtflux"][ite_s])
      self.gbm_s_mflux.append(self.df_short[f"{model}_phtflux"][ite_s])
      self.gbm_s_flnc.append(calc_flux_gbm(self.df_short, ite_s, self.ergcut, cat_is_df=True) * self.df_short.t90[ite_s])

    ######### CHECK IF WE KEEP A VARIABLE FOR THE TIME OF THE CATALOG, BUT GBM IS 10 YEARS OF DATA SO WE SHOULD BROBABLY STICK TO THAT TIME

    # Acceptance limits
    # Number of GRB
    self.nlong_min = 5000
    self.nlong_max = 100000
    self.nshort_min = 1500
    self.nshort_max = 25000
    self.long_short_rate_min = 4
    self.long_short_rate_max = 6.5

    self.flux_lim = [10, 120]
    self.flnc_l_lim = [300, 1800]
    self.flnc_s_lim = [10, 20]
    self.nfluxbin_l = [30, 5, 1]
    self.nfluxbin_s = [30, 3, 1]
    self.nflncbin_l = [30, 4, 1]
    self.nflncbin_s = [30, 1, 1]
    self.usual_bins = np.logspace(-1, 4, 50)
    # l_pflux_bright = np.array([16.44, 27, 44.4, 73, 120 , 1000])
    l_pflux_bright = np.array([12, 14, 16, 19, 22, 27, 35, 45, 57, 73, 120, 1000])
    # s_pflux_bright = np.array([10,  23,  52.4, 120 , 1000])
    s_pflux_bright = np.array([12, 14.5, 17, 20, 25, 34, 53, 120 , 1000])
    self.bin_flux_l = np.concatenate((np.logspace(-1, np.log10(self.flux_lim[0]), self.nfluxbin_l[0] + 1), l_pflux_bright))
    self.bin_flux_s = np.concatenate((np.logspace(-1, np.log10(self.flux_lim[0]), self.nfluxbin_s[0] + 1), s_pflux_bright))
    self.bin_flnc_l = np.concatenate((np.logspace(-1, np.log10(self.flnc_l_lim[0]), self.nflncbin_l[0] + 1),
                                      np.logspace(np.log10(self.flnc_l_lim[0]), np.log10(self.flnc_l_lim[1]), self.nflncbin_l[1] + 1)[1:],
                                      np.logspace(np.log10(self.flnc_l_lim[1]), 4, self.nflncbin_l[2] + 1)[1:]))
    self.bin_flnc_s = np.concatenate((np.logspace(-1, np.log10(self.flnc_s_lim[0]), self.nflncbin_s[0] + 1),
                                      np.logspace(np.log10(self.flnc_s_lim[0]), np.log10(self.flnc_s_lim[1]), self.nflncbin_s[1] + 1)[1:],
                                      np.logspace(np.log10(self.flnc_s_lim[1]), 4, self.nflncbin_s[2] + 1)[1:]))
    pflux_l_hist = np.histogram(self.gbm_l_pflux, bins=self.bin_flux_l, weights=[self.gbm_weight] * len(self.gbm_l_pflux))[0]
    pflux_s_hist = np.histogram(self.gbm_s_pflux, bins=self.bin_flux_s, weights=[self.gbm_weight] * len(self.gbm_s_pflux))[0]
    flnc_l_hist = np.histogram(self.gbm_l_flnc, bins=self.bin_flnc_l, weights=[self.gbm_weight] * len(self.gbm_l_flnc))[0]
    flnc_s_hist = np.histogram(self.gbm_s_flnc, bins=self.bin_flnc_s, weights=[self.gbm_weight] * len(self.gbm_s_flnc))[0]
    #   Flux
    # Binned GBM counts
    # Version treating high and low bins differently
    self.l_low_pflux_bins = pflux_l_hist[self.nfluxbin_l[0]:self.nfluxbin_l[0] + self.nfluxbin_l[1]]
    self.l_high_pflux_bins = pflux_l_hist[self.nfluxbin_l[0] + self.nfluxbin_l[1]:]
    self.s_low_pflux_bins = pflux_s_hist[self.nfluxbin_s[0]:self.nfluxbin_s[0] + self.nfluxbin_s[1]]
    self.s_high_pflux_bins = pflux_s_hist[self.nfluxbin_s[0] + self.nfluxbin_s[1]:]
    # Version treating high and low bins the same way
    self.l_pflux_bins = pflux_l_hist[self.nfluxbin_l[0]:]
    self.s_pflux_bins = pflux_s_hist[self.nfluxbin_s[0]:]

    # Fluence
    # Binned GBM counts
    # Version treating high and low bins differently
    self.l_low_flnc_bins = flnc_l_hist[self.nflncbin_l[0]:self.nflncbin_l[0] + self.nflncbin_l[1]]
    self.l_high_flnc_bins = flnc_l_hist[self.nflncbin_l[0] + self.nflncbin_l[1]:]
    self.s_low_flnc_bins = flnc_l_hist[self.nflncbin_s[0]:self.nflncbin_s[0] + self.nflncbin_s[1]]
    self.s_high_flnc_bins = flnc_l_hist[self.nflncbin_s[0] + self.nflncbin_s[1]:]
    # Version treating high and low bins the same way
    self.l_flnc_bins = flnc_l_hist[self.nflncbin_l[0]:]
    self.s_flnc_bins = flnc_s_hist[self.nflncbin_s[0]:]


    # INITIAL min and max values for distributions (Wandermann & Piran 2021, Lien, 2014, Lan et al 2019 for long GRBs and Ghirlanda, 2016 for short ones)
    # Redshift
    self.l_rate_min = 0.4
    self.l_rate_max = 2.1
    self.l_ind1_z_min = 1.5
    self.l_ind1_z_max = 4.3
    self.l_ind2_z_min = -2.4
    self.l_ind2_z_max = 1
    self.l_zb_min = 2.3
    self.l_zb_max = 3.7

    self.s_rate_min = 0.1
    self.s_rate_max = 1.1
    self.s_ind1_z_min = 0.5
    self.s_ind1_z_max = 4.1
    self.s_ind2_z_min = 0.9
    self.s_ind2_z_max = 4
    self.s_zb_min = 1.7
    self.s_zb_max = 3.3
    # Luminosity
    self.l_ind1_min = -1.5
    self.l_ind1_max = -1.2
    self.l_ind2_min = -2.1
    self.l_ind2_max = -0.8
    self.l_lb_min = 2e51
    self.l_lb_max = 3e+53

    self.s_ind1_min = -1
    self.s_ind1_max = -0.39
    self.s_ind2_min = -3.7
    self.s_ind2_max = -1.7
    self.s_lb_min = 0.91e52
    self.s_lb_max = 3.4e52

    # Narrower parameter space after studying the results of Monte Carlo
    # Redshift
    # self.l_rate_min = 0.4
    # self.l_rate_max = 0.8
    # self.l_ind1_z_min = 1.9
    # self.l_ind1_z_max = 3
    # self.l_ind2_z_min = -2.4
    # self.l_ind2_z_max = -0.01
    # self.l_zb_min = 2
    # self.l_zb_max = 3.5
    #
    # self.s_rate_min = 0.3
    # self.s_rate_max = 0.9
    # self.s_ind1_z_min = 0.6
    # self.s_ind1_z_max = 3.4
    # self.s_ind2_z_min = 1.6
    # self.s_ind2_z_max = 3.5
    # self.s_zb_min = 2
    # self.s_zb_max = 3
    # # Luminosity
    # self.l_ind1_min = -0.9
    # self.l_ind1_max = -0.65
    # self.l_ind2_min = -5
    # self.l_ind2_max = -3
    # self.l_lb_min = 1e52
    # self.l_lb_max = 1.9e52
    #
    # self.s_ind1_min = -1
    # self.s_ind1_max = -0.4
    # self.s_ind2_min = -3.3
    # self.s_ind2_max = -1.7
    # self.s_lb_min = 0.91e52
    # self.s_lb_max = 3.4e52

    # Spectrum indexes gaussian distributions
    self.band_low_l_mu, self.band_low_l_sig = -0.9608, 0.3008
    self.band_high_l_mu, self.band_high_l_sig = -2.1643, 0.2734
    self.band_low_s_mu, self.band_low_s_sig = -0.5749, 0.3063
    self.band_high_s_mu, self.band_high_s_sig = -2.1643, 0.2734
    # T90 gaussian distributions
    # amplitude long : 467, mean long : 1.4875, stdev long : 0.45669
    # amplitude short : 137.5, mean short : -0.025, stdev short : 0.631
    self.log_t90_l_mu, self.log_t90_l_sig = 1.4875, 0.45669
    self.log_t90_s_mu, self.log_t90_s_sig = -0.025, 0.631

    # variables containing the MCMC results
    self.columns = ["long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "long_ind1_lum", "long_ind2_lum", "long_lb", "short_ind1_lum", "short_ind2_lum",
                    "short_lb", "pierson_chi2", "status"]
    self.result_df = pd.DataFrame(columns=self.columns)

    # build_params(l_rate, l_ind1_z, l_ind2_z, l_zb, l_ind1, l_ind2, l_lb, s_rate, s_ind1_z, s_ind2_z, s_zb, s_ind1, s_ind2, s_lb)
    # main :     [0.42, 2.07, -0.7, 3.6, -0.65, -3, 1.12e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52]

    # param_list = [[0.42, 2.07, -0.7, 3.6, -0.36, -1.28, 1.48e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],  # Lan no evo
    #               [0.42, 2.07, -0.7, 3.6, -0.69, -1.76, 2.09e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],  # Lan empirical
    #               [0.42, 2.07, -0.7, 3.6, -0.2, -1.4, 3.16e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],    # Wanderman Piran
    #               [0.42, 2.07, -0.7, 3.6, -0.65, -3, 1.12e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52]]     # Lien

    # ==============================================================================================================================================================
    # Parameters to change
    # ==============================================================================================================================================================
    thread_num = 60

    print("Starting")
    if mode == "catalog":
      param_list = None
      par_size = 1
      fold_name = f"cat_to_validate"
      savefolder = f"Sampled/{fold_name}/"
      sigma_number = 1

      if not (f"{fold_name}" in os.listdir("Sampled/")):
        os.mkdir(f"Sampled/{fold_name}")
      else:
        raise NameError("A simulation with this name already exists, please change it or delete the old simulation before running")

      print(f"Starting the catalog creation")
      if thread_num == 'all':
        print("Parallel MC execution with all threads")
      elif type(thread_num) is int and thread_num > 1:
        print(f"Parallel MC execution with {thread_num} threads")
      else:
        print(f"MC execution with 1 thread")

      rows_ret = [self.get_catalog_sample(ite, thread_num, savefolder, method=param_list, comment="", n_sig=sigma_number) for ite in range(par_size)]
      self.result_df = pd.DataFrame(data=rows_ret, columns=self.columns)
      self.result_df.to_csv(f"{savefolder}catalogs_fit.csv", index=False)
    else:
      if mode == "mc":
        param_list = None
        par_size = 2000
        mctype = "long"
        # mctype = "short"
        fold_name = f"mc{mctype}v2-{par_size}"
        savefile = f"Sampled/{fold_name}/mc_fit.csv"
      elif mode == "parametrized":
        # (l_rate, l_ind1_z, l_ind2_z, l_zb, l_ind1, l_ind2, l_lb, s_rate, s_ind1_z, s_ind2_z, s_zb, s_ind1, s_ind2, s_lb)
        # param_list = build_params([0.2, 0.3, 0.4, 0.5], [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6], [-1.1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4], [2, 2.6, 3.1, 3.6, 4.1, 5], -0.65, -3, 1.12e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52)
        param_list = [[0.796361, 2.242970, -1.507276, 2.294414, -0.715277, -4.629343, 1.131491e52, 0.577574, 3.107413, 1.179197, 2.318962, -0.602085, -2.148571, 1.899889e52]]
        par_size = len(param_list)
        fold_name = f"parametrizedv1-{par_size}"
        savefile = f"Sampled/{fold_name}/longfit_red_lum.csv"
      else:
        raise ValueError("Wrong value for mode. Only 'catalog', 'mc' and 'parametrized' are possible.")

      if not (f"{fold_name}" in os.listdir("Sampled/")):
        os.mkdir(f"Sampled/{fold_name}")
      else:
        raise NameError("A simulation with this name already exists, please change it or delete the old simulation before running")
      self.run_mc(par_size, thread_number=thread_num, method=param_list, savefile=savefile, mctype=mctype)

  def run_mc(self, run_number, thread_number=1, method=None, savefile=None, comment="", mctype="long"):
    print(f"Starting the run for {run_number} iterations")
    if thread_number == 'all':
      print("Parallel execution with all threads")
      with mp.Pool() as pool:
        rows_ret = pool.starmap(self.get_sample, zip(range(run_number), repeat(method), repeat(comment), repeat(savefile), repeat(mctype)))
    elif type(thread_number) is int and thread_number > 1:
      print(f"Parallel execution with {thread_number} threads")
      with mp.Pool(thread_number) as pool:
        rows_ret = pool.starmap(self.get_sample, zip(range(run_number), repeat(method), repeat(comment), repeat(savefile), repeat(mctype)))
    else:
      rows_ret = [self.get_sample(ite, method=method, comment=comment, savefile=savefile, mctype=mctype) for ite in range(run_number)]
    self.result_df = pd.DataFrame(data=rows_ret, columns=self.columns)
    if savefile is not None:
      self.result_df.to_csv(savefile, index=False)

  def get_params(self):
    l_rate_temp = equi_distri(self.l_rate_min, self.l_rate_max)
    l_ind1_z_temp = equi_distri(self.l_ind1_z_min, self.l_ind1_z_max)
    l_ind2_z_temp = equi_distri(self.l_ind2_z_min, self.l_ind2_z_max)
    l_zb_temp = equi_distri(self.l_zb_min, self.l_zb_max)

    s_rate_temp = equi_distri(self.s_rate_min, self.s_rate_max)
    s_ind1_z_temp = equi_distri(self.s_ind1_z_min, self.s_ind1_z_max)
    s_ind2_z_temp = equi_distri(self.s_ind2_z_min, self.s_ind2_z_max)
    s_zb_temp = equi_distri(self.s_zb_min, self.s_zb_max)
    nlong_temp = int(self.n_year * use_scipyquad(red_rate_long, self.zmin, self.zmax, func_args=(l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp), x_logscale=False)[0])

    nshort_temp = int(self.n_year * use_scipyquad(red_rate_short, self.zmin, self.zmax, func_args=(s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp), x_logscale=False)[0])
    grb_number_picking = not (self.nlong_min <= nlong_temp <= self.nlong_max and self.nshort_min <= nshort_temp <= self.nshort_max and self.long_short_rate_min <= nlong_temp / nshort_temp <= self.long_short_rate_max)
    z_loop_ite = 0
    while grb_number_picking:
      l_rate_temp = equi_distri(self.l_rate_min, self.l_rate_max)
      l_ind1_z_temp = equi_distri(self.l_ind1_z_min, self.l_ind1_z_max)
      l_ind2_z_temp = equi_distri(self.l_ind2_z_min, self.l_ind2_z_max)
      l_zb_temp = equi_distri(self.l_zb_min, self.l_zb_max)

      s_rate_temp = equi_distri(self.s_rate_min, self.s_rate_max)
      s_ind1_z_temp = equi_distri(self.s_ind1_z_min, self.s_ind1_z_max)
      s_ind2_z_temp = equi_distri(self.s_ind2_z_min, self.s_ind2_z_max)
      s_zb_temp = equi_distri(self.s_zb_min, self.s_zb_max)
      nlong_temp = int(self.n_year * use_scipyquad(red_rate_long, self.zmin, self.zmax, func_args=(l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp), x_logscale=False)[0])

      nshort_temp = int(self.n_year * use_scipyquad(red_rate_short, self.zmin, self.zmax, func_args=(s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp), x_logscale=False)[0])
      grb_number_picking = not (self.nlong_min <= nlong_temp <= self.nlong_max and self.nshort_min <= nshort_temp <= self.nshort_max and self.long_short_rate_min <= nlong_temp / nshort_temp <= self.long_short_rate_max)
      z_loop_ite += 1

    l_ind1_temp = equi_distri(self.l_ind1_min, self.l_ind1_max)
    l_ind2_temp = equi_distri(self.l_ind2_min, self.l_ind2_max)
    l_lb_temp = equi_distri(self.l_lb_min, self.l_lb_max)
    s_ind1_temp = equi_distri(self.s_ind1_min, self.s_ind1_max)
    s_ind2_temp = equi_distri(self.s_ind2_min, self.s_ind2_max)
    s_lb_temp = equi_distri(self.s_lb_min, self.s_lb_max)

    l_params = l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, nlong_temp
    s_params = s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, nshort_temp
    return l_params, s_params

  def get_set_params(self, params):
    l_rate_temp = params[0]
    l_ind1_z_temp = params[1]
    l_ind2_z_temp = params[2]
    l_zb_temp = params[3]

    s_rate_temp = params[7]
    s_ind1_z_temp = params[8]
    s_ind2_z_temp = params[9]
    s_zb_temp = params[10]
    nlong_temp = int(self.n_year * use_scipyquad(red_rate_long, self.zmin, self.zmax, func_args=(l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp), x_logscale=False)[0])

    nshort_temp = int(self.n_year * use_scipyquad(red_rate_short, self.zmin, self.zmax, func_args=(s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp), x_logscale=False)[0])
    grb_number_picking = not (self.nlong_min <= nlong_temp <= self.nlong_max and self.nshort_min <= nshort_temp <= self.nshort_max and self.long_short_rate_min <= nlong_temp / nshort_temp <= self.long_short_rate_max)
    if grb_number_picking:
      print(f"Number of long ans short burst not matching : {nlong_temp} - {nshort_temp} ratio : {nlong_temp / nshort_temp}")

    l_ind1_temp = params[4]
    l_ind2_temp = params[5]
    l_lb_temp = params[6]

    s_ind1_temp = params[11]
    s_ind2_temp = params[12]
    s_lb_temp = params[13]

    l_params = l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, nlong_temp
    s_params = s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, nshort_temp
    return l_params, s_params

  def get_sample(self, run_iteration, method=None, comment="", savefile=None, mctype="long"):
    # Using a different seed for each thread, somehow the seed what the same without using it
    np.random.seed(os.getpid() + int(time() * 1000) % 2**32)

    end_flux_loop = True
    while end_flux_loop:
      if method is None:
        params = self.get_params()
      elif type(method) is list:
        params = self.get_set_params(method[run_iteration])
      else:
        raise ValueError("Wrong method used")

      l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, nlong_temp = params[0]
      s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, nshort_temp = params[1]

      print(f"Begin of {nlong_temp} longs and {nshort_temp} shorts     [ite {run_iteration}]")
      l_temp_ret = np.array([self.get_long(ite_long, l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp) for ite_long in range(nlong_temp)])
      print(f"Long finished     [ite {run_iteration}]")

      s_temp_ret = np.array([self.get_short(ite_short, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp) for ite_short in range(nshort_temp)])
      print(f"Short finished     [ite {run_iteration}]")

      l_m_flux_temp, l_p_flux_temp, l_flnc_temp = np.array(l_temp_ret[:, 4], dtype=np.float64), np.array(l_temp_ret[:, 5], dtype=np.float64), np.array(l_temp_ret[:, 7], dtype=np.float64)
      s_m_flux_temp, s_p_flux_temp, s_flnc_temp = np.array(s_temp_ret[:, 4], dtype=np.float64), np.array(s_temp_ret[:, 5], dtype=np.float64), np.array(s_temp_ret[:, 7], dtype=np.float64)

      if mctype == "long":
        cond_mode = "l_pflx"
      elif mctype == "short":
        cond_mode = "s_pflx"
      else:
        raise ValueError("Use a correct value for mctype : 'short' or 'long'")
      condition = self.mcmc_condition(l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, params=params, mode=cond_mode)
      pflux_ratio_thresh = 4
      if condition[0] or method is None:
        end_flux_loop = False
      else:
        print(f"====== LOOPING - pflux ratio : {round(condition[2], 2)} > {pflux_ratio_thresh} ======")

    if condition[0]:
      row = [l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(condition[1], 3), "Accepted"]
    else:
      row = [l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(condition[1], 3), "Rejected"]
      print(f"Rejected : pearson chi2 = {np.around(condition[1], 3)}     [ite {run_iteration}]")

    list_param = l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp
    self.hist_plotter(run_iteration, [l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, np.around(condition[1], 3)], list_param, comment=comment, savefile=savefile)

    return row

  def get_catalog_sample(self, run_iteration, thread_number, savefolder, method=None, comment="", n_sig=1):
    # Using a different seed for each thread, somehow the seed what the same without using it
    np.random.seed(os.getpid() + int(time() * 1000) % 2**32)

    # File for saving the catalog
    savefile = f"{savefolder}sampled_grb_cat_{self.n_year}years_v{run_iteration}.txt"
    saveplotfile = f"{savefolder}sampled_grb_cat_{self.n_year}years.csv"

    # Setting the arrays with 0 so that there is no issue while using mcmc_condition
    l_m_flux_temp, l_p_flux_temp, l_flnc_temp = np.zeros(100), np.zeros(100), np.zeros(100)
    s_m_flux_temp, s_p_flux_temp, s_flnc_temp = np.zeros(100), np.zeros(100), np.zeros(100)
    cat_loop_long = True
    cat_loop_short = True
    print(f"Begin of longs")
    while cat_loop_long:
      if method is None:
        params = self.get_params()
      elif type(method) is list:
        params = self.get_set_params(method[run_iteration])
      else:
        raise ValueError("Wrong method used")

      l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, nlong_temp = params[0]

      if thread_number == 'all':
        with mp.Pool() as pool:
          l_temp_ret = np.array(pool.starmap(self.get_long, zip(range(nlong_temp), repeat(l_rate_temp), repeat(l_ind1_z_temp), repeat(l_ind2_z_temp), repeat(l_zb_temp), repeat(l_ind1_temp), repeat(l_ind2_temp), repeat(l_lb_temp))))
      elif type(thread_number) is int and thread_number > 1:
        with mp.Pool(thread_number) as pool:
          l_temp_ret = np.array(pool.starmap(self.get_long, zip(range(nlong_temp), repeat(l_rate_temp), repeat(l_ind1_z_temp), repeat(l_ind2_z_temp), repeat(l_zb_temp), repeat(l_ind1_temp), repeat(l_ind2_temp), repeat(l_lb_temp))))
      else:
        l_temp_ret = np.array([self.get_long(ite_long, l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp) for ite_long in range(nlong_temp)])

      l_m_flux_temp, l_p_flux_temp, l_flnc_temp = np.array(l_temp_ret[:, 4], dtype=np.float64), np.array(l_temp_ret[:, 5], dtype=np.float64), np.array(l_temp_ret[:, 7], dtype=np.float64)
      condition_long = self.mcmc_condition(l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, params=params, mode="l_pflx", n_sig=n_sig)
      if condition_long[0]:
        print(f"Long catalog fitting : chi2 = {condition_long[1]}")
        cat_loop_long = False
      else:
        print(f"Long catalog not fitting  : chi2 = {condition_long[1]} > len(ref) * {n_sig} sigma  -   trying again")
    print(f"Long finished     [ite {run_iteration}]")

    print(f"Begin of shorts")
    while cat_loop_short:
      if method is None:
        params = self.get_params()
      elif type(method) is list:
        params = self.get_set_params(method[run_iteration])
      else:
        raise ValueError("Wrong method used")

      s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, nshort_temp = params[1]

      if thread_number == 'all':
        with mp.Pool() as pool:
          s_temp_ret = np.array(pool.starmap(self.get_short, zip(range(nshort_temp), repeat(s_rate_temp), repeat(s_ind1_z_temp), repeat(s_ind2_z_temp), repeat(s_zb_temp), repeat(s_ind1_temp), repeat(s_ind2_temp), repeat(s_lb_temp))))
      elif type(thread_number) is int and thread_number > 1:
        with mp.Pool(thread_number) as pool:
          s_temp_ret = np.array(pool.starmap(self.get_short, zip(range(nshort_temp), repeat(s_rate_temp), repeat(s_ind1_z_temp), repeat(s_ind2_z_temp), repeat(s_zb_temp), repeat(s_ind1_temp), repeat(s_ind2_temp), repeat(s_lb_temp))))
      else:
        s_temp_ret = np.array([self.get_short(ite_short, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp) for ite_short in range(nshort_temp)])

      s_m_flux_temp, s_p_flux_temp, s_flnc_temp = np.array(s_temp_ret[:, 4], dtype=np.float64), np.array(s_temp_ret[:, 5], dtype=np.float64), np.array(s_temp_ret[:, 7], dtype=np.float64)
      condition_short = self.mcmc_condition(l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, params=params, mode="s_pflx", n_sig=n_sig)
      if condition_short[0]:
        print(f"Short catalog fitting : chi2 = {condition_short[1]}")
        cat_loop_short = False
      else:
        print(f"Short catalog not fitting  : chi2 = {condition_short[1]} > len(ref) * {n_sig} sigma  -   trying again")
    print(f"Short finished     [ite {run_iteration}]")

    # Saving the catalog
    all_grb = np.concatenate((l_temp_ret, s_temp_ret))
    with open(savefile, "w") as f:
      f.write(f"Catalog of synthetic GRBs sampled over {self.n_year} years. Based on differents works, see catalogMC.py for more details\n")
      f.write(f"Parameters l_rate, l_ind1_z, l_ind2_z, l_zb, l_ind1, l_ind2, l_lb, s_rate, s_ind1_z, s_ind2_z, s_zb, s_ind1, s_ind2, s_lb : {l_rate_temp} - {l_ind1_z_temp} - {l_ind2_z_temp} - {l_zb_temp} - {l_ind1_temp} - {l_ind2_temp} - {l_lb_temp} - {s_rate_temp} - {s_ind1_z_temp} - {s_ind2_z_temp} - {s_zb_temp} - {s_ind1_temp} - {s_ind2_temp} - {s_lb_temp}\n")
      f.write("Keys and units : \n")
      f.write("name|t90|light curve name|fluence|mean flux|peak flux|redshift|Band low energy index|Band high energy index|peak energy|luminosity distance|isotropic luminosity|isotropic energy|jet opening angle\n")
      f.write("[dimensionless] | [s] | [dimensionless] | [ph/cm2] | [ph/cm2/s] | [ph/cm2/s] | [dimensionless] | [dimensionless] | [dimensionless] | [keV] | [Gpc] | [erg/s] | [erg] | [°]\n")

    for sample_number, line in enumerate(all_grb):
      if line[13] == "Sample short":
        self.save_grb(savefile, f"sGRB{self.n_year}S{sample_number}", line[6], line[8], line[7], line[4], line[5], line[0], line[9], line[10], line[2], line[11], line[3], line[12], 0)
      elif line[13] == "Sample long":
        self.save_grb(savefile, f"lGRB{self.n_year}S{sample_number}", line[6], line[8], line[7], line[4], line[5], line[0], line[9], line[10], line[2], line[11], line[3], line[12], 0)
      else:
        raise ValueError(f"Error while making the sample, the Type of the burst should be 'Sample short' or 'Sample long' but value is {line[13]}")

    print("Deleting the old source file if it exists : ")
    if f"{int(self.n_year)}sample" in os.listdir("./sources/SampledSpectra"):
      subprocess.call(f"rm -r ./sources/SampledSpectra/{int(self.n_year)}sample", shell=True)
      # os.rmdir(f"./sources/SampledSpectra/{int(n_year)}sample")
    print("Deletion done")

    condition = self.mcmc_condition(l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, params=params, mode="pflx", n_sig=n_sig)
    if condition[0]:
      row = [l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(condition[1], 3), "Accepted"]
    else:
      row = [l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(condition[1], 3), "Rejected"]
      print(f"Rejected : pearson chi2 = {np.around(condition[1], 3)}     [ite {run_iteration}]")

    list_param = l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp
    self.hist_plotter(run_iteration, [l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, np.around(condition[1], 3)], list_param, comment=comment, savefile=saveplotfile)

    return row

  def mcmc_condition(self, l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, params=None, mode="pflx", n_sig=1):
    """
    Condition on the histograms to consider a value is correct
    """
    smp_pflux_l_hist = np.histogram(l_p_flux_temp, bins=self.bin_flux_l[self.nfluxbin_l[0]:])[0]
    smp_pflux_s_hist = np.histogram(s_p_flux_temp, bins=self.bin_flux_s[self.nfluxbin_s[0]:])[0]
    smp_flnc_l_hist = np.histogram(l_flnc_temp, bins=self.bin_flnc_l[self.nflncbin_l[0]:])[0]
    smp_flnc_s_hist = np.histogram(s_flnc_temp, bins=self.bin_flnc_s[self.nflncbin_s[0]:])[0]

    # smp_pflux_l_hist_norm = smp_pflux_l_hist * np.sum(self.l_pflux_bins) / np.sum(smp_pflux_l_hist)
    # smp_pflux_s_hist_norm = smp_pflux_s_hist * np.sum(self.s_pflux_bins) / np.sum(smp_pflux_s_hist)
    # smp_flnc_l_hist_norm = smp_flnc_l_hist * np.sum(self.l_flnc_bins) / np.sum(smp_flnc_l_hist)
    # smp_flnc_s_hist_norm = smp_flnc_s_hist * np.sum(self.s_flnc_bins) / np.sum(smp_flnc_s_hist)

    if mode == "pflx":
      obs_dat = np.concatenate((smp_pflux_l_hist, smp_pflux_s_hist))
      expect_dat = np.concatenate((self.l_pflux_bins, self.s_pflux_bins))
      end_pflx_ratio = smp_pflux_l_hist[-1] / self.l_pflux_bins[-1]
    elif mode == "l_pflx":
      obs_dat = smp_pflux_l_hist
      expect_dat = self.l_pflux_bins
      end_pflx_ratio = smp_pflux_l_hist[-1] / self.l_pflux_bins[-1]
    elif mode == "s_pflx":
      obs_dat = smp_pflux_s_hist
      expect_dat = self.s_pflux_bins
      end_pflx_ratio = 1
    else:
      raise ValueError("Invalid mode for mcmc_condition")

    # cond_test = chisquare(obs_dat, f_exp=expect_dat, ddof=0)
    # print("obs", obs_dat)
    # print("exp", expect_dat)
    # chi2 limit is simply np.sum((obs_dat-expect_dat)**2/expect_dat) with obs_dat = expect_dat + n np.sqrt(expect_dat) (1 sigma error for poisson distributed variable)
    chi2_lim = len(expect_dat) * n_sig**2
    chi2 = np.sum((obs_dat-expect_dat)**2/expect_dat)
    return chi2 < chi2_lim, chi2, end_pflx_ratio

  def save_grb(self, filename, name, t90, lcname, fluence, mean_flux, peak_flux, red, band_low, band_high, ep, dl, lpeak, eiso, thetaj):
    """
    Saves a GRB in a catalog file
    """
    with open(filename, "a") as f:
      # print(f"{name}|{t90}|{lcname}|{fluence}|{mean_flux}|{peak_flux}|{red}|{band_low}|{band_high}|{ep}|{dl}|{lpeak}|{eiso}|{thetaj}\n")
      f.write(f"{name}|{t90}|{lcname}|{fluence}|{mean_flux}|{peak_flux}|{red}|{band_low}|{band_high}|{ep}|{dl}|{lpeak}|{eiso}|{thetaj}\n")

  def hist_plotter(self, iteration, histos, params, comment="", savefile=None):
    if params is not None:
      title = f"{comment}\n{params[0:7]}\n{params[7:]}\nPierson chi2 of pflx : {histos[6]}"
    else:
      title = f"{comment}\nPierson chi2 of pflx : {histos[6]}"

    yscale = "log"
    fig1, ((ax1l, ax2l, ax3l), (ax1l2, ax2l2, ax3l2), (ax1s, ax2s, ax3s), (ax1s2, ax2s2, ax3s2)) = plt.subplots(nrows=4, ncols=3, figsize=(20, 12))
    fig1.suptitle(title)

    ax1l.hist(self.gbm_l_pflux, bins=self.bin_flux_l, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_l_pflux))
    ax1l.hist(histos[1], bins=self.bin_flux_l, histtype="step", color="blue", label="Sample")
    ax1l.axvline(self.flux_lim[0])
    ax1l.axvline(self.flux_lim[1])
    ax1l.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1l.legend()

    ax2l.hist(self.gbm_l_mflux, bins=self.usual_bins, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_l_mflux))
    ax2l.hist(histos[0], bins=self.usual_bins, histtype="step", color="blue", label="Sample")
    ax2l.set(xlabel="mflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax2l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2l.legend()

    ax3l.hist(self.gbm_l_flnc, bins=self.bin_flnc_l, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_l_flnc))
    ax3l.hist(histos[2], bins=self.bin_flnc_l, histtype="step", color="blue", label="Sample")
    ax3l.axvline(self.flnc_l_lim[0])
    ax3l.axvline(self.flnc_l_lim[1])
    ax3l.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax3l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax3l.legend()

    ax1l2.hist(self.gbm_l_pflux, bins=self.usual_bins, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_l_pflux))
    ax1l2.hist(histos[1], bins=self.usual_bins, histtype="step", color="blue", label="Sample")
    ax1l2.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1l2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1l2.legend()

    # print(np.array(histos[0]))
    # print(np.array(histos[1]))
    ax2l2.hist(self.gbm_l_mp_ratio, bins=30, histtype="step", color="red", label="GBM", weights=[1/len(self.gbm_l_mp_ratio)] * len(self.gbm_l_mp_ratio))
    ax2l2.hist(np.array(histos[1]) / np.array(histos[0]), bins=np.logspace(0, 3, 30), histtype="step", color="blue", label="Sample",  weights=[1/len(histos[1])] * len(histos[1]))
    ax2l2.set(xlabel="p/m ratio lGRB", ylabel="proportion over full population", xscale="log", yscale="linear")
    ax2l2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2l2.legend()

    ax3l2.hist(self.gbm_l_flnc, bins=self.usual_bins, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_l_flnc))
    ax3l2.hist(histos[2], bins=self.usual_bins, histtype="step", color="blue", label="Sample")
    ax3l2.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax3l2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax3l2.legend()

    ax1s.hist(self.gbm_s_pflux, bins=self.bin_flux_s, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_s_pflux))
    ax1s.hist(histos[4], bins=self.bin_flux_s, histtype="step", color="blue", label="Sample")
    ax1s.axvline(self.flux_lim[0])
    ax1s.axvline(self.flux_lim[1])
    ax1s.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1s.legend()

    ax2s.hist(self.gbm_s_mflux, bins=self.usual_bins, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_s_mflux))
    ax2s.hist(histos[3], bins=self.usual_bins, histtype="step", color="blue", label="Sample")
    ax2s.set(xlabel="mflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax2s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2s.legend()

    ax3s.hist(self.gbm_s_flnc, bins=self.bin_flnc_s, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_s_flnc))
    ax3s.hist(histos[5], bins=self.bin_flnc_s, histtype="step", color="blue", label="Sample")
    ax3s.axvline(self.flnc_s_lim[0])
    ax3s.axvline(self.flnc_s_lim[1])
    ax3s.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax3s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax3s.legend()

    ax1s2.hist(self.gbm_s_pflux, bins=self.usual_bins, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_s_pflux))
    ax1s2.hist(histos[4], bins=self.usual_bins, histtype="step", color="blue", label="Sample")
    ax1s2.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1s2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1s2.legend()

    ax2s2.hist(self.gbm_s_mp_ratio, bins=30, histtype="step", color="red", label="GBM", weights=[1/len(self.gbm_s_mp_ratio)] * len(self.gbm_s_mp_ratio))
    ax2s2.hist(np.array(histos[4]) / np.array(histos[3]), bins=np.logspace(0, 3, 30), histtype="step", color="blue", label="Sample",  weights=[1/len(histos[4])] * len(histos[4]))
    ax2s2.set(xlabel="p/m ratio sGRB", ylabel="proportion over full population", xscale="log", yscale="linear")
    ax2s2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2s2.legend()

    ax3s2.hist(self.gbm_s_flnc, bins=self.usual_bins, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_s_flnc))
    ax3s2.hist(histos[5], bins=self.usual_bins, histtype="step", color="blue", label="Sample")
    ax3s2.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax3s2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax3s2.legend()

    if savefile is not None:
      # if np.isinf(histos[6]):
      #   pval_suf = "-inf"
      # else:
      #   pval_suf = int(histos[6])
      # plt.savefig(f"{savefile.split('.csv')[0]}_{iteration}_{pval_suf}_{int(histos[7])}")
      plt.savefig(f"{savefile.split('.csv')[0]}_{iteration}_{int(histos[6])}")
    plt.close(fig1)

    fig2, ((axv21, axv22), (axv23, axv24)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
    fig2.suptitle(title)

    axv21.hist(self.gbm_l_pflux, bins=self.bin_flux_l, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_l_pflux))
    axv21.hist(histos[1], bins=self.bin_flux_l, histtype="step", color="blue", label="Sample")
    axv21.axvline(self.flux_lim[0])
    axv21.axvline(self.flux_lim[1])
    axv21.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    axv21.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    axv21.legend()

    axv22.hist(self.gbm_s_pflux, bins=self.bin_flux_s, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_s_pflux))
    axv22.hist(histos[4], bins=self.bin_flux_s, histtype="step", color="blue", label="Sample")
    axv22.axvline(self.flux_lim[0])
    axv22.axvline(self.flux_lim[1])
    axv22.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    axv22.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    axv22.legend()

    axv23.hist(self.gbm_l_pflux, bins=self.usual_bins, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_l_pflux))
    axv23.hist(histos[1], bins=self.usual_bins, histtype="step", color="blue", label="Sample")
    axv23.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    axv23.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    axv23.legend()

    axv24.hist(self.gbm_s_pflux, bins=self.usual_bins, histtype="step", color="red", label="GBM", weights=[self.gbm_weight] * len(self.gbm_s_pflux))
    axv24.hist(histos[4], bins=self.usual_bins, histtype="step", color="blue", label="Sample")
    axv24.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    axv24.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    axv24.legend()

    if savefile is not None:
      plt.savefig(f"{savefile.split('.csv')[0]}_compact_{iteration}_{int(histos[6])}")
    plt.close(fig2)


  def get_short(self, ite_num, short_rate, ind1_z_s, ind2_z_s, zb_s, ind1_s, ind2_s, lb_s):
    """
    Creates the quatities of a short burst according to distributions
    Based on Lana Salmon's thesis and Ghirlanda et al, 2016
    """
    np.random.seed(os.getpid() + int(time() * 1000) % 2 ** 32)
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    z_obs_temp = acc_reject(red_rate_short, [short_rate, ind1_z_s, ind2_z_s, zb_s], self.zmin, self.zmax)
    # lpeak_rest_temp = acc_reject(broken_plaw, [ind1_s, ind2_s, lb_s], self.lmin, self.lmax)
    lpeak_rest_temp = transfo_broken_plaw(ind1_s, ind2_s, lb_s, self.lmin, self.lmax)

    band_low_obs_temp, band_high_obs_temp = pick_normal_alpha_beta(self.band_low_s_mu, self.band_low_s_sig, self.band_high_s_mu, self.band_high_s_sig)

    t90_obs_temp = 1000
    while t90_obs_temp > 2:
      t90_obs_temp = 10 ** np.random.normal(-0.2373, 0.4058)

    lc_temp, gbm_mflux, gbm_pflux = self.closest_lc(t90_obs_temp)[:3]
    pflux_to_mflux = gbm_mflux / gbm_pflux
    # pflux_to_mflux = pflux_to_mflux_calculator(lc_temp, gbm_t90)

    dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
    ep_rest_temp = yonetoku_reverse_short(lpeak_rest_temp)
    ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)
    eiso_rest_temp = amati_short(ep_rest_temp)

    ##################################################################################################################
    # Calculation of spectrum and data saving
    ##################################################################################################################
    ener_range = np.logspace(1, 3, 10001)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    # temp_peak_flux OBSERVED and on this energy range
    temp_mean_flux = temp_peak_flux * pflux_to_mflux

    return [z_obs_temp, ep_obs_temp, ep_rest_temp, lpeak_rest_temp, temp_mean_flux, temp_peak_flux, t90_obs_temp, temp_mean_flux * t90_obs_temp, lc_temp, band_low_obs_temp, band_high_obs_temp, dl_obs_temp,
            eiso_rest_temp, "Sample short", "Sample"]

  def get_long(self, ite_num, long_rate, ind1_z_l, ind2_z_l, zb_l, ind1_l, ind2_l, lb_l):
    """
    Creates the quatities of a long burst according to distributions
    Based on Sarah Antier's thesis
    """
    np.random.seed(os.getpid() + int(time() * 1000) % 2 ** 32)
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    # timelist = []
    # init_time = time()
    z_obs_temp = acc_reject(red_rate_long, [long_rate, ind1_z_l, ind2_z_l, zb_l], self.zmin, self.zmax)

    # timelist.append(time() - init_time)
    # init_time = time()
    # lpeak_rest_temp = acc_reject(broken_plaw, [ind1_l, ind2_l, lb_l], self.lmin, self.lmax)
    lpeak_rest_temp = transfo_broken_plaw(ind1_l, ind2_l, lb_l, self.lmin, self.lmax)
    # timelist.append(time() - init_time)
    # init_time = time()
    band_low_obs_temp, band_high_obs_temp = pick_normal_alpha_beta(self.band_low_l_mu, self.band_low_l_sig, self.band_high_l_mu, self.band_high_l_sig)
    # timelist.append(time() - init_time)
    # init_time = time()
    t90_obs_temp = 0
    while t90_obs_temp <= 2:
      t90_obs_temp = 10 ** np.random.normal(1.4438, 0.4956)
    # timelist.append(time() - init_time)

    lc_temp, gbm_mflux, gbm_pflux = self.closest_lc(t90_obs_temp)[:3]
    pflux_to_mflux = gbm_mflux / gbm_pflux
    # pflux_to_mflux = pflux_to_mflux_calculator(lc_temp, gbm_t90)

    dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
    ep_rest_temp = yonetoku_reverse_long(lpeak_rest_temp)
    # init_time = time()
    ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)
    eiso_rest_temp = amati_long(ep_rest_temp)

    ##################################################################################################################
    # Calculation of spectrum and data saving
    ##################################################################################################################
    ener_range = np.logspace(1, 3, 10001)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    temp_mean_flux = temp_peak_flux * pflux_to_mflux

    # timelist.append(time() - init_time)
    # init_time = time()
    # for times in timelist:
    #   print(f"Time taken : {times:8.6f}s making {times/np.sum(timelist)*100:5.2f}% of the run")
    return [z_obs_temp, ep_obs_temp, ep_rest_temp, lpeak_rest_temp, temp_mean_flux, temp_peak_flux, t90_obs_temp, temp_mean_flux * t90_obs_temp, lc_temp, band_low_obs_temp, band_high_obs_temp, dl_obs_temp,
            eiso_rest_temp, "Sample long", "Sample"]

  def closest_lc(self, searched_time):
    """
    Find the lightcurve file with a duration which is the closest to the sampled t90 time
    """
    abs_diff = np.abs(np.array(self.gbm_cat.df.t90, dtype=float) - searched_time)
    gbm_indexes = np.where(abs_diff == np.min(abs_diff))[0]
    # print(gbm_indexes)
    if len(gbm_indexes) == 0:
      raise ValueError("No GRB found for the closest GRB duration")
    elif len(gbm_indexes) == 1:
      gbm_index = gbm_indexes[0]
    else:
      gbm_index = gbm_indexes[np.random.randint(len(gbm_indexes))]
    return f"LightCurve_{self.gbm_cat.df.name[gbm_index]}.dat", self.gbm_cat.df.mean_flux[gbm_index], self.gbm_cat.df.peak_flux[gbm_index], self.gbm_cat.df.t90[gbm_index]

  def gbm_reference_distri(self, print_bins=True):
    """
    Creates the GBM distribution used for estimating the sample parameters. Distributions are obtained with a kde method on the GBM datasets
    """
    if print_bins:
      pflux_l_hist = np.histogram(self.gbm_l_pflux, bins=self.bin_flux_l, weights=[self.gbm_weight] * len(self.gbm_l_pflux))
      pflux_s_hist = np.histogram(self.gbm_s_pflux, bins=self.bin_flux_s, weights=[self.gbm_weight] * len(self.gbm_s_pflux))
      flnc_l_hist = np.histogram(self.gbm_l_flnc, bins=self.bin_flnc_l, weights=[self.gbm_weight] * len(self.gbm_l_flnc))
      flnc_s_hist = np.histogram(self.gbm_s_flnc, bins=self.bin_flnc_s, weights=[self.gbm_weight] * len(self.gbm_s_flnc))
      print(f"\n== Peakflux for lGRBs ==")
      for ite in range(len(pflux_l_hist[0])):
        print(f"Bin from {pflux_l_hist[1][ite]:8.4f} to {pflux_l_hist[1][ite+1]:8.4f}  : {pflux_l_hist[0][ite]}")
      print(f"Low bins count : {self.l_low_pflux_bins}")
      print(f"High bins count : {self.l_high_pflux_bins}")

      print(f"\n== Peakflux for sGRBs ==")
      for ite in range(len(pflux_s_hist[0])):
        print(f"Bin from {pflux_s_hist[1][ite]:8.4f} to {pflux_s_hist[1][ite+1]:8.4f}  : {pflux_s_hist[0][ite]}")
      print(f"Low bins count : {self.s_low_pflux_bins}")
      print(f"High bins count : {self.s_high_pflux_bins}")

      print(f"\n== Fluence for lGRBs ==")
      for ite in range(len(flnc_l_hist[0])):
        print(f"Bin from {flnc_l_hist[1][ite]:8.4f} to {flnc_l_hist[1][ite+1]:8.4f}  : {flnc_l_hist[0][ite]}")
      print(f"Low bins count : {self.l_low_flnc_bins}")
      print(f"High bins count : {self.l_high_flnc_bins}")

      print(f"\n== Fluence for sGRBs ==")
      for ite in range(len(flnc_s_hist[0])):
        print(f"Bin from {flnc_s_hist[1][ite]:8.4f} to {flnc_s_hist[1][ite+1]:8.4f}  : {flnc_s_hist[0][ite]}")
      print(f"Bins count : {self.s_flnc_bins}")

    yscale = "log"
    fig1, ((ax1l, ax2l, ax3l), (ax1s, ax2s, ax3s)) = plt.subplots(nrows=2, ncols=3, figsize=(20, 6))
    tt = ax1l.hist(self.gbm_l_pflux, bins=self.bin_flux_l, histtype="step", label="all_l", weights=[self.gbm_weight] * len(self.gbm_l_pflux))
    ax1l.axvline(self.flux_lim[0])
    ax1l.axvline(self.flux_lim[1])
    ax1l.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1l.legend()

    ax2l.hist(self.gbm_l_mflux, bins=self.usual_bins, histtype="step", label="all_l", weights=[self.gbm_weight] * len(self.gbm_l_mflux))
    ax2l.set(xlabel="mflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax2l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2l.legend()

    ax3l.hist(self.gbm_l_flnc, bins=self.bin_flnc_l, histtype="step", label="all_l", weights=[self.gbm_weight] * len(self.gbm_l_flnc))
    ax3l.axvline(self.flnc_l_lim[0])
    ax3l.axvline(self.flnc_l_lim[1])
    ax3l.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax3l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax3l.legend()

    ax1s.hist(self.gbm_s_pflux, bins=self.bin_flux_s, histtype="step", label="all_s", weights=[self.gbm_weight] * len(self.gbm_s_pflux))
    ax1s.axvline(self.flux_lim[0])
    ax1s.axvline(self.flux_lim[1])
    ax1s.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1s.legend()

    ax2s.hist(self.gbm_s_mflux, bins=self.usual_bins, histtype="step", label="all_s", weights=[self.gbm_weight] * len(self.gbm_s_mflux))
    ax2s.set(xlabel="mflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax2s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2s.legend()

    ax3s.hist(self.gbm_s_flnc, bins=self.bin_flnc_s, histtype="step", label="all_s", weights=[self.gbm_weight] * len(self.gbm_s_flnc))
    ax3s.axvline(self.flnc_s_lim[0])
    ax3s.axvline(self.flnc_s_lim[1])
    ax3s.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax3s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax3s.legend()
    plt.show()

# from catalogMC import *
# testcat = MCCatalog(mode="mc")
# testcat = MCCatalog(mode="catalog")

# from catalogMC import *
# import matplotlib as mpl
# mpl.use("Qt5Agg")
#
# file = "./Sampled/mclongv9-300/mc_fit.csv"
# leg_mode = "fine"
# MC_explo_pairplot(file, leg_mode, grbtype="short")
# plt.show()