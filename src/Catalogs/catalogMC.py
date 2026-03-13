# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  catalogMC.py
# Contains functions and class to create a synthetic GRB catalogue using Monte Carlo methods
# ================================================================

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import subprocess
from itertools import repeat
from time import time

from src.Catalogs.catalog import Catalog
from src.General.funcmod import calc_flux_gbm, use_scipyquad, equi_distri, red_rate_long, red_rate_short, acc_reject, transfo_broken_plaw, pick_normal_alpha_beta, norm_band_spec_calc, amati_long, amati_short, yonetoku_reverse_long, yonetoku_reverse_short, pflux_to_mflux_calculator
from astropy.cosmology import FlatLambdaCDM


def categorize_pierson_chi2(pierson_chi2_array, mode="fine", grbtype="long"):
  """
  Assigns each Pearson chi2 value in an array to a quality category based on
  sigma-equivalent thresholds, and builds label strings for use as seaborn hue
  categories in pair plots.
  :param pierson_chi2_array: np.ndarray, array of Pearson chi2 values to categorize
  :param mode: str, threshold set to use; one of "fine", "medium_fine",
      "medium_coarse", "coarse", or "very_coarse"; controls how tightly simulations
      are classified, default="fine"
  :param grbtype: str, GRB population being evaluated; "long" uses 12 bins and
      "short" uses 8 bins for chi2 limit computation, default="long"
  :returns: np.ndarray, str, array of category label strings (one per input
      value) to use as a hue column, and the ordered list of unique category
      labels for the hue_order argument
  :raises ValueError: if mode is not one of the five accepted values
  """
  if grbtype == "long":
    nbins = 12
  else:
    nbins = 8
  if mode == "fine":
    siglim = np.array([0.7, 1, 1.5])
    chilim = np.around(nbins * siglim**2, 3)
    f1 = np.sum(np.where(pierson_chi2_array >= 0, np.where(pierson_chi2_array < chilim[0], 1, 0), 0))
    f2 = np.sum(np.where(pierson_chi2_array >= chilim[0], np.where(pierson_chi2_array < chilim[1], 1, 0), 0))
    f3 = np.sum(np.where(pierson_chi2_array >= chilim[1], np.where(pierson_chi2_array < chilim[2], 1, 0), 0))
    f4 = np.sum(np.where(pierson_chi2_array >= chilim[2], 1, 0))

  elif mode == "medium_fine":
    siglim = np.array([1, 1.5, 2])
    chilim = np.around(nbins * siglim**2, 3)
    f1 = np.sum(np.where(pierson_chi2_array >= 0, np.where(pierson_chi2_array < chilim[0], 1, 0), 0))
    f2 = np.sum(np.where(pierson_chi2_array >= chilim[0], np.where(pierson_chi2_array < chilim[1], 1, 0), 0))
    f3 = np.sum(np.where(pierson_chi2_array >= chilim[1], np.where(pierson_chi2_array < chilim[2], 1, 0), 0))
    f4 = np.sum(np.where(pierson_chi2_array >= chilim[2], 1, 0))

  elif mode == "medium_coarse":
    siglim = np.array([1.5, 2, 3])
    chilim = np.around(nbins * siglim**2, 3)
    f1 = np.sum(np.where(pierson_chi2_array >= 0, np.where(pierson_chi2_array < chilim[0], 1, 0), 0))
    f2 = np.sum(np.where(pierson_chi2_array >= chilim[0], np.where(pierson_chi2_array < chilim[1], 1, 0), 0))
    f3 = np.sum(np.where(pierson_chi2_array >= chilim[1], np.where(pierson_chi2_array < chilim[2], 1, 0), 0))
    f4 = np.sum(np.where(pierson_chi2_array >= chilim[2], 1, 0))

  elif mode == "coarse":
    siglim = np.array([2, 3, 5])
    chilim = np.around(nbins * siglim**2, 3)
    f1 = np.sum(np.where(pierson_chi2_array >= 0, np.where(pierson_chi2_array < chilim[0], 1, 0), 0))
    f2 = np.sum(np.where(pierson_chi2_array >= chilim[0], np.where(pierson_chi2_array < chilim[1], 1, 0), 0))
    f3 = np.sum(np.where(pierson_chi2_array >= chilim[1], np.where(pierson_chi2_array < chilim[2], 1, 0), 0))
    f4 = np.sum(np.where(pierson_chi2_array >= chilim[2], 1, 0))

  elif mode == "very_coarse":
    siglim = np.array([3, 5, 9])
    chilim = np.around(nbins * siglim ** 2, 3)
    f1 = np.sum(np.where(pierson_chi2_array >= 0, np.where(pierson_chi2_array < chilim[0], 1, 0), 0))
    f2 = np.sum(np.where(pierson_chi2_array >= chilim[0], np.where(pierson_chi2_array < chilim[1], 1, 0), 0))
    f3 = np.sum(np.where(pierson_chi2_array >= chilim[1], np.where(pierson_chi2_array < chilim[2], 1, 0), 0))
    f4 = np.sum(np.where(pierson_chi2_array >= chilim[2], 1, 0))

  else:
    raise ValueError("Wrong name for mode")

  c1 = f"0 - {siglim[0]} $\sigma$  |  {f1} sims\n  {0} <= $\chi^2$ < {chilim[0]} "
  c2 = f"{siglim[0]} - {siglim[1]} $\sigma$  |  {f2} sims\n  {chilim[0]} <= $\chi^2$ < {chilim[1]} "
  c3 = f"{siglim[1]} - {siglim[2]} $\sigma$  |  {f3} sims\n  {chilim[1]} <= $\chi^2$ < {chilim[2]} "
  c4 = f">= {siglim[2]} $\sigma$  |  {f4} sims\n  $\chi^2$ >= {chilim[2]} "
  categorized = np.where(pierson_chi2_array >= chilim[2], c4, np.where(pierson_chi2_array >= chilim[1], c3, np.where(pierson_chi2_array >= chilim[0], c2, c1)))
  hue_order = [c1, c2, c3, c4]

  return categorized, hue_order


def get_df(select_col, csvfile="../Data/CatData/CatSampling/longred_lum_discreet/longfit_red.csv"):
  """
  Reads a CSV file and returns the selected columns as a pandas DataFrame.
  :param select_col: list, list of str column names to extract from the CSV file
  :param csvfile: str, path to the CSV file to read,
      default="../Data/CatData/CatSampling/longred_lum_discreet/longfit_red.csv"
  :returns: pd.DataFrame, DataFrame containing only the requested columns
  """
  result_df = pd.read_csv(csvfile)
  return result_df[select_col]


def MC_explo_pairplot(fileused, legend_mode, grbtype):
  """
  Reads Monte Carlo exploration results from a CSV file, categorizes each run
  by its Pearson chi2 value, and draws a seaborn pair plot coloured by quality
  category. Handles CSV files that may or may not contain pre-computed GRB counts
  (nlong / nshort), recomputing them from the rate parameters if absent.
  :param fileused: str, path to the CSV file containing MC exploration results
  :param legend_mode: str, threshold mode passed to categorize_pierson_chi2();
      one of "fine", "medium_fine", "medium_coarse", "coarse", "very_coarse"
  :param grbtype: str, GRB population to plot; "long" or "short"
  """
  if grbtype == "long":
    extract_cols = ["nlong", "long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "long_ind1_lum", "long_ind2_lum", "long_lb", "pierson_chi2"]
    select_cols = ["nlong", "long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "long_ind1_lum", "long_ind2_lum", "long_lb_2"]
  elif grbtype == "short":
    extract_cols = ["nshort", "short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "short_ind1_lum", "short_ind2_lum", "short_lb", "pierson_chi2"]
    select_cols = ["nshort", "short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "short_ind1_lum", "short_ind2_lum", "short_lb"]

  try:
    df_selec = get_df(extract_cols, csvfile=fileused)
  except KeyError:
    if grbtype == "long":
      extract_cols = ["long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "long_ind1_lum", "long_ind2_lum", "long_lb", "pierson_chi2"]
      select_cols = ["long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "long_ind1_lum", "long_ind2_lum", "long_lb_2"]
      # select_cols = ["long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "long_ind1_lum", "long_ind2_lum", "long_lb"]
    elif grbtype == "short":
      extract_cols = ["short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "short_ind1_lum", "short_ind2_lum", "short_lb", "pierson_chi2"]
      select_cols = ["short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "short_ind1_lum", "short_ind2_lum", "short_lb"]
    df_selec = get_df(extract_cols, csvfile=fileused)
    if grbtype == "long":
      df_selec['nlong'] = [int(10 * use_scipyquad(red_rate_long, 0, 10, func_args=(df_selec.long_rate.values[ite], df_selec.long_ind1_z.values[ite], df_selec.long_ind2_z.values[ite], df_selec.long_zb.values[ite]), x_logscale=False)[0]) for ite in range(len(df_selec))]
      select_cols = ["nlong", "long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "long_ind1_lum", "long_ind2_lum", "long_lb_2"]
    elif grbtype == "short":
      df_selec['nshort'] = [int(10 * use_scipyquad(red_rate_short, 0, 10, func_args=(df_selec.short_rate.values[ite], df_selec.short_ind1_z.values[ite], df_selec.short_ind2_z.values[ite], df_selec.short_zb.values[ite]), x_logscale=False)[0]) for ite in range(len(df_selec))]
      select_cols = ["nshort", "short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "short_ind1_lum", "short_ind2_lum", "short_lb"]

  pierson_chi2_categories, order_hue = categorize_pierson_chi2(df_selec['pierson_chi2'].values, mode=legend_mode, grbtype=grbtype)

  df_selec['pierson_chi2_category'] = pierson_chi2_categories
  if grbtype == "long":
    df_selec['long_lb_2'] = df_selec['long_lb'] / 1e51

  sns.pairplot(df_selec.sort_values(by="pierson_chi2", ascending=False), hue="pierson_chi2_category", vars=select_cols, corner=False, plot_kws={'s': 20}, palette="rainbow_r")


class MCCatalog:
  """
  Monte Carlo GRB catalog generator. Builds synthetic short and long GRB
  populations by drawing redshifts and luminosities from broken power-law
  distributions and spectra from Band function parameters, then accepts or
  rejects parameter sets based on a Pearson chi2 comparison with the observed
  GBM peak-flux and fluence distributions. Supports three operating modes:
  "catalog" (generate accepted synthetic catalogs), "mc" (explore parameter
  space), and "parametrized" (run with a fixed parameter list).
  """
  def __init__(self, gbm_file="../Data/CatData/allGBM.txt", sttype=None, rf_file="../Data/CatData/rest_frame_properties.txt", mode="catalog"):
    """
    Initialises the MCCatalog by loading the GBM reference catalog, computing
    observed flux and fluence histograms for short and long GRBs, setting the
    parameter space boundaries, and immediately running the selected mode.
    :param gbm_file: str, path to the GBM catalog text file,
        default="../Data/CatData/allGBM.txt"
    :param sttype: list or None, standardized file format descriptor of length 5
        for the GBM catalog file; if None, defaults to [4, newline, 5, "|", 4000]
    :param rf_file: str, path to the rest-frame properties file,
        default="../Data/CatData/rest_frame_properties.txt"
    :param mode: str, operating mode; one of "catalog", "mc", or "parametrized",
        default="catalog"
    :raises ValueError: if mode is not one of the three accepted values
    :raises NameError: if a simulation output folder with the target name already exists
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
      model = self.df_long.flnc_best_fitting_model.values[ite_l]
      p_model = self.df_long.pflx_best_fitting_model.values[ite_l]
      if type(p_model) is str:
        self.gbm_l_pflux.append(self.df_long[f"{p_model}_phtflux"].values[ite_l])
        self.gbm_l_mp_ratio.append(self.df_long[f"{p_model}_phtflux"].values[ite_l] / self.df_long[f"{model}_phtflux"].values[ite_l])
      self.gbm_l_mflux.append(self.df_long[f"{model}_phtflux"].values[ite_l])
      self.gbm_l_flnc.append(calc_flux_gbm(self.df_long, ite_l, self.ergcut, cat_is_df=True) * self.df_long.t90.values[ite_l])

    self.gbm_s_mflux = []
    self.gbm_s_pflux = []
    self.gbm_s_flnc = []
    self.gbm_s_mp_ratio = []
    for ite_s in range(len(self.df_short)):
      model = self.df_short.flnc_best_fitting_model.values[ite_s]
      p_model = self.df_short.pflx_best_fitting_model.values[ite_s]
      if type(p_model) is str:
        self.gbm_s_pflux.append(self.df_short[f"{p_model}_phtflux"].values[ite_s])
        self.gbm_s_mp_ratio.append(self.df_short[f"{p_model}_phtflux"].values[ite_s] / self.df_short[f"{model}_phtflux"].values[ite_s])
      self.gbm_s_mflux.append(self.df_short[f"{model}_phtflux"].values[ite_s])
      self.gbm_s_flnc.append(calc_flux_gbm(self.df_short, ite_s, self.ergcut, cat_is_df=True) * self.df_short.t90.values[ite_s])

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
    l_pflux_bright = np.array([12, 14, 16, 19, 22, 27, 35, 45, 57, 73, 120, 1000])
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
    # self.l_rate_min = 0.4
    # self.l_rate_max = 2.1
    # self.l_ind1_z_min = 1.5
    # self.l_ind1_z_max = 4.3
    # self.l_ind2_z_min = -2.4
    # self.l_ind2_z_max = 1
    # self.l_zb_min = 2.3
    # self.l_zb_max = 3.7
    #
    # self.s_rate_min = 0.1
    # self.s_rate_max = 1.1
    # self.s_ind1_z_min = 0.5
    # self.s_ind1_z_max = 4.1
    # self.s_ind2_z_min = 0.9
    # self.s_ind2_z_max = 4
    # self.s_zb_min = 1.7
    # self.s_zb_max = 3.3
    # # Luminosity
    # self.l_ind1_min = -1.5
    # self.l_ind1_max = -0.1
    # self.l_ind2_min = -2.1
    # self.l_ind2_max = -0.8
    # self.l_lb_min = 2e51
    # self.l_lb_max = 3e+53
    #
    # self.s_ind1_min = -1
    # self.s_ind1_max = -0.39
    # self.s_ind2_min = -3.7
    # self.s_ind2_max = -1.7
    # self.s_lb_min = 0.91e52
    # self.s_lb_max = 3.4e52

    # Narrower parameter space after studying the results of Monte Carlo
    # Redshift
    self.l_rate_min = 0.4
    self.l_rate_max = 0.7
    self.l_ind1_z_min = 2.5
    self.l_ind1_z_max = 3.1
    self.l_ind2_z_min = -2.4
    self.l_ind2_z_max = -0.8
    self.l_zb_min = 2.3
    self.l_zb_max = 3.5

    self.s_rate_min = 0.25
    self.s_rate_max = 0.7
    self.s_ind1_z_min = 1.1
    self.s_ind1_z_max = 4.1
    self.s_ind2_z_min = 1.8
    self.s_ind2_z_max = 4
    self.s_zb_min = 1.7
    self.s_zb_max = 3.3
    # Luminosity
    self.l_ind1_min = -1.35
    self.l_ind1_max = -1.25
    self.l_ind2_min = -2.1
    self.l_ind2_max = -1.7
    self.l_lb_min = 1.5e52
    self.l_lb_max = 1e53

    self.s_ind1_min = -1
    self.s_ind1_max = -0.5
    self.s_ind2_min = -3.7
    self.s_ind2_max = -1.7
    self.s_lb_min = 9.1e51
    self.s_lb_max = 3.4e52

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
    self.columns = ["nlong", "long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "nshort", "short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "long_ind1_lum", "long_ind2_lum", "long_lb", "short_ind1_lum", "short_ind2_lum",
                    "short_lb", "pierson_chi2", "status"]
    self.result_df = pd.DataFrame(columns=self.columns)

    # build_params(l_rate, l_ind1_z, l_ind2_z, l_zb, l_ind1, l_ind2, l_lb, s_rate, s_ind1_z, s_ind2_z, s_zb, s_ind1, s_ind2, s_lb)
    # param_list = [[0.42, 2.07, -0.7, 3.6, -0.36, -1.28, 1.48e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],  # Lan no evo
    #               [0.42, 2.07, -0.7, 3.6, -0.69, -1.76, 2.09e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],  # Lan empirical
    #               [0.42, 2.07, -0.7, 3.6, -0.2, -1.4, 3.16e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],    # Wanderman Piran
    #               [0.42, 2.07, -0.7, 3.6, -0.65, -3, 1.12e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52]]     # Lien

    # ==============================================================================================================================================================
    # Parameters to change
    # ==============================================================================================================================================================
    thread_num = 60
    self.mode = mode

    print("Starting")
    if self.mode == "catalog":
      param_list = None
      # Used when one cat is needed
      # par_size = 1
      # fold_name = f"cat_to_validate"
      # Used to obtain several catalogs
      iteoffsetnum = 11
      par_size = 20
      fold_name = f"multiple_cats"

      savefolder = f"../Data/CatData/CatSampling/{fold_name}/"
      sigma_number = 1

      if not (f"{fold_name}" in os.listdir("../Data/CatData/CatSampling/")):
        os.mkdir(f"../Data/CatData/CatSampling/{fold_name}")
      else:
        if os.path.exists(f"../Data/CatData/CatSampling/{fold_name}/sampled_grb_cat_{self.n_year}years_v{iteoffsetnum}.txt"):
          raise NameError("A simulation with this name already exists, please change it or delete the old simulation before running")

      print(f"Starting the catalog creation")
      if thread_num == 'all':
        print("Parallel MC execution with all threads")
      elif type(thread_num) is int and thread_num > 1:
        print(f"Parallel MC execution with {thread_num} threads")
      else:
        print(f"MC execution with 1 thread")

      rows_ret = [self.get_catalog_sample(ite, thread_num, savefolder, method=param_list, comment="", n_sig=sigma_number) for ite in range(iteoffsetnum, par_size + iteoffsetnum)]
      self.result_df = pd.DataFrame(data=rows_ret, columns=self.columns)
      self.result_df.to_csv(f"{savefolder}catalogs_fit.csv", index=False)
    else:
      print("test")
      if self.mode == "mc":
        param_list = None
        par_size = 2000
        # mctype = "long"
        mctype = "short"
        fold_name = f"mc{mctype}v2-{par_size}"
        savefile = f"../Data/CatData/CatSampling/space_explo/{fold_name}/mc_fit.csv"
      elif self.mode == "parametrized":
        # (l_rate, l_ind1_z, l_ind2_z, l_zb, l_ind1, l_ind2, l_lb, s_rate, s_ind1_z, s_ind2_z, s_zb, s_ind1, s_ind2, s_lb)
        # param_list = build_params([0.2, 0.3, 0.4, 0.5], [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6], [-1.1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4], [2, 2.6, 3.1, 3.6, 4.1, 5], -0.65, -3, 1.12e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52)
        param_list = [[0.796361, 2.242970, -1.507276, 2.294414, -0.715277, -4.629343, 1.131491e52, 0.577574, 3.107413, 1.179197, 2.318962, -0.602085, -2.148571, 1.899889e52]]
        par_size = len(param_list)
        fold_name = f"parametrizedv1-{par_size}"
        savefile = f"../Data/CatData/CatSampling/space_explo/{fold_name}/longfit_red_lum.csv"
      else:
        raise ValueError("Wrong value for mode. Only 'catalog', 'mc' and 'parametrized' are possible.")

      if not (f"space_explo" in os.listdir("../Data/CatData/CatSampling/")):
        os.mkdir(f"../Data/CatData/CatSampling/space_explo")

      if not (f"{fold_name}" in os.listdir("../Data/CatData/CatSampling/space_explo/")):
        os.mkdir(f"../Data/CatData/CatSampling/space_explo/{fold_name}")
      else:
        raise NameError("A simulation with this name already exists, please change it or delete the old simulation before running")
      self.run_mc(par_size, thread_number=thread_num, method=param_list, savefile=savefile, mctype=mctype)

  def run_mc(self, run_number, thread_number=1, method=None, savefile=None, comment="", mctype="long"):
    """
    Runs run_number Monte Carlo iterations using a multiprocessing pool (if
    thread_number > 1) or sequentially, collects the results in self.result_df,
    and optionally saves them to a CSV file.
    :param run_number: int, number of MC iterations to run
    :param thread_number: int or str, number of parallel worker processes; use
        "all" to use all available CPUs, default=1
    :param method: list or None, if None parameters are drawn randomly via
        get_params(); if a list, each iteration uses get_set_params(method[i]),
        default=None
    :param savefile: str or None, path to a CSV file where results are saved;
        if None results are not saved to disk, default=None
    :param comment: str, optional comment string appended to plot titles,
        default=""
    :param mctype: str, population used for the acceptance condition; "long"
        or "short", default="long"
    """
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
    """
    Draws a random set of long and short GRB population parameters uniformly
    from their allowed ranges and rejects combinations that do not produce a
    GRB count ratio (long/short) and absolute counts within the pre-defined
    acceptance limits. Loops until a valid parameter set is found.
    :returns: tuple, tuple of two tuples (l_params, s_params) where:
        l_params = (l_rate, l_ind1_z, l_ind2_z, l_zb, l_ind1, l_ind2, l_lb,
                    nlong) for the long GRB population
        s_params = (s_rate, s_ind1_z, s_ind2_z, s_zb, s_ind1, s_ind2, s_lb,
                    nshort) for the short GRB population
    """
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
    """
    Builds the (l_params, s_params) tuple from a fixed parameter vector,
    computing the expected GRB counts by integrating the redshift rate functions.
    Prints a warning if the resulting counts fall outside the acceptance limits.
    :param params: list, 14-element parameter vector in the order:
        [l_rate, l_ind1_z, l_ind2_z, l_zb, l_ind1, l_ind2, l_lb,
         s_rate, s_ind1_z, s_ind2_z, s_zb, s_ind1, s_ind2, s_lb]
    :returns: tuple, tuple of two tuples (l_params, s_params); same structure
        as the return value of get_params()
    """
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
    """
    Runs one full MC iteration: draws parameters, generates long and short GRB
    populations, evaluates the acceptance condition, and returns a result row.
    Also calls hist_plotter() to produce and optionally save diagnostic plots.
    Loops until the acceptance condition is met when method is None (random
    parameter draw mode).
    :param run_iteration: int, index of this iteration (used for seeding and logging)
    :param method: list or None, if None parameters are drawn randomly; if a
        list, uses method[run_iteration] as the fixed parameter vector, default=None
    :param comment: str, optional comment string appended to plot titles,
        default=""
    :param savefile: str or None, path prefix for saving diagnostic plot files;
        if None plots are not saved, default=None
    :param mctype: str, population used for the chi2 acceptance condition;
        "long" or "short", default="long"
    :returns: list, result row with columns matching self.columns:
        [nlong, l_rate, l_ind1_z, l_ind2_z, l_zb, nshort, s_rate, s_ind1_z,
         s_ind2_z, s_zb, l_ind1, l_ind2, l_lb, s_ind1, s_ind2, s_lb,
         pierson_chi2, status]
    :raises ValueError: if method is not None and not a list
    """
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
      row = [nlong_temp, l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, nshort_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(condition[1], 3), "Accepted"]
    else:
      row = [nlong_temp, l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, nshort_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(condition[1], 3), "Rejected"]
      print(f"Rejected : pearson chi2 = {np.around(condition[1], 3)}     [ite {run_iteration}]")

    list_param = l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp
    self.hist_plotter(run_iteration, [l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, np.around(condition[1], 3)], list_param, comment=comment, savefile=savefile)

    return row

  def get_catalog_sample(self, run_iteration, thread_number, savefolder, method=None, comment="", n_sig=1):
    """
    Generates one accepted synthetic GRB catalog by independently iterating on
    long and short populations until each passes the chi2 acceptance criterion,
    then writes the full catalog to a text file and produces diagnostic plots.
    :param run_iteration: int, iteration index used to name the output file and seed the RNG
    :param thread_number: int or str, number of parallel workers for GRB
        generation inside each iteration; use "all" for all CPUs
    :param savefolder: str, path to the folder where the catalog text file and
        diagnostic plots are saved
    :param method: list or None, if None parameters are drawn randomly; if a
        list, uses method[run_iteration] as the fixed parameter vector, default=None
    :param comment: str, optional comment string appended to plot titles, default=""
    :param n_sig: int or float, sigma threshold for the chi2 acceptance criterion;
        a set is accepted when chi2 < n_bins * n_sig², default=1
    :returns: list, result row with columns matching self.columns:
        [nlong, l_rate, l_ind1_z, l_ind2_z, l_zb, nshort, s_rate, s_ind1_z,
         s_ind2_z, s_zb, l_ind1, l_ind2, l_lb, s_ind1, s_ind2, s_lb,
         pierson_chi2, status]
    :raises ValueError: if method is not None and not a list, or if a GRB
        type tag is neither "Sample short" nor "Sample long"
    """
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
        print(f"Long catalog not fitting  : chi2 = {condition_long[1]} > len(ref) * {n_sig} sigma² ({len(self.l_pflux_bins) * n_sig**2}) -   trying again")
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
        print(f"Short catalog not fitting  : chi2 = {condition_short[1]} > len(ref) * {n_sig} sigma² ({len(self.s_pflux_bins) * n_sig**2}) -   trying again")
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
    if f"{int(self.n_year)}sample" in os.listdir("../Data/sources/SampledSpectra"):
      subprocess.call(f"rm -r ../Data/sources/SampledSpectra/{int(self.n_year)}sample", shell=True)
      # os.rmdir(f"../Data/sources/SampledSpectra/{int(n_year)}sample")
    print("Deletion done")

    condition = self.mcmc_condition(l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, params=params, mode="pflx", n_sig=n_sig)
    if condition[0]:
      row = [nlong_temp, l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, nshort_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(condition[1], 3), "Accepted"]
    else:
      row = [nlong_temp, l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, nshort_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(condition[1], 3), "Rejected"]
      print(f"Rejected : pearson chi2 = {np.around(condition[1], 3)}     [ite {run_iteration}]")

    list_param = l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp
    self.hist_plotter(run_iteration, [l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, np.around(condition[1], 3)], list_param, comment=comment, savefile=saveplotfile)

    return row

  def mcmc_condition(self, l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, params=None, mode="pflx", n_sig=1):
    """
    Evaluates the Pearson chi2 acceptance condition by comparing the simulated
    peak-flux histogram(s) against the GBM reference histograms.
    :param l_m_flux_temp: np.ndarray, mean photon fluxes of the simulated long GRBs [ph/cm2/s]
    :param l_p_flux_temp: np.ndarray, peak photon fluxes of the simulated long GRBs [ph/cm2/s]
    :param l_flnc_temp: np.ndarray, photon fluences of the simulated long GRBs [ph/cm2]
    :param s_m_flux_temp: np.ndarray, mean photon fluxes of the simulated short GRBs [ph/cm2/s]
    :param s_p_flux_temp: np.ndarray, peak photon fluxes of the simulated short GRBs [ph/cm2/s]
    :param s_flnc_temp: np.ndarray, photon fluences of the simulated short GRBs [ph/cm2]
    :param params: tuple or None, parameter set (unused in the condition itself but kept
        for interface consistency), default=None
    :param mode: str, which distribution(s) to compare; "pflx" compares both long and short
        peak fluxes, "l_pflx" compares long only, "s_pflx" compares short only, default="pflx"
    :param n_sig: int or float, sigma threshold; accepted when chi2 < n_bins * n_sig², default=1
    :returns: bool, float, float, True if the chi2 condition is met, the chi2 value, and
        the ratio of the last long peak-flux bin (1 for "s_pflx" mode)
    :raises ValueError: if mode is not one of the three accepted values
    """
    smp_pflux_l_hist = np.histogram(l_p_flux_temp, bins=self.bin_flux_l[self.nfluxbin_l[0]:])[0]
    smp_pflux_s_hist = np.histogram(s_p_flux_temp, bins=self.bin_flux_s[self.nfluxbin_s[0]:])[0]
    smp_flnc_l_hist = np.histogram(l_flnc_temp, bins=self.bin_flnc_l[self.nflncbin_l[0]:])[0]
    smp_flnc_s_hist = np.histogram(s_flnc_temp, bins=self.bin_flnc_s[self.nflncbin_s[0]:])[0]

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

    chi2_lim = len(expect_dat) * n_sig**2
    chi2 = np.sum((obs_dat-expect_dat)**2/expect_dat)
    return chi2 < chi2_lim, chi2, end_pflx_ratio

  def save_grb(self, filename, name, t90, lcname, fluence, mean_flux, peak_flux, red, band_low, band_high, ep, dl, lpeak, eiso, thetaj):
    """
    Appends a single synthetic GRB entry as a pipe-separated line to a catalog
    text file.
    :param filename: str, path to the catalog file to append to
    :param name: str, GRB identifier (e.g. "lGRB10S42")
    :param t90: float, T90 duration [s]
    :param lcname: str, filename of the associated light curve
    :param fluence: float, photon fluence [ph/cm2]
    :param mean_flux: float, mean photon flux [ph/cm2/s]
    :param peak_flux: float, peak photon flux [ph/cm2/s]
    :param red: float, observed redshift
    :param band_low: float, Band function low-energy spectral index alpha
    :param band_high: float, Band function high-energy spectral index beta
    :param ep: float, observed peak energy [keV]
    :param dl: float, luminosity distance [Gpc]
    :param lpeak: float, isotropic peak luminosity [erg/s]
    :param eiso: float, isotropic energy [erg]
    :param thetaj: float, jet opening angle [deg]
    """
    with open(filename, "a") as f:
      f.write(f"{name}|{t90}|{lcname}|{fluence}|{mean_flux}|{peak_flux}|{red}|{band_low}|{band_high}|{ep}|{dl}|{lpeak}|{eiso}|{thetaj}\n")

  def hist_plotter(self, iteration, histos, params, comment="", savefile=None):
    """
    Produces two diagnostic figures comparing the simulated and GBM reference
    flux/fluence distributions for both long and short GRBs, and optionally
    saves them and the underlying data as HDF5 files.
    :param iteration: int, iteration index used for naming output files
    :param histos: list, seven-element list containing:
        histos[0] - np.ndarray, simulated long GRB mean fluxes
        histos[1] - np.ndarray, simulated long GRB peak fluxes
        histos[2] - np.ndarray, simulated long GRB fluences
        histos[3] - np.ndarray, simulated short GRB mean fluxes
        histos[4] - np.ndarray, simulated short GRB peak fluxes
        histos[5] - np.ndarray, simulated short GRB fluences
        histos[6] - float, Pearson chi2 value (used in title and filename)
    :param params: tuple or None, 14-element parameter tuple used in the plot
        title; if None only the chi2 is shown in the title
    :param comment: str, optional comment string prepended to the plot title, default=""
    :param savefile: str or None, path prefix for output files (without
        extension); if None figures and data files are not saved, default=None
    """
    if params is not None:
      title = f"{comment}\n{params[0:7]}\n{params[7:]}\nPierson chi2 of pflx : {histos[6]}"
    else:
      title = f"{comment}\nPierson chi2 of pflx : {histos[6]}"

    plt.rcParams.update({'font.size': 13})

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

    if savefile is not None and self.mode == "catalog":
      hist_df_long = pd.DataFrame({"l_m_flux_temp": histos[0], "l_p_flux_temp": histos[1], "l_flnc_temp": histos[2]})
      hist_df_short = pd.DataFrame({"s_m_flux_temp": histos[3], "s_p_flux_temp": histos[4], "s_flnc_temp": histos[5]})
      hist_df_long.to_hdf(f"{savefile.split('.csv')[0]}_{iteration}_{int(histos[6])}long.h5", key="catalog", mode="w")
      hist_df_short.to_hdf(f"{savefile.split('.csv')[0]}_{iteration}_{int(histos[6])}short.h5", key="catalog", mode="w")

  def get_short(self, ite_num, short_rate, ind1_z_s, ind2_z_s, zb_s, ind1_s, ind2_s, lb_s):
    """
    Generates the observable and intrinsic properties of a single synthetic short GRB
    by drawing from the redshift and luminosity distributions and computing
    the Band spectrum normalisation.
    :param ite_num: int, iteration index (useful for testing)
    :param short_rate: float, overall rate normalisation for the short GRB
        redshift distribution
    :param ind1_z_s: float, low-z slope of the broken power-law redshift rate
    :param ind2_z_s: float, high-z slope of the broken power-law redshift rate
    :param zb_s: float, break redshift of the short GRB rate distribution
    :param ind1_s: float, low-luminosity slope of the broken power-law luminosity function
    :param ind2_s: float, high-luminosity slope of the broken power-law luminosity function
    :param lb_s: float, break luminosity of the short GRB luminosity function [erg/s]
    :returns: list, 15-element list:
        [z_obs, ep_obs [keV], ep_rest [keV], lpeak_rest [erg/s],
         mean_flux [ph/cm2/s], peak_flux [ph/cm2/s], t90 [s],
         fluence [ph/cm2], lc_name, band_low, band_high,
         dl [Gpc], eiso [erg], "Sample short", "Sample"]
    """
    np.random.seed((os.getpid() + int(time() * 1000)) % 2 ** 32)
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    z_obs_temp = acc_reject(red_rate_short, [short_rate, ind1_z_s, ind2_z_s, zb_s], self.zmin, self.zmax)
    lpeak_rest_temp = transfo_broken_plaw(ind1_s, ind2_s, lb_s, self.lmin, self.lmax)

    band_low_obs_temp, band_high_obs_temp = pick_normal_alpha_beta(self.band_low_s_mu, self.band_low_s_sig, self.band_high_s_mu, self.band_high_s_sig)

    t90_obs_temp = 1000
    while t90_obs_temp > 2:
      t90_obs_temp = 10 ** np.random.normal(-0.2373, 0.4058)

    lc_temp, gbm_mflux, gbm_pflux = self.closest_lc(t90_obs_temp)[:3]
    if np.isnan(gbm_pflux):
      pflux_to_mflux = pflux_to_mflux_calculator(lc_temp)
    else:
      pflux_to_mflux = gbm_mflux / gbm_pflux

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
    Generates the observable and intrinsic properties of a single synthetic long GRB
    by drawing from the redshift and luminosity distributions and computing
    the Band spectrum normalisation.
    :param ite_num: int, iteration index (useful for testing)
    :param long_rate: float, overall rate normalisation for the long GRB
        redshift distribution
    :param ind1_z_l: float, low-z slope of the broken power-law redshift rate
    :param ind2_z_l: float, high-z slope of the broken power-law redshift rate
    :param zb_l: float, break redshift of the long GRB rate distribution
    :param ind1_l: float, low-luminosity slope of the broken power-law luminosity function
    :param ind2_l: float, high-luminosity slope of the broken power-law luminosity function
    :param lb_l: float, break luminosity of the long GRB luminosity function [erg/s]
    :returns: list, 15-element list:
        [z_obs, ep_obs [keV], ep_rest [keV], lpeak_rest [erg/s],
         mean_flux [ph/cm2/s], peak_flux [ph/cm2/s], t90 [s],
         fluence [ph/cm2], lc_name, band_low, band_high,
         dl [Gpc], eiso [erg], "Sample long", "Sample"]
    """
    np.random.seed((os.getpid() + int(time() * 1000)) % 2 ** 32)
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    z_obs_temp = acc_reject(red_rate_long, [long_rate, ind1_z_l, ind2_z_l, zb_l], self.zmin, self.zmax)

    lpeak_rest_temp = transfo_broken_plaw(ind1_l, ind2_l, lb_l, self.lmin, self.lmax)
    band_low_obs_temp, band_high_obs_temp = pick_normal_alpha_beta(self.band_low_l_mu, self.band_low_l_sig, self.band_high_l_mu, self.band_high_l_sig)
    t90_obs_temp = 0
    while t90_obs_temp <= 2:
      t90_obs_temp = 10 ** np.random.normal(1.4438, 0.4956)

    lc_temp, gbm_mflux, gbm_pflux = self.closest_lc(t90_obs_temp)[:3]
    if np.isnan(gbm_pflux):
      pflux_to_mflux = pflux_to_mflux_calculator(lc_temp)
    else:
      pflux_to_mflux = gbm_mflux / gbm_pflux

    dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
    ep_rest_temp = yonetoku_reverse_long(lpeak_rest_temp)
    ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)
    eiso_rest_temp = amati_long(ep_rest_temp)

    ##################################################################################################################
    # Calculation of spectrum and data saving
    ##################################################################################################################
    ener_range = np.logspace(1, 3, 10001)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    temp_mean_flux = temp_peak_flux * pflux_to_mflux

    return [z_obs_temp, ep_obs_temp, ep_rest_temp, lpeak_rest_temp, temp_mean_flux, temp_peak_flux, t90_obs_temp, temp_mean_flux * t90_obs_temp, lc_temp, band_low_obs_temp, band_high_obs_temp, dl_obs_temp,
            eiso_rest_temp, "Sample long", "Sample"]

  def closest_lc(self, searched_time):
    """
    Finds the GBM light curve file whose GRB T90 duration is closest to the requested time.
    If multiple GRBs share the minimum distance, one is chosen at random.
    :param searched_time: float, target T90 duration to match [s]
    :returns: str, float, float, float, the light curve filename, the GRB mean
        flux [ph/cm2/s], the GRB peak flux [ph/cm2/s], and the GRB T90 [s]
    :raises ValueError: if no matching GRB is found (should not occur in normal use)
    """
    abs_diff = np.abs(np.array(self.gbm_cat.df.t90.values, dtype=float) - searched_time)
    gbm_indexes = np.where(abs_diff == np.min(abs_diff))[0]
    if len(gbm_indexes) == 0:
      raise ValueError("No GRB found for the closest GRB duration")
    elif len(gbm_indexes) == 1:
      gbm_index = gbm_indexes[0]
    else:
      gbm_index = gbm_indexes[np.random.randint(len(gbm_indexes))]
    return f"LightCurve_{self.gbm_cat.df.name.values[gbm_index]}.dat", self.gbm_cat.df.mean_flux.values[gbm_index], self.gbm_cat.df.peak_flux.values[gbm_index], self.gbm_cat.df.t90.values[gbm_index]

  def gbm_reference_distri(self, print_bins=True):
    """
    Displays the GBM reference flux and fluence distributions used for the
    chi2 comparison, plotting peak flux, mean flux, and fluence histograms for
    both long and short GRBs. Optionally prints the bin-by-bin counts to stdout.
    :param print_bins: bool, if True print the histogram bin edges and counts
        for all four distributions (long/short peak flux and fluence) to stdout, default=True
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

####################################################################################################
# USE EXAMPLES
####################################################################################################
# from src.Catalogs.catalogMC import MCCatalog
# testcat = MCCatalog(mode="mc") # To explore the variable space
# testcat = MCCatalog(mode="parametrized") # To execute with pre-set parameters
# testcat = MCCatalog(mode="catalog") # To create a catalogue

# To show the plots to explore the variable space with the mode "mc"
# from catalogMC import *
# import matplotlib as mpl
# mpl.use("Qt5Agg")
#
# file = "../Data/CatData/CatSampling/mclongv9-300/mc_fit.csv" # file obtained with the "mc" mode
# leg_mode = "fine"
# MC_explo_pairplot(file, leg_mode, grbtype="short")
# plt.show()