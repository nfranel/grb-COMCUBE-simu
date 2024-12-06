import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
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

class MCCatalog:
  """

  """
  def __init__(self, gbm_file="GBM/allGBM.txt", sttype=None, rf_file="GBM/rest_frame_properties.txt", mcmc_number=10):
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
    self.bin_flux_l = np.concatenate((np.logspace(-1, np.log10(self.flux_lim[0]), self.nfluxbin_l[0] + 1),
                                      np.logspace(np.log10(self.flux_lim[0]), np.log10(self.flux_lim[1]), self.nfluxbin_l[1] + 1)[1:],
                                      np.logspace(np.log10(self.flux_lim[1]), 4, self.nfluxbin_l[2] + 1)[1:]))
    self.bin_flux_s = np.concatenate((np.logspace(-1, np.log10(self.flux_lim[0]), self.nfluxbin_s[0] + 1),
                                      np.logspace(np.log10(self.flux_lim[0]), np.log10(self.flux_lim[1]), self.nfluxbin_s[1] + 1)[1:],
                                      np.logspace(np.log10(self.flux_lim[1]), 4, self.nfluxbin_s[2] + 1)[1:]))
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

    # min and max values for distributions (estimated with ranges from Pescalli 2016, Ghirlanda 2016)
    # # Redshift
    # self.l_rate_min = 0.2
    # self.l_rate_max = 0.6
    # self.l_ind1_z_min = 1.5
    # self.l_ind1_z_max = 2.5
    # self.l_ind2_z_min = -0.5
    # self.l_ind2_z_max = -0.9
    # self.l_zb_min = 3
    # self.l_zb_max = 4.2
    #
    # self.s_rate_min = 0.4
    # self.s_rate_max = 1.1
    # self.s_ind1_z_min = 0.5
    # self.s_ind1_z_max = 4.1
    # self.s_ind2_z_min = 1.1
    # self.s_ind2_z_max = 3.7
    # self.s_zb_min = 2
    # self.s_zb_max = 3.3
    # # Luminosity
    # self.l_ind1_min = -1.6
    # self.l_ind1_max = -0.7
    # self.l_ind2_min = -2.6
    # self.l_ind2_max = -1.6
    # self.l_lb_min = 2e51
    # self.l_lb_max = 6e52
    #
    # self.s_ind1_min = -1
    # self.s_ind1_max = -0.39
    # self.s_ind2_min = -3.7
    # self.s_ind2_max = -1.7
    # self.s_lb_min = 0.91e52
    # self.s_lb_max = 3.4e52

    # Redshift
    self.l_rate_min = 0.2
    self.l_rate_max = 0.7
    self.l_ind1_z_min = 2
    self.l_ind1_z_max = 4.3
    self.l_ind2_z_min = -2.4
    self.l_ind2_z_max = -0.01
    self.l_zb_min = 2
    self.l_zb_max = 5

    self.s_rate_min = 0.4
    self.s_rate_max = 1.1
    self.s_ind1_z_min = 0.5
    self.s_ind1_z_max = 4.1
    self.s_ind2_z_min = 1.1
    self.s_ind2_z_max = 3.7
    self.s_zb_min = 2
    self.s_zb_max = 3.3
    # Luminosity
    self.l_ind1_min = -1.1
    self.l_ind1_max = 0
    self.l_ind2_min = -5
    self.l_ind2_max = -2
    self.l_lb_min = 5e50
    self.l_lb_max = 1e52

    self.s_ind1_min = -1
    self.s_ind1_max = -0.39
    self.s_ind2_min = -3.7
    self.s_ind2_max = -1.7
    self.s_lb_min = 0.91e52
    self.s_lb_max = 3.4e52

    # Spectrum indexes gaussian distributions
    self.band_low_l_mu, self.band_low_l_sig = -0.95, 0.31
    self.band_high_l_mu, self.band_high_l_sig = -2.17, 0.30
    self.band_low_s_mu, self.band_low_s_sig = -0.57, 0.32
    self.band_high_s_mu, self.band_high_s_sig = -2.17, 0.31
    # T90 gaussian distributions
    # amplitude long : 467, mean long : 1.4875, stdev long : 0.45669
    # amplitude short : 137.5, mean short : -0.025, stdev short : 0.631
    self.log_t90_l_mu, self.log_t90_l_sig = 1.4875, 0.45669
    self.log_t90_s_mu, self.log_t90_s_sig = -0.025, 0.631

    # variables containing the MCMC results
    self.columns = ["long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "long_ind1_lum", "long_ind2_lum", "long_lb", "short_ind1_lum", "short_ind2_lum",
                    "short_lb", "log_pvalue", "status"]
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

    # mc_mode = "long_lum"
    mc_mode = "long_red_lum"
    # mc_mode = "short_lum"
    # mc_mode = "short_red"
    # mc_mode = "mc"

    self.cond_option = "long_pflux"
    # self.cond_option = "long_flnc"
    # self.cond_option = "long"
    # self.cond_option = "short_pflux"
    # self.cond_option = "short_flnc"
    # self.cond_option = "short"
    # self.cond_option = None

    print("Starting")
    if mc_mode == "long_lum":
      savefile = "Sampled/longlum/longfit_lum.csv"
      comm = "Long-Luminosity"
      if not (f"longlum" in os.listdir("Sampled/")):
        os.mkdir("Sampled/longlum")
      # param_list = build_params(0.42, 2.07, -0.7, 3.6,  [-0.55, -0.6, -0.65, -0.70, -0.75], [-2.8, -2.9, -3, -3.1, -3.2], [0.7e+52, 0.9e+52, 1.12e+52, 1.4e+52],
      #                           0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52)
      param_list = build_params(0.42, 2.07, -0.7, 3.6,  [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2], [-1.8, -2.1, -2.4, -2.7, -3], [1e51, 5e51, 1e52, 2e52],
                                0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52)
      par_size = len(param_list)

      self.run_mc(par_size, thread_number=thread_num, method=param_list, savefile=savefile, comment=comm)
      # for ite_mc in range(len(histograms)):
        # self.hist_plotter(ite_mc, histograms[ite_mc], param_list[ite_mc], comment=comm, savefile=savefile)

      select_cols = ["long_ind1_lum", "long_ind2_lum", "long_lb", "log_pvalue"]
      df_selec = self.result_df[select_cols]
      plt.subplots(1, 1)
      title = f"Log p-value on {self.cond_option}"
      plt.suptitle(title)
      sns.pairplot(df_selec, hue="log_pvalue", corner=True, plot_kws={'s': 10})
      plt.savefig(f"{savefile.split('.csv')[0]}_df")
      plt.close()

    elif mc_mode == "long_red_lum":

      def categorize_log_pvalue(value):
        if value < -20:
          return "< -20"
        elif value < -10:
          return "-10 - -20"
        elif value < -5:
          return "-5 - -10"
        elif value < -2:
          return "-2 - -5"
        else:
          return "0 - -2"

      fold_name = "longredlum3000v4"
      savefile = f"Sampled/{fold_name}/longfit_red_lum.csv"
      comm = "Long-Redshift-Luminosity"
      if not (f"{fold_name}" in os.listdir("Sampled/")):
        os.mkdir(f"Sampled/{fold_name}")
      # l_rate, l_ind1_z, l_ind2_z, l_zb, l_ind1, l_ind2, l_lb, s_rate, s_ind1_z, s_ind2_z, s_zb, s_ind1, s_ind2, s_lb
      # param_list = build_params([0.2, 0.3, 0.4, 0.5], [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6], [-1.1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4], [2, 2.6, 3.1, 3.6, 4.1, 5], -0.65, -3, 1.12e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52)
      # param_list = [[0.42, 2.07, -0.7, 3.6, -0.65, -3, 1.12e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52]]
      # param_list = [[0.2, 2.4, -0.6, 5, -0.8, -2.5, 1e51, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],
      #              [0.2, 2.4, -0.6, 5, -0.8, -3, 1e51, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],
      #              [0.2, 2.4, -0.6, 5, -0.8, -3, 1e51, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],
      #              [0.2, 2.4, -0.4, 3.6, -0.2, -3, 1e51, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],
      #              [0.2, 2.6, -1.1, 4.1, -0.8, -2.1, 1e51, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],
      #              [0.2, 3, -0.6, 5, -0.6, -3, 5e51, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52],
      #              [0.5, 2.6, -0.4, 4.1, -0.2, -3, 1e51, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52]]
      # par_size = len(param_list)

      param_list = None
      par_size = 3000
      # param_list = build_params([0.2, 0.4, 0.5], [2.4, 2.6, 3.], [-1.1, -0.8, -0.6, -0.4], [2, 2.6, 3.6, 4.1, 5], [-0.8, -0.7, -0.6, -0.3, -0.2], [-1.8, -2.1, -2.5, -3], [1e51, 5e51, 1e52, 2e52], 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52)
      # param_list = build_params(0.2, 2.1, [-1.1, -0.6, -0.4], [2, 4.1, 5], -0.6, -2.1, 1e52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52)
      self.run_mc(par_size, thread_number=thread_num, method=param_list, savefile=savefile)
      # for ite_mc in range(len(histograms)):
      #   self.hist_plotter(ite_mc, histograms[ite_mc], param_list[ite_mc], comment=comm, savefile=savefile)

      self.result_df['log_pvalue_category'] = self.result_df['log_pvalue'].apply(categorize_log_pvalue)

      select_cols = ["long_rate", "long_ind1_z", "long_ind2_z", "long_zb", "long_ind1_lum", "long_ind2_lum", "long_lb", "log_pvalue_category"]
      df_selec = self.result_df[select_cols]
      plt.subplots(1, 1)
      title = f"Log p-value on {self.cond_option}"
      plt.suptitle(title)
      sns.pairplot(df_selec, hue="log_pvalue_category", corner=True, plot_kws={'s': 50}, palette="viridis_r")
      plt.savefig(f"{savefile.split('.csv')[0]}_df")
      plt.close()

    elif mc_mode == "short_lum":
      savefile = "Sampled/shortlum/shortfit_lum.csv"
      comm = "Short-Luminosity"
      if not (f"shortlum" in os.listdir("Sampled/")):
        os.mkdir("Sampled/shortlum")
      param_list = build_params(0.42, 2.07, -0.7, 3.6, -0.65, -3, 1.12e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52)
      par_size = len(param_list)

      self.run_mc(par_size, thread_number=thread_num, method=param_list, savefile=savefile)
      # for ite_mc in range(len(histograms)):
      #   self.hist_plotter(ite_mc, histograms[ite_mc], param_list[ite_mc], comment=comm, savefile=savefile)

      select_cols = ["short_ind1_lum", "short_ind2_lum", "short_lb", "log_pvalue"]
      df_selec = self.result_df[select_cols]
      plt.subplots(1, 1)
      title = f"Log p-value on {self.cond_option}"
      plt.suptitle(title)
      sns.pairplot(df_selec, hue="log_pvalue", corner=True, plot_kws={'s': 10})
      plt.savefig(f"{savefile.split('.csv')[0]}_df")
      plt.close()

    elif mc_mode == "short_red":
      savefile = "Sampled/shortred/shortfit_red.csv"
      comm = "Short-Redshift"
      if not (f"shortred" in os.listdir("Sampled/")):
        os.mkdir("Sampled/shortred")
      param_list = build_params(0.42, 2.07, -0.7, 3.6, -0.65, -3, 1.12e+52, 0.25, 2.8, 3.5, 2.3, -0.53, -3.4, 2.8e52)
      par_size = len(param_list)

      self.run_mc(par_size, thread_number=thread_num, method=param_list, savefile=savefile)
      # for ite_mc in range(len(histograms)):
      #   self.hist_plotter(ite_mc, histograms[ite_mc], param_list[ite_mc], comment=comm, savefile=savefile)

      select_cols = ["short_rate", "short_ind1_z", "short_ind2_z", "short_zb", "log_pvalue"]
      df_selec = self.result_df[select_cols]
      plt.subplots(1, 1)
      title = f"Log p-value on {self.cond_option}"
      plt.suptitle(title)
      sns.pairplot(df_selec, hue="log_pvalue", corner=True, plot_kws={'s': 10})
      plt.savefig(f"{savefile.split('.csv')[0]}_df")
      plt.close()

    elif mc_mode == "mc":
      savefile = "Sampled/mcfit/mc_fit.csv"
      comm = "All fit"
      if not (f"mcfit" in os.listdir("Sampled/")):
        os.mkdir("Sampled/mcfit")

      self.run_mc(mcmc_number, thread_number=thread_num, method=None, savefile=savefile)
      # for ite_mc in range(len(histograms)):
      #   self.hist_plotter(ite_mc, histograms[ite_mc], None, comment=comm, savefile=savefile)

      plt.subplots(1, 1)
      title = f"Log p-value on {self.cond_option}"
      plt.suptitle(title)
      sns.pairplot(self.result_df, hue="log_pvalue", corner=True, plot_kws={'s': 10})
      plt.savefig(f"{savefile.split('.csv')[0]}_df")
      plt.close()

  def run_mc(self, run_number, thread_number=1, method=None, savefile=None, comment=""):
    print(f"Starting the run for {run_number} iterations")
    if thread_number == 'all':
      print("Parallel execution with all threads")
      with mp.Pool() as pool:
        rows_ret = pool.starmap(self.get_sample, zip(range(run_number), repeat(method), repeat(comment), repeat(savefile)))
    elif type(thread_number) is int and thread_number > 1:
      print(f"Parallel execution with {thread_number} threads")
      with mp.Pool(thread_number) as pool:
        rows_ret = pool.starmap(self.get_sample, zip(range(run_number), repeat(method), repeat(comment), repeat(savefile)))
    else:
      rows_ret = [self.get_sample(ite, method=method, comment=comment, savefile=savefile) for ite in range(run_number)]
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

  def get_sample(self, run_iteration, method=None, comment="", savefile=None):
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

      print(f"début des {nlong_temp} longs et {nshort_temp} courts     [ite {run_iteration}]")
      l_m_flux_temp, l_p_flux_temp, l_flnc_temp = [], [], []
      s_m_flux_temp, s_p_flux_temp, s_flnc_temp = [], [], []
      run_times_long = []
      run_times_short = []
      for ite_long in range(nlong_temp):
        # if ite_long % 100 == 0:
        #   print(f"number of iteration for the long bursts : {ite_long}     [ite {run_itation}]", end="\r")
        init_time = time()
        l_temp_ret = self.get_long(l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp)
        run_times_long.append(time() - init_time)
        l_m_flux_temp.append(l_temp_ret[0])
        l_p_flux_temp.append(l_temp_ret[2])
        l_flnc_temp.append(l_temp_ret[2])
      print(f"Long finished     [ite {run_iteration}]")
      for ite_short in range(nshort_temp):
        # if ite_short % 100 == 0:
        #   print(f"number of iteration for the short bursts : {ite_short}     [ite {run_itation}]", end="\r")
        init_time = time()
        s_temp_ret = self.get_short(s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp)
        run_times_short.append(time() - init_time)
        s_m_flux_temp.append(s_temp_ret[0])
        s_p_flux_temp.append(s_temp_ret[2])
        s_flnc_temp.append(s_temp_ret[2])
      print(f"Short finished     [ite {run_iteration}]")
      # fig, ax = plt.subplots(1, 1)
      # ax.hist(run_times_long, histtype="step", label="long run times")
      # ax.hist(run_times_short, histtype="step", label="short run times")
      # plt.show()

      condition = self.mcmc_condition(l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, params=params)
      pflux_ratio_thresh = 4
      if condition[2] < pflux_ratio_thresh or method is None:
        end_flux_loop = False
      else:
        print(f"====== LOOPING - pflux ratio : {round(condition[2], 2)} > {pflux_ratio_thresh} ======")

    if condition[0]:
      row = [l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(np.log10(condition[1]), 3), "Accepted"]
      # data_row = pd.DataFrame(data=[row], columns=self.columns)
      # self.result_df = pd.concat([self.result_df, data_row], ignore_index=True)
    else:
      row = [l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp, np.around(np.log10(condition[1]), 3), "Rejected"]
      # data_row = pd.DataFrame(data=[row], columns=self.columns)
      # self.result_df = pd.concat([self.result_df, data_row], ignore_index=True)
      # print(data_row)
      print(f"Rejected : log_pvalue = {np.around(np.log10(condition[1]), 3)}     [ite {run_iteration}]")

    list_param = l_rate_temp, l_ind1_z_temp, l_ind2_z_temp, l_zb_temp, l_ind1_temp, l_ind2_temp, l_lb_temp, s_rate_temp, s_ind1_z_temp, s_ind2_z_temp, s_zb_temp, s_ind1_temp, s_ind2_temp, s_lb_temp
    self.hist_plotter(run_iteration, [l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, np.around(np.log10(condition[1]), 3)], list_param, comment=comment, savefile=savefile)

    # creating histograms
    # smp_pflux_l_hist = np.histogram(l_p_flux_temp, bins=self.bin_flux_l[self.nfluxbin_l[0]:])[0]
    # smp_pflux_s_hist = np.histogram(s_p_flux_temp, bins=self.bin_flux_s[self.nfluxbin_s[0]:])[0]
    # smp_flnc_l_hist = np.histogram(l_flnc_temp, bins=self.bin_flnc_l[self.nflncbin_l[0]:])[0]
    # smp_flnc_s_hist = np.histogram(s_flnc_temp, bins=self.bin_flnc_s[self.nflncbin_s[0]:])[0]

    # hist_l_m_flux_temp = np.histogram(l_m_flux_temp, bins=self.usual_bins)[0]
    # hist_l_p_flux_temp = np.histogram(l_p_flux_temp, bins=self.bin_flux_l)[0]
    # hist_l_flnc_temp = np.histogram(l_flnc_temp, bins=self.bin_flnc_l)[0]
    # hist_s_m_flux_temp = np.histogram(s_m_flux_temp, bins=self.usual_bins)[0]
    # hist_s_p_flux_temp = np.histogram(s_p_flux_temp, bins=self.bin_flux_s)[0]
    # hist_s_flnc_temp = np.histogram(s_flnc_temp, bins=self.bin_flnc_s)[0]
    # return hist_l_m_flux_temp, hist_l_p_flux_temp, hist_l_flnc_temp, hist_s_m_flux_temp, hist_s_p_flux_temp, hist_s_flnc_temp
    # return [l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, np.around(np.log10(condition[1]), 3), row]
    return row

  def mcmc_condition(self, l_m_flux_temp, l_p_flux_temp, l_flnc_temp, s_m_flux_temp, s_p_flux_temp, s_flnc_temp, params=None):
    """
    Condition on the histograms to consider a value is correct
    """
    smp_pflux_l_hist = np.histogram(l_p_flux_temp, bins=self.bin_flux_l[self.nfluxbin_l[0]:])[0]
    smp_pflux_s_hist = np.histogram(s_p_flux_temp, bins=self.bin_flux_s[self.nfluxbin_s[0]:])[0]
    smp_flnc_l_hist = np.histogram(l_flnc_temp, bins=self.bin_flnc_l[self.nflncbin_l[0]:])[0]
    smp_flnc_s_hist = np.histogram(s_flnc_temp, bins=self.bin_flnc_s[self.nflncbin_s[0]:])[0]

    smp_pflux_l_hist_norm = smp_pflux_l_hist * np.sum(self.l_pflux_bins) / np.sum(smp_pflux_l_hist)
    smp_pflux_s_hist_norm = smp_pflux_s_hist * np.sum(self.s_pflux_bins) / np.sum(smp_pflux_s_hist)
    smp_flnc_l_hist_norm = smp_flnc_l_hist * np.sum(self.l_flnc_bins) / np.sum(smp_flnc_l_hist)
    smp_flnc_s_hist_norm = smp_flnc_s_hist * np.sum(self.s_flnc_bins) / np.sum(smp_flnc_s_hist)

    if self.cond_option is None:
      obs_dat = np.concatenate((smp_pflux_l_hist_norm, smp_pflux_s_hist_norm, smp_flnc_l_hist_norm, smp_flnc_s_hist_norm))
      expect_dat = np.concatenate((self.l_pflux_bins, self.s_pflux_bins, self.l_flnc_bins, self.s_flnc_bins))
      end_pflx_ratio = 1
    elif self.cond_option == "long_pflux":
      obs_dat = smp_pflux_l_hist_norm
      expect_dat = self.l_pflux_bins
      end_pflx_ratio = smp_pflux_l_hist_norm[-1] / self.l_pflux_bins[-1]
    elif self.cond_option == "long_flnc":
      obs_dat = smp_flnc_l_hist_norm
      expect_dat = self.l_flnc_bins
      end_pflx_ratio = 1
    elif self.cond_option == "long":
      obs_dat = np.concatenate((smp_pflux_l_hist_norm, smp_flnc_l_hist_norm))
      expect_dat = np.concatenate((self.l_pflux_bins, self.l_flnc_bins))
      end_pflx_ratio = smp_pflux_l_hist_norm[-1] / self.l_pflux_bins[-1]
    elif self.cond_option == "short_pflux":
      obs_dat = smp_pflux_s_hist_norm
      expect_dat = self.s_pflux_bins
      end_pflx_ratio = smp_pflux_s_hist_norm[-1] / self.s_pflux_bins[-1]
    elif self.cond_option == "short_flnc":
      obs_dat = smp_flnc_s_hist_norm
      expect_dat = self.s_flnc_bins
      end_pflx_ratio = 1
    elif self.cond_option == "short":
      obs_dat = np.concatenate((smp_pflux_s_hist_norm, smp_flnc_s_hist_norm))
      expect_dat = np.concatenate((self.s_pflux_bins, self.s_flnc_bins))
      end_pflx_ratio = smp_pflux_s_hist_norm[-1] / self.s_pflux_bins[-1]
    else:
      raise ValueError("Wrong value for cond_option : must be None, 'long_pflux', 'long_flnc', 'long', 'short_pflux', 'short_flnc' or 'short'")

    cond_test = chisquare(obs_dat, f_exp=expect_dat, ddof=0)
    return cond_test[1] > 0.95, cond_test[1], end_pflx_ratio

  def hist_plotter(self, iteration, histos, params, comment="", savefile=None):
    if params is not None:
      title = f"{comment}\n{params[0:7]}\n{params[7:]}\nLog p-value on {self.cond_option} : {histos[6]}"
    else:
      title = f"{comment}\nLog p-value on {self.cond_option} : {histos[6]}"

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
      if np.isinf(histos[6]):
        pval_suf = "-inf"
      else:
        pval_suf = int(histos[6])
      plt.savefig(f"{savefile.split('.csv')[0]}_{iteration}_{pval_suf}")
    plt.close(fig1)

    # plt.show()

  def get_short(self, short_rate, ind1_z_s, ind2_z_s, zb_s, ind1_s, ind2_s, lb_s):
    """
    Creates the quatities of a short burst according to distributions
    Based on Lana Salmon's thesis and Ghirlanda et al, 2016
    """
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    z_obs_temp = acc_reject(red_rate_short, [short_rate, ind1_z_s, ind2_z_s, zb_s], self.zmin, self.zmax)
    # lpeak_rest_temp = acc_reject(broken_plaw, [ind1_s, ind2_s, lb_s], self.lmin, self.lmax)
    lpeak_rest_temp = transfo_broken_plaw(ind1_s, ind2_s, lb_s, self.lmin, self.lmax)

    ep_rest_temp = yonetoku_reverse_short(lpeak_rest_temp)
    # ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)

    band_low_obs_temp, band_high_obs_temp = pick_lognormal_alpha_beta(self.band_low_s_mu, self.band_low_s_sig, self.band_high_s_mu, self.band_high_s_sig)

    t90_obs_temp = 1000
    while t90_obs_temp > 2:
      t90_obs_temp = 10 ** norm.rvs(-0.025, 0.631)

    lc_temp = self.closest_lc(t90_obs_temp)
    # times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
    # pflux_to_mflux = np.mean(counts) / np.max(counts)
    pflux_to_mflux = pflux_to_mflux_calculator(lc_temp)


    dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
    # eiso_rest_temp = amati_short(ep_rest_temp)

    ##################################################################################################################
    # Calculation of spectrum and data saving
    ##################################################################################################################
    ener_range = np.logspace(1, 3, 10001)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    temp_mean_flux = temp_peak_flux * pflux_to_mflux

    return [temp_mean_flux, temp_peak_flux, temp_mean_flux * t90_obs_temp]

  def get_long(self, long_rate, ind1_z_l, ind2_z_l, zb_l, ind1_l, ind2_l, lb_l):
    """
    Creates the quatities of a long burst according to distributions
    Based on Sarah Antier's thesis
    """
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    timelist = []
    init_time = time()
    z_obs_temp = acc_reject(red_rate_long, [long_rate, ind1_z_l, ind2_z_l, zb_l], self.zmin, self.zmax)
    timelist.append(time() - init_time)
    init_time = time()
    # lpeak_rest_temp = acc_reject(broken_plaw, [ind1_l, ind2_l, lb_l], self.lmin, self.lmax)
    lpeak_rest_temp = transfo_broken_plaw(ind1_l, ind2_l, lb_l, self.lmin, self.lmax)
    timelist.append(time() - init_time)
    init_time = time()
    band_low_obs_temp, band_high_obs_temp = pick_lognormal_alpha_beta(self.band_low_l_mu, self.band_low_l_sig, self.band_high_l_mu, self.band_high_l_sig)
    timelist.append(time() - init_time)
    init_time = time()
    t90_obs_temp = 0
    while t90_obs_temp <= 2:
      t90_obs_temp = 10 ** norm.rvs(1.4875, 0.45669)
    timelist.append(time() - init_time)

    lc_temp = self.closest_lc(t90_obs_temp)
    # times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
    # pflux_to_mflux = np.mean(counts) / np.max(counts)
    pflux_to_mflux = pflux_to_mflux_calculator(lc_temp)

    dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
    ep_rest_temp = yonetoku_reverse_long(lpeak_rest_temp)
    init_time = time()
    # ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)
    # eiso_rest_temp = amati_long(ep_rest_temp)

    ##################################################################################################################
    # Calculation of spectrum and data saving
    ##################################################################################################################
    ener_range = np.logspace(1, 3, 10001)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    temp_mean_flux = temp_peak_flux * pflux_to_mflux
    timelist.append(time() - init_time)
    init_time = time()
    # for times in timelist:
    #   print(f"Time taken : {times:8.6f}s making {times/np.sum(timelist)*100:5.2f}% of the run")
    return [temp_mean_flux, temp_peak_flux, temp_mean_flux * t90_obs_temp]

  def closest_lc(self, searched_time):
    """
    Find the lightcurve file with a duration which is the closest to the sampled t90 time
    """
    abs_diff = np.abs(np.array(self.gbm_cat.df.t90, dtype=float) - searched_time)
    gbm_index = np.argmin(abs_diff)
    # print(searched_time, float(self.gbm_cat.t90[gbm_index]))
    return f"LightCurve_{self.gbm_cat.df.name[gbm_index]}.dat"

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


testcat = MCCatalog()
