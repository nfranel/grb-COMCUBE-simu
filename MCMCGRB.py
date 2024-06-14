import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import skewnorm
from catalog import Catalog
from funcmod import extract_lc, calc_flux_gbm, use_scipyquad
from funcsample import *
# from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
# import warnings

from astropy.cosmology import Planck18
from astropy.cosmology import FlatLambdaCDM


class GRBSample:
  """
  Class to create GRB samples
  """

  def __init__(self, version_long=None, version_short=None):
    """
    Initialisation of the different attributes
    """
    #################################################################################################################
    # General attributres
    #################################################################################################################
    self.zmin = 0
    self.zmax = 10
    self.epmin = 1e0
    self.epmax = 1e5
    self.thetaj_min = 0
    self.thetaj_max = 15
    self.lmin = 1e49  # erg/s
    self.lmax = 1e55
    self.n_year = 10
    gbmduty = 0.587
    self.gbm_weight = 1 / gbmduty / 10
    self.sample_weight = 1 / self.n_year
    if version_long is None:
      self.version_long = 1
    else:
      self.version_long = version_long
    if version_short is None:
      self.version_short = 1
    else:
      self.version_short = version_long
    self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # self.cosmo = Planck18
    self.filename = f"./Sampled/sampled_grb_cat_{self.n_year}years.txt"
    self.columns = ["Redshift", "EpeakObs", "Epeak", "PeakLuminosity", "MeanFlux", "PeakFlux", "T90", "Fluence", "LightCurveName", "BandLow", "BandHigh", "LuminosityDistance", "EnergyIso", "Type", "Cat"]
    self.sample_df = pd.DataFrame(columns=self.columns)
    self.gbm_df = pd.DataFrame(columns=self.columns)
    #################################################################################################################
    # Short GRB attributes
    #################################################################################################################
    # PARAMETERS
    self.al1_s, self.al2_s, self.lb_s = 0.53, 3.4, 2.8  # TODO PUT DISPERSION
    self.band_low_short = -0.6
    self.band_high_short = -2.5
    self.short_rate = 0.20  # +0.04 -0.07 [Gpc-3.yr-1] # Ghirlanda 2016
    self.nshort = None  # func to calculate ?

    #################################################################################################################
    # long GRB attributes
    #################################################################################################################
    self.band_low_l_mu, self.band_low_l_sig = -0.87, 0.33
    self.band_high_l_mu, self.band_high_l_sig = -2.36, 0.31
    self.t90_mu, self.t90_sig = 58.27, 18
    self.nlong = None

    #################################################################################################################
    # GBM attributes
    #################################################################################################################
    self.cat_name = "GBM/allGBM.txt"
    self.rest_cat_file = "GBM/rest_frame_properties.txt"
    self.sttype = [4, '\n', 5, '|', 4000]
    self.gbm_cat = None

    self.kde_short_log_ep_obs = None
    self.kde_short_log_t90 = None
    self.kde_short_band_low = None
    self.kde_short_band_high = None

    self.kde_long_log_ep_obs = None
    self.kde_long_log_t90 = None
    self.kde_long_band_low = None
    self.kde_long_band_high = None

    #################################################################################################################
    # Setting some attributes
    #################################################################################################################
    self.gbm_cat = Catalog(self.cat_name, self.sttype, self.rest_cat_file)
    for ite_gbm in range(len(self.gbm_cat.df)):
      band_rest = True
      best_pflux_mod = self.gbm_cat.df.pflx_best_fitting_model[ite_gbm]

      # Affecting the values for constructing the row
      ep_obs_temp = self.gbm_cat.df.flnc_band_epeak[ite_gbm]
      if band_rest:
        ep_temp = self.gbm_cat.df.rest_epeak_band[ite_gbm]
        z_temp = ep_temp / ep_obs_temp - 1
        plum_temp = self.gbm_cat.df.rest_liso_band[ite_gbm]
        dl_temp = np.nan
        eiso_temp = self.gbm_cat.df.rest_eiso_band[ite_gbm]
      else:
        ep_temp = self.gbm_cat.df.rest_epeak_comp[ite_gbm]
        z_temp = ep_temp / ep_obs_temp - 1
        plum_temp = self.gbm_cat.df.rest_liso_comp[ite_gbm]
        dl_temp = np.nan
        eiso_temp = self.gbm_cat.df.rest_eiso_comp[ite_gbm]

      temp_mean_flux = calc_flux_gbm(self.gbm_cat, ite_gbm, (10, 1000))
      if type(best_pflux_mod) == str:
        temp_peak_flux = self.gbm_cat.df[f"{best_pflux_mod}_phtflux"][ite_gbm]
      else:
        if np.isnan(best_pflux_mod):
          temp_peak_flux = np.nan
        else:
          raise ValueError("A value for pflx_best_fitting_model is not set properly")
      temp_t90 = self.gbm_cat.df.t90[ite_gbm]
      temp_fluence = temp_mean_flux * temp_t90
      lc_temp = self.closest_lc(temp_t90)
      temp_band_low = self.gbm_cat.df.flnc_band_alpha[ite_gbm]
      temp_band_high = self.gbm_cat.df.flnc_band_beta[ite_gbm]
      if temp_t90 < 2:
        temp_type = f"GBM short {self.gbm_cat.df.flnc_best_fitting_model[ite_gbm]}"
      else:
        temp_type = f"GBM long {self.gbm_cat.df.flnc_best_fitting_model[ite_gbm]}"
      temp_cat = "GBM"
      row = [z_temp, ep_obs_temp, ep_temp, plum_temp, temp_mean_flux, temp_peak_flux, temp_t90, temp_fluence, lc_temp, temp_band_low, temp_band_high, dl_temp, eiso_temp, temp_type, temp_cat]
      # print(row[:8] + row[9:-2])
      # print(np.isnan(row[:8] + row[9:-2]).any())
      data_row = pd.DataFrame(data=[row], columns=self.columns)
      self.gbm_df = pd.concat([self.gbm_df, data_row], ignore_index=True)
    self.gbm_distri()

    #################################################################################################################
    # Creating GRBs
    #################################################################################################################
    # Setting up a save file
    with open(self.filename, "w") as f:
      f.write(f"Catalog of synthetic GRBs sampled over {self.n_year} years\n")
      f.write(f"Based on differents works, see MCMCGRB.py for more details\n")
      f.write("Keys and units : \n")
      f.write("name|t90|light curve name|fluence|mean flux|redshift|Band low energy index|Band high energy index|peak energy|luminosity distance|isotropic luminosity|isotropic energy|jet opening angle\n")
      f.write("[dimensionless] | [s] | [dimensionless] | [ph/cm2] | [ph/cm2/s] | [dimensionless] | [dimensionless] | [dimensionless] | [keV] | [Gpc] | [erg/s] | [erg] | [°]\n")

    if self.version_long == 0 or self.version_long == 1:
      self.nlong = int(self.n_year * use_scipyquad(red_rate_long, self.zmin, self.zmax, func_args=(), x_logscale=False)[0])
    elif self.version_long == 2:
      self.nlong = int(self.n_year * use_scipyquad(red_rate_long_v2, self.zmin, self.zmax, func_args=(), x_logscale=False)[0])
    else:
      raise ValueError("Please use a correct number for version_long")

    if self.version_short == 0 or self.version_short == 1:
      self.nshort = int(self.n_year * use_scipyquad(red_rate_short, self.zmin, self.zmax, func_args=(self.short_rate,), x_logscale=False)[0])
    else:
      raise ValueError("Please use a correct number for version_long")

    # Long GRBs
    print("nlong : ", self.nlong)
    start_time = time()
    for ite in range(self.nlong):
      self.add_long(ite)
    print(f"Time taken for long bursts : {time() - start_time}s")

    # Short GRBs
    print("nshort : ", self.nshort)
    start_time = time()
    for ite in range(self.nshort):
      self.add_short(ite)
    print(f"Time taken for short bursts : {time() - start_time}s")

  def gbm_distri(self):
    """
    Creates the GBM distribution used for estimating the sample parameters. Distributions are obtained with a kde method on the GBM datasets
    """
    df_gbm_short = self.gbm_df.loc[self.gbm_df.T90 < 2]
    df_gbm_long = self.gbm_df.loc[self.gbm_df.T90 >= 2]

    # short_bools_ep = np.where(df_gbm_short.Type == 'GBM short flnc_comp', True, False)
    # short_bools_index = np.where(df_gbm_short.Type == 'GBM short flnc_comp', True, np.where(df_gbm_short.Type == 'GBM short flnc_plaw', True, False))

    # long_bools_ep = np.where(df_gbm_long.Type == 'GBM long flnc_band', True, np.where(df_gbm_long.Type == 'GBM long flnc_comp', True, np.where(df_gbm_long.Type == 'GBM long flnc_sbpl', True, False)))
    long_bools_ep = np.where(df_gbm_long.Type == 'GBM long flnc_band', True, False)
    long_bools_index = np.where(df_gbm_long.Type == 'GBM long flnc_band', True, np.where(df_gbm_long.Type == 'GBM long flnc_sbpl', True, False))

    # self.kde_short_log_ep_obs = gaussian_kde(np.log10(df_gbm_short[short_bools_ep].EpeakObs.values))
    self.kde_short_log_ep_obs = gaussian_kde(np.log10(df_gbm_short.EpeakObs.values[np.logical_not(np.isnan(df_gbm_short.EpeakObs.values))]))
    self.kde_short_log_t90 = gaussian_kde(np.log10(df_gbm_short.T90.values[np.logical_not(np.isnan(df_gbm_short.T90.values))]))
    # df_temp_index_s = df_gbm_short[short_bools_index]
    # self.kde_short_band_low = gaussian_kde(df_temp_index_s[df_temp_index_s.BandLow < 3].BandLow.values)
    # self.kde_short_band_high = gaussian_kde(df_temp_index_s[df_temp_index_s.BandHigh > -8].BandHigh.values)
    self.kde_short_band_low = gaussian_kde(df_gbm_short.BandLow.values[np.logical_not(np.isnan(df_gbm_short.BandLow.values))])
    self.kde_short_band_high = gaussian_kde(df_gbm_short.BandHigh.values[np.logical_not(np.isnan(df_gbm_short.BandHigh.values))])

    self.kde_long_log_ep_obs = gaussian_kde(np.log10(df_gbm_long[long_bools_ep].EpeakObs.values[np.logical_not(np.isnan(df_gbm_long[long_bools_ep].EpeakObs.values))]))
    self.kde_long_log_t90 = gaussian_kde(np.log10(df_gbm_long.T90.values[np.logical_not(np.isnan(df_gbm_long.T90.values))]))
    df_temp_index_l = df_gbm_long[long_bools_index]
    self.kde_long_band_low = gaussian_kde(df_temp_index_l[df_temp_index_l.BandLow < 0.5].BandLow.values[np.logical_not(np.isnan(df_temp_index_l[df_temp_index_l.BandLow < 0.5].BandLow.values))])
    self.kde_long_band_high = gaussian_kde(df_temp_index_l[df_temp_index_l.BandHigh > -8].BandHigh.values[np.logical_not(np.isnan(df_temp_index_l[df_temp_index_l.BandHigh > -8].BandHigh.values))])

  def add_short(self, sample_number):
    """
    Creates the quatities of a short burst according to distributions
    Based on Lana Salmon's thesis and Ghirlanda et al, 2016
    """
    if self.version_short == 0:
      ##################################################################################################################
      # picking according to distributions
      ##################################################################################################################
      z_obs_temp = acc_reject(red_rate_short, [self.short_rate], self.zmin, self.zmax)
      ep_obs_temp = 10**self.kde_short_log_ep_obs.resample(1)[0][0]
      ep_rest_temp = ep_obs_temp * (1 + z_obs_temp)

      band_low_obs_temp = self.kde_short_band_low.resample(1)[0][0]
      band_high_obs_temp = self.kde_short_band_high.resample(1)[0][0]
      if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
        ampl_norm = -1
      else:
        ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)
      while ampl_norm < 0:
        band_low_obs_temp = self.kde_short_band_low.resample(1)[0][0]
        band_high_obs_temp = self.kde_short_band_high.resample(1)[0][0]
        if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
          ampl_norm = -1
        else:
          ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)

      t90_obs_temp = 10 ** self.kde_short_log_t90.resample(1)[0][0]
      lc_temp = self.closest_lc(t90_obs_temp)
      times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
      pflux_to_mflux = np.mean(counts) / np.max(counts)

      ##################################################################################################################
      # Calculation other parameters with relations
      ##################################################################################################################
      dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
      eiso_rest_temp = amati_short(ep_rest_temp)
      lpeak_rest_temp = yonetoku_short(ep_rest_temp)  # / 2
    elif self.version_short == 1:
      ##################################################################################################################
      # picking according to distributions
      ##################################################################################################################
      z_obs_temp = acc_reject(red_rate_short, [self.short_rate], self.zmin, self.zmax)
      ep_rest_temp = acc_reject(epeak_distribution_short, [], self.epmin, self.epmax)
      ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)

      band_low_obs_temp = -0.6
      band_high_obs_temp = -2.5

      t90_obs_temp = 10 ** self.kde_short_log_t90.resample(1)[0][0]
      lc_temp = self.closest_lc(t90_obs_temp)
      times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
      pflux_to_mflux = np.mean(counts) / np.max(counts)

      dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
      eiso_rest_temp = amati_short(ep_rest_temp)
      lpeak_rest_temp = yonetoku_short(ep_rest_temp)  # / 2
    else:
      raise ValueError("Please use a correct number of version")

    ##################################################################################################################
    # Calculation of spectrum and data saving
    ##################################################################################################################
    ener_range = np.logspace(1, 3, 100001)
    # norm_val, spec, temp_peak_flux = norm_band_spec_calc(self.band_low_short, self.band_high_short, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(self.band_low_short, self.band_high_short, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    temp_mean_flux = temp_peak_flux * pflux_to_mflux

    data_row = pd.DataFrame(data=[[z_obs_temp, ep_obs_temp, ep_rest_temp, lpeak_rest_temp, temp_mean_flux, temp_peak_flux, t90_obs_temp, temp_mean_flux * t90_obs_temp, lc_temp, band_low_obs_temp, band_high_obs_temp,
                                   dl_obs_temp, eiso_rest_temp, "Sample short", "Sample"]], columns=self.columns)
    self.sample_df = pd.concat([self.sample_df, data_row], ignore_index=True)

    # self.thetaj_short.append(acc_reject(skewnorm.pdf, [2, 2.5, 3], self.thetaj_min, self.thetaj_max))
    self.save_grb(f"sGRB{self.n_year}S{sample_number}", t90_obs_temp, lc_temp, temp_mean_flux * t90_obs_temp,
                  temp_mean_flux, z_obs_temp, self.band_low_short, self.band_high_short,
                  ep_rest_temp, dl_obs_temp, lpeak_rest_temp, eiso_rest_temp, 0)

  def add_long(self, sample_number):
    """
    Creates the quatities of a long burst according to distributions
    Based on Sarah Antier's thesis
    """
    if self.version_long == 0:
      ##################################################################################################################
      # picking according to distributions
      ##################################################################################################################
      z_obs_temp = acc_reject(red_rate_long, [], self.zmin, self.zmax)
      ep_obs_temp = 10**self.kde_long_log_ep_obs.resample(1)[0][0]
      ep_rest_temp = ep_obs_temp * (1 + z_obs_temp)

      band_low_obs_temp = self.kde_long_band_low.resample(1)[0][0]
      band_high_obs_temp = self.kde_long_band_high.resample(1)[0][0]
      if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
        ampl_norm = -1
      else:
        ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)
      while ampl_norm < 0:
        band_low_obs_temp = self.kde_long_band_low.resample(1)[0][0]
        band_high_obs_temp = self.kde_long_band_high.resample(1)[0][0]
        if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
          ampl_norm = -1
        else:
          ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)

      # With a distribution : the value is taken in a log distribution and then put back in linear value
      t90_obs_temp = 10 ** self.kde_long_log_t90.resample(1)[0][0]
      lc_temp = self.closest_lc(t90_obs_temp)
      times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
      pflux_to_mflux = np.mean(counts) / np.max(counts)

      ##################################################################################################################
      # Calculation other parameters with relations
      ##################################################################################################################
      dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).to_value("Gpc")
      eiso_rest_temp = amati_long(ep_rest_temp)
      lpeak_rest_temp = yonetoku_long(ep_rest_temp)
    elif self.version_long == 1:
      ##################################################################################################################
      # picking according to distributions
      ##################################################################################################################
      z_obs_temp = acc_reject(red_rate_long, [], self.zmin, self.zmax)
      lpeak_rest_temp = acc_reject(lpeak_function_long, [], self.lmin, self.lmax)  # / 3.5

      band_low_obs_temp = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
      band_high_obs_temp = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
      if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
        ampl_norm = -1
      else:
        ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)
      while ampl_norm < 0:
        band_low_obs_temp = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
        band_high_obs_temp = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
        if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
          ampl_norm = -1
        else:
          ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)


      t90_obs_temp = 10 ** self.kde_long_log_t90.resample(1)[0][0]
      lc_temp = self.closest_lc(t90_obs_temp)
      times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
      pflux_to_mflux = np.mean(counts) / np.max(counts)

      dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
      ep_rest_temp = yonetoku_reverse_long(lpeak_rest_temp)
      ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)
      eiso_rest_temp = amati_long(ep_rest_temp)
    elif self.version_long == 2:
      ##################################################################################################################
      # picking according to distributions
      ##################################################################################################################
      z_obs_temp = acc_reject(red_rate_long_v2, [], self.zmin, self.zmax)
      lpeak_rest_temp = acc_reject(lpeak_function_long_v2, [z_obs_temp], self.lmin, self.lmax)  # / 3.5

      band_low_obs_temp = -0.9
      band_high_obs_temp = -2.2
      # band_low_obs_temp = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
      # band_high_obs_temp = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
      # if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
      #   ampl_norm = -1
      # else:
      #   ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)
      # while ampl_norm < 0:
      #   band_low_obs_temp = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
      #   band_high_obs_temp = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
      #   if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
      #     ampl_norm = -1
      #   else:
      #     ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)

      t90_obs_temp = 10 ** self.kde_long_log_t90.resample(1)[0][0]
      lc_temp = self.closest_lc(t90_obs_temp)
      times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
      pflux_to_mflux = np.mean(counts) / np.max(counts)

      dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
      ep_rest_temp = yonetoku_reverse_long_v2(lpeak_rest_temp)
      ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)
      eiso_rest_temp = amati_long(ep_rest_temp)
    elif self.version_long == 3:
      ##################################################################################################################
      # picking according to distributions
      ##################################################################################################################
      z_obs_temp = acc_reject(red_rate_long_v2, [], self.zmin, self.zmax)
      lpeak_rest_temp = acc_reject(lpeak_function_long_v2, [z_obs_temp], self.lmin, self.lmax)  # / 3.5

      band_low_obs_temp = -0.9
      band_high_obs_temp = -2.2
      # if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
      #   ampl_norm = -1
      # else:
      #   ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)
      # while ampl_norm < 0:
      #   band_low_obs_temp = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
      #   band_high_obs_temp = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
      #   if (band_low_obs_temp - band_high_obs_temp) / (band_low_obs_temp + 2) < 0:
      #     ampl_norm = -1
      #   else:
      #     ampl_norm = normalisation_calc(band_low_obs_temp, band_high_obs_temp)

      t90_obs_temp = 10 ** self.kde_long_log_t90.resample(1)[0][0]
      lc_temp = self.closest_lc(t90_obs_temp)
      times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
      pflux_to_mflux = np.mean(counts) / np.max(counts)

      dl_obs_temp = self.cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc
      ep_rest_temp = yonetoku_reverse_long_v2(lpeak_rest_temp)
      ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)
      eiso_rest_temp = amati_long(ep_rest_temp)
    else:
      raise ValueError("Please use a correct number of version")

    ##################################################################################################################
    # Calculation of spectrum and data saving
    ##################################################################################################################
    ener_range = np.logspace(1, 5, 100001)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    temp_mean_flux = temp_peak_flux * pflux_to_mflux

    # pb : flux int = 1 entre 0 et inf mais ici entre 1e1 et 1e3... il y a certainement un rapport a prendre
    # voir pk diff entre les 2 manière de faire un spectre
    # comparer val totale de flux avec K ep**2

    # stop
    data_row = pd.DataFrame(data=[[z_obs_temp, ep_obs_temp, ep_rest_temp, lpeak_rest_temp, temp_mean_flux, temp_peak_flux, t90_obs_temp, temp_mean_flux * t90_obs_temp, lc_temp, band_low_obs_temp, band_high_obs_temp, dl_obs_temp, eiso_rest_temp, "Sample long",
                                   "Sample"]], columns=self.columns)
    self.sample_df = pd.concat([self.sample_df, data_row], ignore_index=True)

    self.save_grb(f"lGRB{self.n_year}S{sample_number}", t90_obs_temp, lc_temp, temp_mean_flux * t90_obs_temp, temp_mean_flux, z_obs_temp, band_low_obs_temp, band_high_obs_temp, ep_rest_temp, dl_obs_temp, lpeak_rest_temp, eiso_rest_temp, 0)

  def closest_lc(self, searched_time):
    """
    Find the lightcurve with a duration which is the closest to the sampled t90 time
    """
    abs_diff = np.abs(np.array(self.gbm_cat.df.t90, dtype=float) - searched_time)
    gbm_index = np.argmin(abs_diff)
    # print(searched_time, float(self.gbm_cat.t90[gbm_index]))
    return f"LightCurve_{self.gbm_cat.df.name[gbm_index]}.dat"

  def save_grb(self, name, t90, lcname, fluence, mean_flux, red, band_low, band_high, ep, dl, lpeak, eiso, thetaj):
    """
    Saves a GRB in a catalog file
    """
    with open(self.filename, "a") as f:
      f.write(f"{name}|{t90}|{lcname}|{fluence}|{mean_flux}|{red}|{band_low}|{band_high}|{ep}|{dl}|{lpeak}|{eiso}|{thetaj}\n")

  def short_comparison(self):
    """
    Compare the distribution of the created quatities and the seed distributions
    """
    zx_lana = [0.1717557251908397, 0.4961832061068702, 0.8301526717557253, 1.1641221374045803, 1.4885496183206108,
               1.8225190839694656, 2.1469465648854964, 2.471374045801527, 2.8053435114503817, 3.1393129770992365,
               3.463740458015267, 3.7881679389312977, 4.112595419847328, 4.456106870229008, 4.770992366412214,
               5.104961832061069, 5.438931297709924, 5.763358778625954, 6.087786259541985, 6.412213740458015,
               6.7461832061068705, 7.080152671755726, 7.404580152671756, 7.729007633587787, 8.053435114503817,
               8.377862595419847, 8.711832061068703, 9.045801526717558, 9.379770992366412, 9.694656488549619]
    zy_lana = [30.959302325581397, 48.98255813953489, 68.96802325581396, 89.8982558139535, 84.95639534883722,
               101.96220930232559, 91.06104651162791, 81.03197674418605, 60.10174418604652, 44.98546511627907,
               44.84011627906977, 33.06686046511628, 34.0843023255814, 29.86918604651163, 23.76453488372093,
               19.98546511627907, 11.84593023255814, 13.880813953488373, 20.857558139534884, 7.921511627906977,
               6.758720930232559, 1.9622093023255816, 6.904069767441861, 8.06686046511628, 5.741279069767442,
               5.886627906976744, 3.997093023255814, 5.886627906976744, 4.869186046511628, 4.869186046511628]
    dlx_lana = [1.8921475875118259, 5.392620624408704, 8.987701040681173, 12.393566698202461, 15.894039735099339,
                19.489120151371807, 22.989593188268685, 26.584673604541155, 30.085146641438033, 33.6802270577105,
                37.27530747398297, 40.77578051087985, 44.276253547776726, 47.96594134342479, 51.27719962157048,
                54.872280037842955, 58.46736045411542, 62.062440870387896, 65.56291390728477, 69.06338694418164,
                72.65846736045413, 76.0643330179754, 79.65941343424788, 83.15988647114476, 86.66035950804164,
                90.35004730368969, 93.85052034058657, 97.44560075685904, 100.85146641438033, 104.3519394512772]
    dly_lana = [73.01587301587301, 92.06349206349206, 105.05050505050504, 109.95670995670994, 103.03030303030302,
                94.08369408369407, 63.059163059163055, 47.907647907647906, 50.93795093795094, 28.86002886002886,
                35.93073593073593, 33.04473304473304, 24.098124098124096, 18.75901875901876, 17.027417027417027,
                9.956709956709956, 13.997113997113996, 15.873015873015872, 7.936507936507936, 5.916305916305916,
                2.02020202020202, 4.906204906204906, 7.936507936507936, 7.07070707070707, 5.916305916305916,
                2.8860028860028857, 3.8961038961038956, 3.8961038961038956, 5.916305916305916, 3.8961038961038956]
    epx_lana = [114.22704035961927, 148.3592882128584, 191.80879893956018, 244.59423414764007, 313.34007010386443,
                405.1069753046362, 516.5916835949072, 664.8277877836776, 855.6002766702244, 1106.177196031111,
                1423.5950006326557, 1832.09591835528, 2357.81626976942, 3048.3423874059117, 3887.2407094435803,
                5002.685338365965, 6497.540631502288, 8285.652083699546, 10761.492564694065, 13660.234894889703]
    epy_lana = [8.734939759036145, 0, 22.289156626506028, 0, 25.903614457831328, 31.927710843373497, 35.8433734939759,
                62.65060240963856, 97.89156626506025, 150.90361445783134, 224.69879518072293, 157.83132530120483,
                94.57831325301206, 44.87951807228916, 19.87951807228916, 12.951807228915664, 1.8072289156626509,
                1.8072289156626509, 1.2048192771084338, 0]
    eisox_lana = [-1.735223160434258, -1.6375150784077201, -1.5379975874547647, -1.436670687575392, -1.3371531966224366,
                  -1.235826296743064, -1.1381182147165259, -1.040410132689988, -0.9390832328106152, -0.8413751507840772,
                  -0.7400482509047045, -0.6369119420989143, -0.5410132689987938, -0.4414957780458384,
                  -0.34016887816646557, -0.244270205066345, -0.14294330518697218, -0.043425814234016924,
                  0.05428226779252121, 0.15542168674698797]
    eisoy_lana = [8.433734939759036, 0, 22.590361445783135, 0, 25.30120481927711, 31.32530120481928, 35.5421686746988,
                  62.65060240963856, 97.59036144578315, 150.60240963855424, 224.69879518072293, 157.53012048192772,
                  94.57831325301206, 44.578313253012055, 19.87951807228916, 12.650602409638555, 1.8072289156626509,
                  1.8072289156626509, 0.9036144578313254, 0.9036144578313254]
    thx_lana = [0.733695652173913, 1.25, 1.7391304347826086, 2.2282608695652173, 2.717391304347826, 3.233695652173913,
                3.7228260869565215, 4.21195652173913, 4.701086956521739, 5.217391304347826, 5.706521739130435,
                6.195652173913043, 6.684782608695652, 7.201086956521739, 7.690217391304348, 8.179347826086957,
                8.668478260869565, 9.184782608695652, 9.67391304347826, 10.16304347826087]
    thy_lana = [38.93280632411067, 27.865612648221344, 48.81422924901186, 57.905138339920946, 93.08300395256917,
                97.82608695652173, 82.80632411067194, 93.08300395256917, 85.96837944664031, 95.8498023715415,
                57.11462450592885, 82.80632411067194, 44.86166007905138, 29.841897233201582, 19.960474308300395,
                11.660079051383399, 14.82213438735178, 7.707509881422925, 2.9644268774703555, 3.7549407114624507]

    df_short = self.sample_df.loc[self.sample_df.Type == "Sample short"]
    n_sample = len(df_short)

    df_gbm_short = self.gbm_df.loc[self.gbm_df.T90 < 2]

    comp_fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(27, 12))
    ax1.hist(df_short.Redshift, bins=45, histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax1.step(zx_lana, zy_lana, color="red", label="Model")
    ax1.set(title="Redshift distributions", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale="linear")
    ax1.legend()

    ax2.hist(df_short.LuminosityDistance, bins=45, histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax2.step(dlx_lana, dly_lana, color="red", label="Model")
    ax2.set(title="DL distributions", xlabel="Luminosity distance (Gpc)", ylabel="Number of GRB", xscale="linear", yscale="linear")
    ax2.legend()

    ax3.hist(df_short.Epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), 56), histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax3.step(epx_lana, epy_lana, color="red", label="Model")
    ax3.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale="linear")
    ax3.legend()

    # ax4.hist(self.ep_o_short, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), 56), histtype="step", color="blue", weights=[len(self.short_gbm_epeak) / n_sample] * n_sample, label="Distribution")
    ax4.hist(df_gbm_short.Epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), 56), histtype="step", color="red", label="Model (GBM)")
    ax4.set(title="Ep obs distributions", xlabel="Obs frame peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale="linear")
    ax4.legend()

    ax4.hist(np.log10(np.array(df_short.EnergyIso) / 1e52), bins=28, histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax4.step(eisox_lana, eisoy_lana, color="red", label="Model")
    ax4.set(title="Eiso distributions", xlabel="Log10(Eiso/1e52) (erg)", ylabel="Number of GRB", xscale="linear", yscale="linear")
    ax4.legend()

    # ax6.hist(self.thetaj_short, bins=np.linspace(0, 15, 30), histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax6.step(thx_lana, thy_lana, color="red", label="Model")
    ax6.set(title="thetaj distributions", xlabel="Thetaj (°)", ylabel="Number of GRB", xscale="linear", yscale="linear")
    ax6.legend()

  def short_distri(self, yscale="log", nbin=50):
    """
    Compare the distribution of the created quatities and the seed distributions
    """
    fluence_min, fluence_max = 1e-8, 1e4
    flux_min, flux_max = 1e-8, 1e5

    df_short = self.sample_df.loc[self.sample_df.Type == "Sample short"]
    n_sample = len(df_short)

    df_gbm_short = self.gbm_df.loc[self.gbm_df.T90 < 2]
    n_gbm = len(df_gbm_short)

    comp_fig, ((ax1, ax2, ax3, ax1r), (ax4, ax5, ax6, ax2r)) = plt.subplots(2, 4, figsize=(27, 12))

    ax1.hist(df_short.EpeakObs, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax1.hist(df_gbm_short.EpeakObs, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax1.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1.legend()
    ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    alpha_bin = np.linspace(np.min(df_short.BandLow), np.max(df_short.BandLow), nbin)
    ax2.hist(df_short.BandLow, bins=alpha_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax2.hist(df_gbm_short.BandLow, bins=alpha_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax2.set(title="Band low energy index", xlabel="Alpha", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax2.legend()
    ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    beta_bin = np.linspace(np.min(df_short.BandHigh), np.max(df_short.BandHigh), nbin)
    ax3.hist(df_short.BandHigh, bins=beta_bin, histtype="step", color="green", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax3.hist(df_gbm_short.BandHigh, bins=beta_bin, histtype="step", color="orange", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax3.set(title="Band high energy index", xlabel="Beta", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax3.legend()
    ax3.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax3.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    fluence_bin = np.logspace(np.log10(fluence_min), np.log10(fluence_max), nbin)
    ax4.hist(df_short.Fluence, bins=fluence_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax4.hist(df_gbm_short.Fluence, bins=fluence_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax4.set(title="Fluence distributions", xlabel="Photon fluence (photon/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax4.legend()
    ax4.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax4.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    flux_bin = np.logspace(np.log10(flux_min), np.log10(flux_max), nbin)
    ax5.hist(df_short.MeanFlux, bins=flux_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax5.hist(df_gbm_short.MeanFlux, bins=flux_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax5.set(title="Mean flux distributions", xlabel="Photon flux (photon/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax5.legend()
    ax5.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax5.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    ax6.hist(df_short.T90, bins=np.logspace(-3, np.log10(2), nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax6.hist(df_gbm_short.T90, bins=np.logspace(-3, np.log10(2), nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax6.set(title="T90 distributions", xlabel="T90 (s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax6.legend()
    ax6.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax6.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    ax1r.hist(df_short.Redshift, bins=np.linspace(self.zmin, self.zmax, nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax1r.hist(df_gbm_short.Redshift, bins=np.linspace(self.zmin, self.zmax, nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax1r.set(title="Redshift distribution", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax1r.legend()
    ax1r.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1r.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    lum_bin = np.logspace(np.log10(self.lmin), np.log10(self.lmax), nbin)
    ax2r.hist(df_short.PeakLuminosity, bins=lum_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax2r.hist(df_gbm_short.PeakLuminosity, bins=lum_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax2r.set(title="Peak luminosity distribution", xlabel="Peak Luminosity (erg/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax2r.legend()
    ax2r.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2r.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    plt.suptitle(f"version : {self.version_short}")
    plt.show()

  def long_distri(self, yscale="log", nbin=50):
    """
    Compare the distribution of the created quatities and the seed distributions
    """
    fluence_min, fluence_max = 1e-8, 1e4
    flux_min, flux_max = 1e-8, 1e5

    df_long = self.sample_df.loc[self.sample_df.Type == "Sample long"]
    n_sample = len(df_long)

    df_gbm_long = self.gbm_df.loc[self.gbm_df.T90 >= 2]
    n_gbm = len(df_gbm_long)

    comp_fig, ((ax1, ax2, ax3, ax1r), (ax4, ax5, ax6, ax2r)) = plt.subplots(2, 4, figsize=(27, 12))

    ax1.hist(df_long.EpeakObs, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax1.hist(df_gbm_long.EpeakObs, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax1.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1.legend()
    ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    alpha_bin = np.linspace(np.min(df_long.BandLow), np.max(df_long.BandLow), nbin)
    ax2.hist(df_long.BandLow, bins=alpha_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax2.hist(df_gbm_long.BandLow, bins=alpha_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax2.set(title="Band low energy index", xlabel="Alpha", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax2.legend()
    ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    beta_bin = np.linspace(np.min(df_long.BandHigh), np.max(df_long.BandHigh), nbin)
    ax3.hist(df_long.BandHigh, bins=beta_bin, histtype="step", color="green", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax3.hist(df_gbm_long.BandHigh, bins=beta_bin, histtype="step", color="orange", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax3.set(title="Band high energy index", xlabel="Beta", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax3.legend()
    ax3.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax3.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    fluence_bin = np.logspace(np.log10(fluence_min), np.log10(fluence_max), nbin)
    ax4.hist(df_long.Fluence, bins=fluence_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax4.hist(df_gbm_long.Fluence, bins=fluence_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax4.set(title="Fluence distributions", xlabel="Photon fluence (photon/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax4.legend()
    ax4.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax4.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    flux_bin = np.logspace(np.log10(flux_min), np.log10(flux_max), nbin)
    ax5.hist(df_long.MeanFlux, bins=flux_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax5.hist(df_gbm_long.MeanFlux, bins=flux_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax5.set(title="Mean flux distributions", xlabel="Photon flux (photon/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax5.legend()
    ax5.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax5.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    ax6.hist(df_long.T90, bins=np.logspace(np.log10(2), 3, nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax6.hist(df_gbm_long.T90, bins=np.logspace(np.log10(2), 3, nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax6.set(title="T90 distributions", xlabel="T90 (s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax6.legend()
    ax6.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax6.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    ax1r.hist(df_long.Redshift, bins=np.linspace(self.zmin, self.zmax, nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax1r.hist(df_gbm_long.Redshift, bins=np.linspace(self.zmin, self.zmax, nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax1r.set(title="Redshift distribution", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax1r.legend()
    ax1r.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax1r.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    lum_bin = np.logspace(np.log10(self.lmin), np.log10(self.lmax), nbin)
    ax2r.hist(df_long.PeakLuminosity, bins=lum_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax2r.hist(df_gbm_long.PeakLuminosity, bins=lum_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax2r.set(title="Peak luminosity distribution", xlabel="Peak Luminosity (erg/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax2r.legend()
    ax2r.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    ax2r.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    plt.suptitle(f"version : {self.version_long}")
    plt.show()

  def red_lum_dist(self, yscale="log", nbin=50):
    df_long = self.sample_df.loc[self.sample_df.Type == "Sample long"]
    n_sample_l = len(df_long)

    df_short = self.sample_df.loc[self.sample_df.Type == "Sample short"]
    n_sample_s = len(df_short)

    lum_bin = np.logspace(np.log10(self.lmin), np.log10(self.lmax), nbin)

    red_fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2, figsize=(27, 12))
    axs1.hist(df_long.Redshift, bins=np.linspace(self.zmin, self.zmax, nbin), histtype="step", color="blue", label=f"Sample, {n_sample_l} GRB", weights=[self.sample_weight] * n_sample_l)
    axs1.set(title="Redshift distribution", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    axs1.legend()
    axs1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    axs1.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)
    plt.suptitle(f"Long redshift version : {self.version_long}")

    axs2.hist(df_long.PeakLuminosity, bins=lum_bin, histtype="step", color="blue", label=f"Sample, {n_sample_l} GRB", weights=[self.sample_weight] * n_sample_l)
    axs2.set(title="Peak luminosity distribution", xlabel="Peak Luminosity (erg/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    axs2.legend()
    axs2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    axs2.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    axs3.hist(df_short.Redshift, bins=np.linspace(self.zmin, self.zmax, nbin), histtype="step", color="blue", label=f"Sample, {n_sample_s} GRB", weights=[self.sample_weight] * n_sample_s)
    axs3.set(title="Redshift distribution", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    axs3.legend()
    axs3.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    axs3.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)

    axs4.hist(df_short.PeakLuminosity, bins=lum_bin, histtype="step", color="blue", label=f"Sample, {n_sample_s} GRB", weights=[self.sample_weight] * n_sample_s)
    axs4.set(title="Peak luminosity distribution", xlabel="Peak Luminosity (erg/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    axs4.legend()
    axs4.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
    axs4.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)
    plt.suptitle(f"Short redshift version : {self.version_short}")

    plt.show()

  def distri_pairplot(self):
    fraction = self.n_year/10/0.6
    selec_cols = ["Redshift", "Epeak", "PeakLuminosity", "MeanFlux", "T90", "Fluence", "Type"]
    if fraction < 1:
      selec_gbm_df = self.gbm_df[selec_cols].sample(frac=fraction)
      selec_samp_df = self.sample_df[selec_cols]
    else:
      selec_gbm_df = self.gbm_df[selec_cols]
      selec_samp_df = self.sample_df[selec_cols].sample(frac=1/fraction)
    plot_df = pd.concat([selec_samp_df, selec_gbm_df], ignore_index=True)
    plot_df[["Epeak", "PeakLuminosity", "MeanFlux", "T90", "Fluence"]] = plot_df[["Epeak", "PeakLuminosity", "MeanFlux", "T90", "Fluence"]].apply(np.log10, axis=1)
    plot_df.rename(columns={"Epeak": "LogEpeak", "PeakLuminosity": "LogPeakLuminosity", "MeanFlux": "LogMeanFlux", "T90": "LogT90", "Fluence": "LogFluence"})

    colors = ["lightskyblue", "navy", "lightgreen", "limegreen", "forestgreen", "darkgreen", "thistle", "plum", "violet", "purple"]
    order = ['Sample long', 'Sample short', 'GBM long flnc_band', 'GBM long flnc_comp', 'GBM long flnc_sbpl', 'GBM long flnc_plaw', 'GBM short flnc_band', 'GBM short flnc_comp', 'GBM short flnc_sbpl', 'GBM short flnc_plaw']
    p1 = sns.pairplot(plot_df, hue="Type", hue_order=order, corner=True, palette=colors, plot_kws={'s': 10})

    selec_cols = ["MeanFlux", "T90", "Fluence", "Cat"]
    if fraction < 1:
      selec_gbm_df = self.gbm_df[selec_cols].sample(frac=fraction)
      selec_samp_df = self.sample_df[selec_cols]
    else:
      selec_gbm_df = self.gbm_df[selec_cols]
      selec_samp_df = self.sample_df[selec_cols].sample(frac=1/fraction)
    plot_df = pd.concat([selec_samp_df, selec_gbm_df], ignore_index=True)
    plot_df[["MeanFlux", "T90", "Fluence"]] = plot_df[["MeanFlux", "T90", "Fluence"]].apply(np.log10, axis=1)
    plot_df.rename(columns={"MeanFlux": "LogMeanFlux", "T90": "LogT90", "Fluence": "LogFluence"})

    sns.pairplot(plot_df, hue="Cat", corner=True, palette=["lightskyblue", "lightgreen"], plot_kws={'s': 10})

    selec_cols = ["Epeak", "MeanFlux", "T90", "Fluence", "Type"]
    if fraction < 1:
      selec_gbm_df = self.gbm_df[selec_cols].sample(frac=fraction)
      selec_samp_df = self.sample_df[selec_cols]
    else:
      selec_gbm_df = self.gbm_df[selec_cols]
      selec_samp_df = self.sample_df[selec_cols].sample(frac=1/fraction)
    plot_df = pd.concat([selec_samp_df, selec_gbm_df], ignore_index=True)
    plot_df[["Epeak", "MeanFlux", "T90", "Fluence"]] = plot_df[["Epeak", "MeanFlux", "T90", "Fluence"]].apply(np.log10, axis=1)
    plot_df.rename(columns={"Epeak": "LogEpeak", "MeanFlux": "LogMeanFlux", "T90": "LogT90", "Fluence": "LogFluence"})

    sns.pairplot(plot_df, hue="Type", hue_order=order, corner=True, palette=colors, plot_kws={'s': 10})

  def distri_gbm_spectral_type(self):
    selec_cols = ["Epeak", "BandLow", "BandHigh", "MeanFlux", "Fluence", "Type"]
    full_df = self.gbm_df[selec_cols]
    self.tt = full_df
    # full_df[["Epeak", "MeanFlux", "Fluence"]] = full_df[["Epeak", "MeanFlux", "Fluence"]].apply(np.log10, axis=1)
    # full_df.rename(columns={"Epeak": "LogEpeak", "MeanFlux": "LogMeanFlux", "Fluence": "LogFluence"})
    # short_bools = np.where(full_df.Type == 'GBM short flnc_band', True, np.where(full_df.Type == 'GBM short flnc_comp', True, np.where(full_df.Type == 'GBM short flnc_sbpl', True, np.where(full_df.Type == 'GBM short flnc_plaw', True, False))))
    # long_bools = np.where(full_df.Type == 'GBM long flnc_band', True, np.where(full_df.Type == 'GBM long flnc_comp', True, np.where(full_df.Type == 'GBM long flnc_sbpl', True, np.where(full_df.Type == 'GBM long flnc_plaw', True, False))))
    # short_df = full_df[short_bools]
    # long_df = full_df[long_bools]


    #
    # Short burst
    #
    df_short_comp = full_df[full_df.Type == 'GBM short flnc_comp']
    df_short_band = full_df[full_df.Type == 'GBM short flnc_band']
    df_short_sbpl = full_df[full_df.Type == 'GBM short flnc_sbpl']
    df_short_plaw = full_df[full_df.Type == 'GBM short flnc_plaw']

    col_plot = ["Epeak", "BandLow", "BandHigh"]
    colors = ["lightskyblue", "navy", "lightgreen", "forestgreen"]
    nbin = 30
    scales = ["log", "linear", "linear"]

    figdist, axs = plt.subplots(1, 3, figsize=(27, 6))
    for ite_ax, ax in enumerate(axs):
      liminf = np.min(full_df[col_plot[ite_ax]])
      limsup = np.max(full_df[col_plot[ite_ax]])
      if scales[ite_ax] == "log":
        bins = np.logspace(np.log10(liminf), np.log10(limsup), nbin)
        ax.set(xscale=scales[ite_ax], yscale="log", xlabel=f"log {col_plot[ite_ax]}", ylabel=f"log number of count")
      elif scales[ite_ax] == "linear":
        bins = np.linspace(liminf, limsup, nbin)
        ax.set(xscale=scales[ite_ax], yscale="log", xlabel=f"{col_plot[ite_ax]}", ylabel=f"log number of count")
      else:
        raise ValueError('Wrong scale used')
      ax.hist(df_short_comp[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[0], label='GBM short flnc_comp')
      ax.hist(df_short_band[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[1], label='GBM short flnc_band')
      ax.hist(df_short_sbpl[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[2], label='GBM short flnc_sbpl')
      ax.hist(df_short_plaw[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[3], label='GBM short flnc_plaw')
    axs[0].legend()

    col_plot = ["Epeak", "MeanFlux", "Fluence"]
    colors = ["lightskyblue", "navy", "lightgreen", "forestgreen"]
    nbin = 30
    scales = ["log", "log", "log"]
    figdist, axs = plt.subplots(1, 3, figsize=(27, 6))
    for ite_ax, ax in enumerate(axs):
      liminf = np.min(full_df[col_plot[ite_ax]])
      limsup = np.max(full_df[col_plot[ite_ax]])
      if scales[ite_ax] == "log":
        bins = np.logspace(np.log10(liminf), np.log10(limsup), nbin)
        ax.set(xscale=scales[ite_ax], yscale="log", xlabel=f"log {col_plot[ite_ax]}", ylabel=f"log number of count")
      elif scales[ite_ax] == "linear":
        bins = np.linspace(liminf, limsup, nbin)
        ax.set(xscale=scales[ite_ax], yscale="log", xlabel=f"{col_plot[ite_ax]}", ylabel=f"log number of count")
      else:
        raise ValueError('Wrong scale used')
      ax.hist(df_short_comp[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[0], label='GBM short flnc_comp')
      ax.hist(df_short_band[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[1], label='GBM short flnc_band')
      ax.hist(df_short_sbpl[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[2], label='GBM short flnc_sbpl')
      ax.hist(df_short_plaw[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[3], label='GBM short flnc_plaw')
    axs[0].legend()

    #
    # Long burst
    #
    col_plot = ["Epeak", "BandLow", "BandHigh"]
    colors = ["lightskyblue", "navy", "lightgreen", "forestgreen"]
    nbin = 30
    scales = ["log", "linear", "linear"]

    df_long_comp = full_df[full_df.Type == 'GBM long flnc_comp']
    df_long_band = full_df[full_df.Type == 'GBM long flnc_band']
    df_long_sbpl = full_df[full_df.Type == 'GBM long flnc_sbpl']
    df_long_plaw = full_df[full_df.Type == 'GBM long flnc_plaw']

    figdist, axs = plt.subplots(1, 3, figsize=(27, 6))
    for ite_ax, ax in enumerate(axs):
      liminf = np.min(full_df[col_plot[ite_ax]])
      limsup = np.max(full_df[col_plot[ite_ax]])
      if scales[ite_ax] == "log":
        bins = np.logspace(np.log10(liminf), np.log10(limsup), nbin)
        ax.set(xscale=scales[ite_ax], yscale="log", xlabel=f"log {col_plot[ite_ax]}", ylabel=f"log number of count")
      elif scales[ite_ax] == "linear":
        bins = np.linspace(liminf, limsup, nbin)
        ax.set(xscale=scales[ite_ax], yscale="log", xlabel=f"{col_plot[ite_ax]}", ylabel=f"log number of count")
      else:
        raise ValueError('Wrong scale used')
      ax.hist(df_long_comp[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[0], label='GBM long flnc_comp')
      ax.hist(df_long_band[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[1], label='GBM long flnc_band')
      ax.hist(df_long_sbpl[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[2], label='GBM long flnc_sbpl')
      ax.hist(df_long_plaw[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[3], label='GBM long flnc_plaw')
    axs[0].legend()

    col_plot = ["Epeak", "MeanFlux", "Fluence"]
    colors = ["lightskyblue", "navy", "lightgreen", "forestgreen"]
    nbin = 30
    scales = ["log", "log", "log"]
    figdist, axs = plt.subplots(1, 3, figsize=(27, 6))
    for ite_ax, ax in enumerate(axs):
      liminf = np.min(full_df[col_plot[ite_ax]])
      limsup = np.max(full_df[col_plot[ite_ax]])
      if scales[ite_ax] == "log":
        bins = np.logspace(np.log10(liminf), np.log10(limsup), nbin)
        ax.set(xscale=scales[ite_ax], yscale="log", xlabel=f"log {col_plot[ite_ax]}", ylabel=f"log number of count")
      elif scales[ite_ax] == "linear":
        bins = np.linspace(liminf, limsup, nbin)
        ax.set(xscale=scales[ite_ax], yscale="log", xlabel=f"{col_plot[ite_ax]}", ylabel=f"log number of count")
      else:
        raise ValueError('Wrong scale used')
      ax.hist(df_long_comp[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[0], label='GBM long flnc_comp')
      ax.hist(df_long_band[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[1], label='GBM long flnc_band')
      ax.hist(df_long_sbpl[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[2], label='GBM long flnc_sbpl')
      ax.hist(df_long_plaw[col_plot[ite_ax]].values, histtype="step", bins=bins, color=colors[3], label='GBM long flnc_plaw')
    axs[0].legend()

# comparer Ep band et comp
# comp et plaw semblent avoir un flux plus élevé
#
# voir la comparaison avec uniquement les GRB avec un spectre de band ?
#
# epcomp = []
# epband = []
# ep_all = []
# for ite, model in enumerate(smp.gbm_cat.flnc_best_fitting_model):
#   smodel = model.strip()
#   if smodel == "flnc_comp":
#     epcomp.append(float(smp.gbm_cat.flnc_comp_epeak[ite]))
#   elif smodel == "flnc_band":
#     epband.append(float(smp.gbm_cat.flnc_band_epeak[ite]))
#   if smp.gbm_cat.flnc_band_epeak[ite].strip() != "":
#     ep_all.append(float(smp.gbm_cat.flnc_band_epeak[ite]))
# epmin = np.min(ep_all)
# epmax = np.max(ep_all)
# bins = np.logspace(np.log10(epmin), np.log10(epmax), 30)
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# ax.hist(epcomp, bins=bins, color="blue", label="ep comp")
# ax.hist(epband, bins=bins, color="green", label="ep band")
# ax.set(xscale="log", yscale="log", xlabel="Epeak", ylabel="Number")
# ax.legend()
# plt.show()


# def add_short(self, sample_number, version):
#   """
#   Creates the quatities of a short burst according to distributions
#   Based on Lana Salmon's thesis and Ghirlanda et al, 2016
#   """
#   ##################################################################################################################
#   # picking according to distributions
#   ##################################################################################################################
#   z_temp = acc_reject(red_rate_short, [self.short_rate, version], self.zmin, self.zmax)
#   ep_temp = acc_reject(epeak_distribution_short, [version], self.epmin, self.epmax)
#   ep_obs_temp = ep_temp / (1 + z_temp)
#   temp_t90 = acc_reject(t90_short_distri, [version], 1e-3, 2)
#   lc_temp = self.closest_lc(temp_t90)
#   ##################################################################################################################
#   # Calculation other parameters with relations
#   ##################################################################################################################
#   lpeak_temp = yonetoku_short(ep_temp, version)  # / 2
#   dl_temp = self.cosmo.luminosity_distance(z_temp).value / 1000  # Gpc
#   eiso_temp = amati_short(ep_temp, version)
#   temp_band_high = -0.6
#   temp_band_low = -2.5
#
#   ##################################################################################################################
#   # Calculation of spectrum
#   ##################################################################################################################
#   ener_range = np.logspace(1, 3, 100001)
#   norm_val, spec = norn_band_spec_calc(self.band_low_short, self.band_high_short, z_temp, dl_temp, ep_temp, lpeak_temp, ener_range)
#   temp_peak_flux = trapezoid(spec, ener_range)
#
#   times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
#   temp_mean_flux = temp_peak_flux / np.max(counts) * np.mean(counts)
#
#   data_row = pd.DataFrame(data=[[z_temp, ep_obs_temp, ep_temp, lpeak_temp, temp_mean_flux, temp_peak_flux, temp_t90, temp_mean_flux * temp_t90, lc_temp, temp_band_low, temp_band_high, dl_temp, eiso_temp, "Sample short",
#                                  "Sample"]], columns=self.columns)
#   self.sample_df = pd.concat([self.sample_df, data_row], ignore_index=True)
#
#   # self.thetaj_short.append(acc_reject(skewnorm.pdf, [2, 2.5, 3], self.thetaj_min, self.thetaj_max))
#   self.save_grb(f"sGRB{self.n_year}S{sample_number}", temp_t90, lc_temp, temp_mean_flux * temp_t90,
#                 temp_mean_flux, z_temp, self.band_low_short, self.band_high_short,
#                 ep_temp, dl_temp, lpeak_temp, eiso_temp, 0)
#
# def add_long(self, sample_number, version):
#   """
#   Creates the quatities of a long burst according to distributions
#   Based on Sarah Antier's thesis
#   """
#   ##################################################################################################################
#   # picking according to distributions
#   ##################################################################################################################
#   z_temp = acc_reject(red_rate_long, [version], self.zmin, self.zmax)
#   lpeak_temp = acc_reject(lpeak_function_long, [version], self.lmin, self.lmax)  # / 3.5
#   temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
#   temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
#   while (temp_band_low - temp_band_high) / (temp_band_low + 2) < 0:
#     temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
#     temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
#   ##################################################################################################################
#   # Calculation other parameters with relations
#   ##################################################################################################################
#   dl_temp = self.cosmo.luminosity_distance(z_temp).to_value("Gpc")
#   ep_temp = yonetoku_reverse_long(lpeak_temp, version)
#   epObstemp = ep_temp / (1 + z_temp)
#   ##################################################################################################################
#   # Calculation of spectrum
#   ##################################################################################################################
#   ampl_norm = normalisation_calc(temp_band_low, temp_band_high)
#   while ampl_norm < 0:
#     ##################################################################################################################
#     # picking according to distributions
#     ##################################################################################################################
#     z_temp = acc_reject(red_rate_long, [version], self.zmin, self.zmax)
#     lpeak_temp = acc_reject(lpeak_function_long, [version], self.lmin, self.lmax)  # / 3.5
#     temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
#     temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
#     while (temp_band_low - temp_band_high) / (temp_band_low + 2) < 0:
#       temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
#       temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
#
#     ##################################################################################################################
#     # Calculation other parameters with relations
#     ##################################################################################################################
#     dl_temp = self.cosmo.luminosity_distance(z_temp).to_value("Gpc")
#     ep_temp = yonetoku_reverse_long(lpeak_temp, version)
#     ##################################################################################################################
#     # Calculation of spectrum
#     ##################################################################################################################
#     ampl_norm = normalisation_calc(temp_band_low, temp_band_high)
#
#   ener_range = np.logspace(1, 3, 100)
#   norm_val, spec = norn_band_spec_calc(temp_band_low, temp_band_high, z_temp, dl_temp, ep_temp, lpeak_temp, ener_range)
#   eiso_temp = amati_long(ep_temp, version)
#   # With a distribution : the value is taken in a log distribution and then put back in linear value
#   temp_t90 = 10**acc_reject(t90_long_log_distri, [version], np.log10(2), 3)
#   lc_temp = self.closest_lc(temp_t90)
#
#   # With Eiso
#   # temp_t90 = (1+z_temp) * eiso_temp / lpeak_temp
#   temp_peak_flux = trapezoid(spec, ener_range)
#
#   times, counts = extract_lc(f"./sources/Light_Curves/{lc_temp}")
#   temp_mean_flux = temp_peak_flux / np.max(counts) * np.mean(counts)
#
#   data_row = pd.DataFrame(data=[[z_temp, epObstemp, ep_temp, lpeak_temp, temp_mean_flux, temp_peak_flux, temp_t90, temp_mean_flux * temp_t90, lc_temp, temp_band_low, temp_band_high, dl_temp, eiso_temp, "Sample long",
#                                  "Sample"]], columns=self.columns)
#   self.sample_df = pd.concat([self.sample_df, data_row], ignore_index=True)
#
#   self.save_grb(f"lGRB{self.n_year}S{sample_number}", temp_t90, lc_temp, temp_mean_flux * temp_t90, temp_mean_flux, z_temp, temp_band_low, temp_band_high, ep_temp, dl_temp, lpeak_temp, eiso_temp, 0)


# df_gbm_short = smp.gbm_df.loc[smp.gbm_df.T90 < 2]
# df_gbm_long = smp.gbm_df.loc[smp.gbm_df.T90 >= 2]
#
# # short_bools = np.where(smp.gbm_df.Type == 'GBM short flnc_band', True, np.where(smp.gbm_df.Type == 'GBM short flnc_comp', True, False))
# # long_bools = np.where(smp.gbm_df.Type == 'GBM long flnc_band', True, np.where(smp.gbm_df.Type == 'GBM long flnc_comp', True, False))
# # df_gbm_short = smp.gbm_df.loc[short_bools]
# # df_gbm_long = smp.gbm_df.loc[long_bools]
#
# # df_gbm_short = df_gbm_short.loc[df_gbm_short.EpeakObs < 10000]
# # df_gbm_long = df_gbm_long.loc[df_gbm_long.EpeakObs < 10000]
#
# ep_short = np.log10(df_gbm_short.EpeakObs.values)
# ep_long = np.log10(df_gbm_long.EpeakObs.values)
# # ep_short = df_gbm_short.EpeakObs.values
# # ep_long = df_gbm_long.EpeakObs.values
# def exp_cutoff(x, A, alpha, x0, xcut):
#   return A * (x/x0)**(-alpha) * np.exp(-x/xcut)
# def sbplaw(x, A, xb, alpha1, alpha2, delta):
#   return A * (x/xb)**(-alpha1) * (1/2*(1+(x/xb)**(1/delta)))**((alpha1-alpha2)*delta)
# def skew(x, ampl, skewness, mu, sigma):
#   return ampl * skewnorm.pdf(x, skewness, mu, sigma)
# bin1, bin2 = 60, 100
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
# hist_s = ax1.hist(ep_short, bins=bin1)
# ax1.set(xscale="linear", yscale="linear", xlabel="Log Epeak short", ylabel="Log number", ylim=(1, 60))
# hist_l = ax2.hist(ep_long, bins=bin2)
# ax2.set(xscale="linear", yscale="linear", xlabel="Log Epeak long", ylabel="Log number", ylim=(1, 300))
# xs, ys, xl, yl = (hist_s[1][1:] + hist_s[1][:-1])/2, hist_s[0], (hist_l[1][1:] + hist_l[1][:-1])/2, hist_l[0]
# # p1 = [8.2,  3.33, -5.89, 35.,  0.12]# p2 = [200,  2.20, -5.17, 7.95, 0.048]# p1 = [11.9, 3.16, -4.87, 35., 0.05]# p2 = [82, 2.17, -7.49, 8.58, 0.049]
# p1 = [16, -5.2, 3.15, 0.66]
# p2 = [80,  1.6,  1.9,  0.44]
# # p2 = [130,  2.18, -5.05, 7.53, 0.04]
# ax1.plot(np.linspace(0, 4, 100), skew(np.linspace(0, 4, 100), *p1), color="red")
# ax2.plot(np.linspace(0, 6, 100), skew(np.linspace(0, 6, 100), *p2), color="red")
# bnd = ([0.01, 1, -10, 0, 0], [np.inf, 4, 0, 35, 10])
# # bnd = ([1, 0, 0, -10, 0, 99], [3, 100, 35, 0, 10, 100])# bnd2 = ([1, 0, 0, -10, 0, 99], [10, 100, 35, 0, 10, 100])
# f1 = curve_fit(skew, xs, ys)[0]
# f2 = curve_fit(skew, xl, yl)[0]
# print(f1, f2)
# ax1.plot(np.linspace(0, 4, 100), skew(np.linspace(0, 4, 100), *f1))
# ax2.plot(np.linspace(0, 6, 100), skew(np.linspace(0, 6, 100), *f2))
# kde1 = gaussian_kde(np.log10(df_gbm_short.EpeakObs.values))
# kde2 = gaussian_kde(np.log10(df_gbm_long.EpeakObs.values))
# ax1.plot(np.linspace(0, 4, 100), np.max(ys)*kde1.pdf(np.linspace(0, 4, 100)))
# ax2.plot(np.linspace(0, 6, 100), np.max(yl)*kde2.pdf(np.linspace(0, 6, 100)))
# # c1 = sns.histplot(np.log10(df_gbm_short.EpeakObs.values), ax=ax1, bins=bin1, kde=True)
# # c2 = sns.histplot(np.log10(df_gbm_long.EpeakObs.values), ax=ax2, bins=bin2, kde=True)
# # epmin, epmax = np.log10(1e0), np.log10(2e4)
# ep_short_dist = kde1.resample(len(ep_short))[0]
# ep_long_dist = kde2.resample(len(ep_long))[0]
# hist_s = ax1.hist(ep_short_dist, histtype="step", bins=bin1)
# hist_l = ax2.hist(ep_long_dist, histtype="step", bins=bin2)
# plt.show()


# alpha = -0.6892844452574826
# beta = -2.3989831213742
# ep = 752.2489411396507
# red = 6.731476096232795
#
# x = np.logspace(-4, 8, 10000000)
# A = normalisation_calc(alpha, beta)
# y = x * band_norm(x, A, alpha, beta)
# print(trapezoid(y, x=x))
#
# x = np.logspace(-4, 8, 1000)
# A = normalisation_calc(alpha, beta)
# y = x * band_norm(x, A, alpha, beta)
# print(trapezoid(y, x=x))
#
# x = np.logspace(1, 3, 10000000)
# A = normalisation_calc(alpha, beta)
# y = x * band_norm(x, A, alpha, beta)
# print(trapezoid(y, x=x))
#
# x = np.logspace(1, 3, 100)
# A = normalisation_calc(alpha, beta)
# y = x * band_norm(x, A, alpha, beta)
# print(trapezoid(y, x=x))

# xb = (alpha - beta) / (alpha + 2)
# def temp_func(x):
#   return x ** (alpha + 1) * np.exp(-(alpha + 2) * x)
#
#
# IntEner = np.logspace(-10, np.log10(xb), 100000000)
# IntFlu = temp_func(IntEner)
# best = trapezoid(IntFlu, x=IntEner)
# print("best : ", best)
#
# for beg in [-9, -8, -7, -6, -5]:
#   for num in [1000, 10000, 100000, 1000000, 10000000]:
#     IntEner = np.logspace(beg, np.log10(xb), num)
#     IntFlu = temp_func(IntEner)
#     print(beg, num, (best - trapezoid(IntFlu, x=IntEner)) / best)
