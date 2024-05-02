import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, trapezoid, simpson, IntegrationWarning
from scipy.stats import skewnorm
from catalog import Catalog
from funcmod import *
from scipy.optimize import curve_fit
import warnings

from astropy.cosmology import WMAP9, Planck18


class GRBSample:
  """
  Class to create GRB samples
  """

  def __init__(self, version):
    """
    Initialisation of the different attributes
    """
    #################################################################################################################
    # General attributres
    #################################################################################################################
    self.zmin = 0
    self.zmax = 10
    self.epmin = 1e-1
    self.epmax = 1e5
    self.thetaj_min = 0
    self.thetaj_max = 15
    self.lmin = 1e47  # erg/s
    self.lmax = 1e53
    self.n_year = 2
    gbmduty = 0.587
    self.gbm_weight = 1 / gbmduty / 10
    self.sample_weight = 1 / self.n_year
    self.version = version
    self.filename = f"./Sampled/sampled_grb_cat_{self.n_year}years.txt"
    self.columns = ["Redshift", "Epeak", "PeakLuminosity", "MeanFlux", "T90", "Fluence", "LightCurveName", "BandLow", "BandHigh", "LuminosityDistance", "EnergyIso", "Type", "Cat"]
    self.sample_df = pd.DataFrame(columns=self.columns)
    self.gbm_df = pd.DataFrame(columns=self.columns)
    #################################################################################################################
    # Short GRB attributes
    #################################################################################################################
    # PARAMETERS
    self.al1_s, self.al2_s, self.lb_s = 0.53, 3.4, 2.8
    self.band_low_short = -0.6
    self.band_high_short = -2.5
    self.short_rate = 0.20  # +0.04 -0.07 [Gpc-3.yr-1]
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
    self.sttype = [4, '\n', 5, '|', 4000]
    self.gbm_cat = None

    #################################################################################################################
    # Setting some attributes
    #################################################################################################################
    self.gbm_cat = Catalog(self.cat_name, self.sttype)
    for ite_gbm, gbm_ep in enumerate(self.gbm_cat.flnc_band_epeak):
      if gbm_ep.strip() != "" and self.gbm_cat.pflx_best_fitting_model[ite_gbm].strip() != "":
        ep_temp = float(gbm_ep)
        temp_t90 = float(self.gbm_cat.t90[ite_gbm])
        lc_temp = self.closest_lc(temp_t90)
        peak_model = getattr(self.gbm_cat, "pflx_best_fitting_model")[ite_gbm].strip()
        temp_mean_flux = float(getattr(self.gbm_cat, f"{peak_model}_phtflux")[ite_gbm]) #calc_flux_gbm(self.gbm_cat, ite_gbm, (10, 1000))
        temp_fluence = temp_mean_flux * temp_t90
        temp_band_low = float(self.gbm_cat.flnc_band_alpha[ite_gbm])
        temp_band_high = float(self.gbm_cat.flnc_band_beta[ite_gbm])
        if temp_t90 < 2:
          temp_type = f"GBM short {self.gbm_cat.flnc_best_fitting_model[ite_gbm]}"
        else:
          temp_type = f"GBM long {self.gbm_cat.flnc_best_fitting_model[ite_gbm]}"

        data_row = pd.DataFrame(data=[[None, ep_temp, None, temp_mean_flux, temp_t90, temp_fluence, lc_temp, temp_band_low, temp_band_high, None, None, temp_type, "GBM"]], columns=self.columns)
        self.gbm_df = pd.concat([self.gbm_df, data_row], ignore_index=True)
      else:
        print("Information : Find null Epeak in catalog")

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
    # Long GRBs
    self.nlong = int(self.n_year * int(quad(red_rate_long, self.zmin, self.zmax, args=version)[0]))
    print("nlong : ", self.nlong)
    for ite in range(self.nlong):
      self.add_long(ite, version)
    # Short GRBs
    self.nshort = int(self.n_year * int(quad(red_rate_short, self.zmin, self.zmax, (self.short_rate, version))[0]))
    print("nshort : ", self.nshort)
    for ite in range(self.nshort):
      self.add_short(ite, version)

  def add_short(self, sample_number, version):
    """
    Creates the quatities of a short burst according to distributions
    Based on Lana Salmon's thesis and Ghirlanda et al, 2016
    """
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    z_temp = acc_reject(red_rate_short, [self.short_rate, version], self.zmin, self.zmax)
    ep_temp = acc_reject(epeak_distribution_short, [version], self.epmin, self.epmax)
    temp_t90 = acc_reject(t90_short_distri, [version], 1e-3, 2)
    lc_temp = self.closest_lc(temp_t90)
    ##################################################################################################################
    # Calculation other parameters with relations
    ##################################################################################################################
    lpeak_temp = yonetoku_short(ep_temp, version)  # / 2
    dl_temp = Planck18.luminosity_distance(z_temp).value / 1000  # Gpc
    eiso_temp = amati_short(ep_temp, version)
    temp_band_high = -0.6
    temp_band_low = -2.5

    ##################################################################################################################
    # Calculation of spectrum
    ##################################################################################################################
    ener_range = np.logspace(1, 3, 100001)
    norm_val, spec = nomr_band_spec_calc(self.band_low_short, self.band_high_short, z_temp, dl_temp, ep_temp, lpeak_temp, ener_range)
    temp_mean_flux = trapezoid(spec, ener_range)

    data_row = pd.DataFrame(data=[[z_temp, ep_temp, lpeak_temp, temp_mean_flux, temp_t90, temp_mean_flux * temp_t90, lc_temp, temp_band_low, temp_band_high, dl_temp, eiso_temp, "Sample short", "Sample"]],
                            columns=self.columns)
    self.sample_df = pd.concat([self.sample_df, data_row], ignore_index=True)

    # self.thetaj_short.append(acc_reject(skewnorm.pdf, [2, 2.5, 3], self.thetaj_min, self.thetaj_max))
    self.save_grb(f"sGRB{self.n_year}S{sample_number}", temp_t90, lc_temp, temp_mean_flux * temp_t90,
                  temp_mean_flux, z_temp, self.band_low_short, self.band_high_short,
                  ep_temp, dl_temp, lpeak_temp, eiso_temp, 0)

  def add_long(self, sample_number, version):
    """
    Creates the quatities of a long burst according to distributions
    Based on Sarah Antier's thesis
    """
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    z_temp = acc_reject(red_rate_long, [version], self.zmin, self.zmax)
    lpeak_temp = acc_reject(lpeak_function_long, [version], self.lmin, self.lmax)  # / 3.5
    temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
    temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
    while (temp_band_low - temp_band_high) / (temp_band_low + 2) < 0:
      temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
      temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
    ##################################################################################################################
    # Calculation other parameters with relations
    ##################################################################################################################
    dl_temp = Planck18.luminosity_distance(z_temp).to_value("Gpc")
    ep_temp = yonetoku_long(lpeak_temp, version)
    ##################################################################################################################
    # Calculation of spectrum
    ##################################################################################################################
    ampl_norm = normalisation_calc(temp_band_low, temp_band_high)
    while ampl_norm < 0:
      ##################################################################################################################
      # picking according to distributions
      ##################################################################################################################
      z_temp = acc_reject(red_rate_long, [version], self.zmin, self.zmax)
      lpeak_temp = acc_reject(lpeak_function_long, [version], self.lmin, self.lmax)  # / 3.5
      temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
      temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
      while (temp_band_low - temp_band_high) / (temp_band_low + 2) < 0:
        temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
        temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)

      ##################################################################################################################
      # Calculation other parameters with relations
      ##################################################################################################################
      dl_temp = Planck18.luminosity_distance(z_temp).to_value("Gpc")
      ep_temp = yonetoku_long(lpeak_temp, version)
      ##################################################################################################################
      # Calculation of spectrum
      ##################################################################################################################
      ampl_norm = normalisation_calc(temp_band_low, temp_band_high)

    ener_range = np.logspace(1, 3, 100)
    norm_val, spec = nomr_band_spec_calc(temp_band_low, temp_band_high, z_temp, dl_temp, ep_temp, lpeak_temp, ener_range)
    eiso_temp = amati_long(ep_temp, version)
    # With a distribution : the value is taken in a log distribution and then put back in linear value
    temp_t90 = 10**acc_reject(t90_long_log_distri, [version], np.log10(2), 3)
    lc_temp = self.closest_lc(temp_t90)

    # With Eiso
    # temp_t90 = (1+z_temp) * eiso_temp / lpeak_temp
    temp_mean_flux = trapezoid(spec, ener_range)

    data_row = pd.DataFrame(data=[[z_temp, ep_temp, lpeak_temp, temp_mean_flux, temp_t90, temp_mean_flux * temp_t90, lc_temp, temp_band_low, temp_band_high, dl_temp, eiso_temp, "Sample long", "Sample"]],
                            columns=self.columns)
    self.sample_df = pd.concat([self.sample_df, data_row], ignore_index=True)

    self.save_grb(f"lGRB{self.n_year}S{sample_number}", temp_t90, lc_temp, temp_mean_flux * temp_t90, temp_mean_flux, z_temp, temp_band_low, temp_band_high, ep_temp, dl_temp, lpeak_temp, eiso_temp, 0)

  def closest_lc(self, searched_time):
    """
    Find the lightcurve with a duration which is the closest to the sampled t90 time
    """
    abs_diff = np.abs(np.array(self.gbm_cat.t90, dtype=float) - searched_time)
    gbm_index = np.argmin(abs_diff)
    # print(searched_time, float(self.gbm_cat.t90[gbm_index]))
    return f"LightCurve_{self.gbm_cat.name[gbm_index]}.dat"

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

  def short_distri(self):
    """
    Compare the distribution of the created quatities and the seed distributions
    """
    yscale = "log"
    nbin = 50

    fluence_min, fluence_max = 1e-8, 1e4
    flux_min, flux_max = 1e-8, 1e5

    df_short = self.sample_df.loc[self.sample_df.Type == "Sample short"]
    n_sample = len(df_short)

    df_gbm_short = self.gbm_df.loc[self.gbm_df.T90 < 2]
    n_gbm = len(df_gbm_short)

    comp_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(27, 12))

    ax1.hist(df_short.Epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax1.hist(df_gbm_short.Epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax1.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1.legend()

    fluence_bin = np.logspace(np.log10(fluence_min), np.log10(fluence_max), nbin)
    ax2.hist(df_short.Fluence, bins=fluence_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax2.hist(df_gbm_short.Fluence, bins=fluence_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax2.set(title="Fluence distributions", xlabel="Photon fluence (photon/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax2.legend()

    flux_bin = np.logspace(np.log10(flux_min), np.log10(flux_max), nbin)
    ax3.hist(df_short.MeanFlux, bins=flux_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax3.hist(df_gbm_short.MeanFlux, bins=flux_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax3.set(title="Mean flux distributions", xlabel="Photon flux (photon/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax3.legend()

    ax4.hist(df_short.T90, bins=np.logspace(-3, np.log10(2), nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax4.hist(df_gbm_short.T90, bins=np.logspace(-3, np.log10(2), nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax4.set(title="T90 distributions", xlabel="T90 (s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax4.legend()

    plt.show()

  def long_distri(self):
    """
    Compare the distribution of the created quatities and the seed distributions
    """
    yscale = "log"
    nbin = 50

    fluence_min, fluence_max = 1e-8, 1e4
    flux_min, flux_max = 1e-8, 1e5

    df_long = self.sample_df.loc[self.sample_df.Type == "Sample long"]
    n_sample = len(df_long)

    df_gbm_long = self.gbm_df.loc[self.gbm_df.T90 >= 2]
    n_gbm = len(df_gbm_long)

    comp_fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(27, 12))

    ax1.hist(df_long.Epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax1.hist(df_gbm_long.Epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax1.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax1.legend()

    alpha_bin = np.linspace(np.min(df_long.BandLow), np.max(df_long.BandLow), nbin)
    ax2.hist(df_long.BandLow, bins=alpha_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax2.hist(df_gbm_long.BandLow, bins=alpha_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax2.set(title="Band low energy index", xlabel="Alpha", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax2.legend()

    beta_bin = np.linspace(np.min(df_long.BandHigh), np.max(df_long.BandHigh), nbin)
    ax3.hist(df_long.BandHigh, bins=beta_bin, histtype="step", color="green", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax3.hist(df_gbm_long.BandHigh, bins=beta_bin, histtype="step", color="orange", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax3.set(title="Band high energy index", xlabel="Beta", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax3.legend()

    fluence_bin = np.logspace(np.log10(fluence_min), np.log10(fluence_max), nbin)
    ax4.hist(df_long.Fluence, bins=fluence_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax4.hist(df_gbm_long.Fluence, bins=fluence_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax4.set(title="Fluence distributions", xlabel="Photon fluence (photon/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax4.legend()

    flux_bin = np.logspace(np.log10(flux_min), np.log10(flux_max), nbin)
    ax5.hist(df_long.MeanFlux, bins=flux_bin, histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax5.hist(df_gbm_long.MeanFlux, bins=flux_bin, histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax5.set(title="Mean flux distributions", xlabel="Photon flux (photon/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax5.legend()

    ax6.hist(df_long.T90, bins=np.logspace(np.log10(2), 3, nbin), histtype="step", color="blue", label=f"Sample, {n_sample} GRB", weights=[self.sample_weight] * n_sample)
    ax6.hist(df_gbm_long.T90, bins=np.logspace(np.log10(2), 3, nbin), histtype="step", color="red", label=f"GBM, {n_gbm} GRB", weights=[self.gbm_weight] * n_gbm)
    ax6.set(title="T90 distributions", xlabel="T90 (s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax6.legend()

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
    p1 = sns.pairplot(plot_df, hue="Type", corner=True, palette=colors, plot_kws={'s': 10})

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

    sns.pairplot(plot_df, hue="Type", corner=True, palette=colors, plot_kws={'s': 10})
