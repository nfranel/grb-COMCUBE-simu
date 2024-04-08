
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

  def __init__(self):
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
    self.n_year = 0.5
    self.filename = f"./Sampled/sampled_grb_cat_{self.n_year}years.txt"
    #################################################################################################################
    # Short GRB attributes
    #################################################################################################################
    # PARAMETERS
    self.al1_s, self.al2_s, self.lb_s = 0.53, 3.4, 2.8
    self.p1, self.zp, self.p2 = 2.8, 2.3, 3.5
    self.a1, self.a2, self.epb = -0.53, 4, 1600  # Took -a1 because the results were coherent only this way
    self.ma, self.qa = 1.1, 0.042
    self.my, self.qy = 0.84, 0.034
    self.mu, self.sigma, self.skew = 2.5, 3, 2
    self.band_low_short = -0.6
    self.band_high_short = -2.5
    self.short_rate = 0.20  # +0.04 -0.07 [Gpc-3.yr-1]
    self.nshort = None  # func to calculate ?
    self.z_short = []
    self.dl_short = []
    self.ep_short = []
    self.ep_o_short = []
    self.eiso_short = []
    self.liso_short = []
    self.thetaj_short = []
    self.mean_flux_short = []
    self.t90_short = []
    self.fluence_short = []

    self.lc_short = []

    #################################################################################################################
    # long GRB attributes
    #################################################################################################################
    self.red0, self.n1, self.n2, self.z1 = 0.42, 2.07, -0.7, 3.6
    self.al1_l, self.al2_l, self.lb_l = -0.65, -3, 10 ** 52.05
    self.ampl, self.l0, self.ind1 = 372, 1e52, 0.5
    self.band_low_l_mu, self.band_low_l_sig = -0.87, 0.33
    self.band_high_l_mu, self.band_high_l_sig = -2.36, 0.31
    self.t90_mu, self.t90_sig = 58.27, 18

    self.nlong = None
    self.z_long = []
    self.dl_long = []
    self.lbol_long = []
    self.eiso_long = []
    self.ep_long = []
    self.band_low_long = []
    self.band_high_long = []
    self.mean_flux_long = []
    self.t90_long = []
    self.fluence_long = []

    self.lc_long = []

    #################################################################################################################
    # GBM attributes
    #################################################################################################################
    self.cat_name = "GBM/allGBM.txt"
    self.sttype = [4, '\n', 5, '|', 4000]
    self.gbm_cat = None
    self.all_gbm_epeak = []
    self.short_gbm_epeak = []
    self.long_gbm_epeak = []
    self.short_gbm_ph_flux = []
    self.long_gbm_ph_flux = []
    self.short_gbm_en_flux = []
    self.long_gbm_en_flux = []
    self.short_gbm_alpha = []
    self.long_gbm_alpha = []
    self.short_gbm_beta = []
    self.long_gbm_beta = []
    self.gbm_t90 = []
    self.short_gbm_ph_fluence = []
    self.long_gbm_ph_fluence = []
    self.short_gbm_t90 = []
    self.long_gbm_t90 = []


    #################################################################################################################
    # Setting some attributes
    #################################################################################################################
    self.gbm_cat = Catalog(self.cat_name, self.sttype)
    for ite_gbm, gbm_ep in enumerate(self.gbm_cat.flnc_band_epeak):
      if gbm_ep.strip() != "":
        self.all_gbm_epeak.append(float(gbm_ep))
        self.gbm_t90.append(float(self.gbm_cat.t90[ite_gbm]))
        if float(self.gbm_cat.t90[ite_gbm]) <= 2:
          self.short_gbm_epeak.append(float(gbm_ep))
          self.short_gbm_ph_flux.append(calc_flux_gbm(self.gbm_cat, ite_gbm, (10, 1000)))
          self.short_gbm_ph_fluence.append(self.short_gbm_ph_flux[-1]*float(self.gbm_cat.t90[ite_gbm]))
          self.short_gbm_en_flux.append(float(self.gbm_cat.fluence[ite_gbm]) / float(self.gbm_cat.t90[ite_gbm]))
          self.short_gbm_alpha.append(float(self.gbm_cat.flnc_band_alpha[ite_gbm]))
          self.short_gbm_beta.append(float(self.gbm_cat.flnc_band_beta[ite_gbm]))
          self.short_gbm_t90.append(float(self.gbm_cat.t90[ite_gbm]))
        else:
          self.long_gbm_epeak.append(float(gbm_ep))
          self.long_gbm_ph_flux.append(calc_flux_gbm(self.gbm_cat, ite_gbm, (10, 1000)))
          self.long_gbm_ph_fluence.append(self.long_gbm_ph_flux[-1]*float(self.gbm_cat.t90[ite_gbm]))
          self.long_gbm_en_flux.append(float(self.gbm_cat.fluence[ite_gbm]) / float(self.gbm_cat.t90[ite_gbm]))
          self.long_gbm_alpha.append(float(self.gbm_cat.flnc_band_alpha[ite_gbm]))
          self.long_gbm_beta.append(float(self.gbm_cat.flnc_band_beta[ite_gbm]))
          self.long_gbm_t90.append(float(self.gbm_cat.t90[ite_gbm]))
      else:
        print("Information : Find null Epeak in catalog")

    #################################################################################################################
    # Creating GRBs
    #################################################################################################################
    # Setting up a save file
    with open(self.filename, "w") as f:
      f.write("Header : \n")
      f.write(f"Catalog of synthetic GRBs sampled over {self.n_year} years\n")
      f.write(f"Based on differents works, see MCMCGRB.py for more details\n")
      f.write("Keys : \n")
      f.write("name|t90|light curve name|fluence|mean flux|redshift|Band low energy index|Band high energy index|peak energy|luminosity distance|isotropic luminosity|isotropic energy|jet opening angle\n")
    # Long GRBs
    self.nlong = int(self.n_year * int(quad(red_rate_long, self.zmin, self.zmax, (self.red0, self.n1, self.n2, self.z1))[0]))
    print("nlong : ", self.nlong)
    for ite in range(self.nlong):
      self.add_long(ite)
    # Short GRBs
    self.nshort = int(self.n_year * int(quad(red_rate_short, self.zmin, self.zmax, (self.short_rate, self.p1, self.zp, self.p2))[0]))
    print("nshort : ", self.nshort)
    for ite in range(self.nshort):
      self.add_short(ite)

  def add_short(self, sample_number):
    """
    Creates the quatities of a short burst according to distributions
    Based on Lana Salmon's thesis and Ghirlanda et al, 2016
    """
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    # self.z_short.append(acc_reject(redshift_distribution_short, [self.p1, self.zp, self.p2], self.zmin, self.zmax))
    self.z_short.append(acc_reject(red_rate_short, [self.short_rate, self.p1, self.zp, self.p2], self.zmin, self.zmax))
    self.ep_short.append(acc_reject(epeak_distribution_short, [self.a1, self.a2, self.epb], self.epmin, self.epmax))
    ##################################################################################################################
    # Calculation other parameters with relations
    ##################################################################################################################
    self.dl_short.append(Planck18.luminosity_distance(self.z_short[-1]).value / 1000)  # Gpc
    self.ep_o_short.append(self.ep_short[-1] / (1 + self.z_short[-1]))
    self.eiso_short.append(amati_short(self.ep_short[-1], self.qa, self.ma))
    self.liso_short.append(yonetoku_short(self.ep_short[-1], self.qy, self.my) / 2)
    self.band_low_short = -0.6
    self.band_high_short = -2.5

    ##################################################################################################################
    # Calculation of spectrum
    ##################################################################################################################
    ampl_norm = normalisation_calc(self.band_low_short, self.band_high_short)
    ener_range = np.logspace(1, 3, 100001)
    norm = (1 + self.z_short[-1])**2 / (4*np.pi*(self.dl_short[-1] * Gpc_to_cm)**2) * self.liso_short[-1] / (self.ep_short[-1]**2 * keV_to_erg)
    spec_norm = band_norm((1+self.z_short[-1])*ener_range/self.ep_short[-1], ampl_norm, self.band_low_short, self.band_high_short)
    spec = norm * spec_norm
    # print("======")
    # # print(trapezoid(spec, ener_range))
    # print(simpson(spec_norm * (1+self.z_short[-1])*ener_range/self.ep_short[-1], (1+self.z_short[-1])*ener_range/self.ep_short[-1]))
    # print(simpson(spec, (1+self.z_short[-1])*ener_range/self.ep_short[-1]))
    self.mean_flux_short.append(trapezoid(spec, ener_range))

    # self.t90_short.append((1+self.z_short[-1]) * self.eiso_short[-1] / self.liso_short[-1])
    self.t90_short.append(acc_reject(t90_short_distri, [], 1e-3, 2))

    self.lc_short.append(self.closest_lc(self.t90_short[-1]))
    self.fluence_short.append(self.mean_flux_long[-1] * self.t90_short[-1])

    # todo, check if rejection method is really needed here or if a skewed normal law can be directly use
    self.thetaj_short.append(acc_reject(skewnorm.pdf, [self.skew, self.mu, self.sigma], self.thetaj_min, self.thetaj_max))
    self.save_grb(f"sGRB{self.n_year}S{sample_number}", self.t90_short[-1], self.lc_short[-1], self.fluence_short[-1],
                  self.mean_flux_short[-1], self.z_short[-1], self.band_low_short, self.band_high_short,
                  self.ep_short[-1], self.dl_short[-1], self.liso_short[-1], self.eiso_short[-1], self.thetaj_short[-1])

  def add_long(self, sample_number):
    """
    Creates the quatities of a long burst according to distributions
    Based on Sarah Antier's thesis
    """
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    z_temp = acc_reject(red_rate_long, [self.red0, self.n1, self.n2, self.z1], self.zmin, self.zmax)
    lbol_temp = acc_reject(luminosity_function, [self.al1_l, self.al2_l, self.lb_l], self.lmin, self.lmax) / 3.5
    temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
    temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
    while (temp_band_low - temp_band_high) / (temp_band_low + 2) < 0:
      temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
      temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
    ##################################################################################################################
    # Calculation other parameters with relations
    ##################################################################################################################
    dl_temp = Planck18.luminosity_distance(z_temp).to_value("Gpc")
    ep_temp = yonetoku_long(lbol_temp, self.ampl, self.l0, self.ind1)
    ##################################################################################################################
    # Calculation of spectrum
    ##################################################################################################################
    ampl_norm = normalisation_calc(temp_band_low, temp_band_high)
    while ampl_norm < 0:
      ##################################################################################################################
      # picking according to distributions
      ##################################################################################################################
      z_temp = acc_reject(red_rate_long, [self.red0, self.n1, self.n2, self.z1], self.zmin, self.zmax)
      lbol_temp = acc_reject(luminosity_function, [self.al1_l, self.al2_l, self.lb_l], self.lmin, self.lmax) / 3.5
      temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
      temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)
      while (temp_band_low - temp_band_high) / (temp_band_low + 2) < 0:
        temp_band_low = np.random.normal(loc=self.band_low_l_mu, scale=self.band_low_l_sig)
        temp_band_high = np.random.normal(loc=self.band_high_l_mu, scale=self.band_high_l_sig)

      ##################################################################################################################
      # Calculation other parameters with relations
      ##################################################################################################################
      dl_temp = Planck18.luminosity_distance(z_temp).to_value("Gpc")
      ep_temp = yonetoku_long(lbol_temp, self.ampl, self.l0, self.ind1)
      ##################################################################################################################
      # Calculation of spectrum
      ##################################################################################################################
      ampl_norm = normalisation_calc(temp_band_low, temp_band_high)

    ener_range = np.logspace(1, 3, 100)
    norm = (1 + z_temp)**2 / (4*np.pi*(dl_temp * Gpc_to_cm)**2) * lbol_temp / (ep_temp**2 * keV_to_erg)
    spec_norm = band_norm((1+z_temp)*ener_range/ep_temp, ampl_norm, temp_band_low, temp_band_high)
    spec = norm * spec_norm

    eiso_temp = amati_long(ep_temp)
    # With a distribution
    temp_t90 = acc_reject(t90_long_distri, [], 2, 1000)
    # With Eiso
    # temp_t90 = (1+z_temp) * eiso_temp / lbol_temp

    self.z_long.append(z_temp)
    self.dl_long.append(dl_temp)
    self.lbol_long.append(lbol_temp)
    self.ep_long.append(ep_temp)
    self.band_low_long.append(temp_band_low)
    self.band_high_long.append(temp_band_high)
    # spec is N(E) so now we need to integrate it vs E (and not vs (1+z)E/Ep)
    self.mean_flux_long.append(trapezoid(spec, ener_range))
    self.t90_long.append(temp_t90)
    self.lc_long.append(self.closest_lc(temp_t90))
    self.eiso_long.append(eiso_temp)
    self.fluence_long.append(self.mean_flux_long[-1] * temp_t90)
    self.save_grb(f"lGRB{self.n_year}S{sample_number}", self.t90_long[-1], self.lc_long[-1], self.fluence_long[-1],
                  self.mean_flux_long[-1], self.z_long[-1], self.band_low_long[-1], self.band_high_long[-1],
                  self.ep_long[-1], self.dl_long[-1], self.lbol_long[-1], eiso_temp, 0)


  def closest_lc(self, searched_time):
    """
    Find the lightcurve with a duration which is the closest to the sampled t90 time
    """
    abs_diff = np.abs(np.array(self.gbm_t90) - searched_time)
    gbm_index = np.argmin(abs_diff)
    # print(searched_time, self.gbm_t90[gbm_index])
    return f"LightCurve_{self.gbm_cat.name[gbm_index]}.dat"

  def save_grb(self, name, t90, lcname, fluence, mean_flux, red, band_low, band_high, ep, dl, liso, eiso, thetaj):
    """
    Saves a GRB in a catalog file
    """
    with open(self.filename, "a") as f:
      f.write(f"{name}|{t90}|{lcname}|{fluence}|{mean_flux}|{red}|{band_low}|{band_high}|{ep}|{dl}|{liso}|{eiso}|{thetaj}\n")

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

    n_sample = len(self.z_short)
    comp_fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(27, 12))
    ax1.hist(self.z_short, bins=45, histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax1.step(zx_lana, zy_lana, color="red", label="Model")
    ax1.set(title="Redshift distributions", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale="linear")
    ax1.legend()

    ax2.hist(self.dl_short, bins=45, histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax2.step(dlx_lana, dly_lana, color="red", label="Model")
    ax2.set(title="DL distributions", xlabel="Luminosity distance (Gpc)", ylabel="Number of GRB", xscale="linear", yscale="linear")
    ax2.legend()

    ax3.hist(self.ep_short, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), 56), histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax3.step(epx_lana, epy_lana, color="red", label="Model")
    ax3.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale="linear")
    ax3.legend()

    ax4.hist(self.ep_o_short, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), 56), histtype="step", color="blue", weights=[len(self.short_gbm_epeak) / n_sample] * n_sample, label="Distribution")
    ax4.hist(self.short_gbm_epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), 56), histtype="step", color="red", label="Model (GBM)")
    ax4.set(title="Ep obs distributions", xlabel="Obs frame peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale="linear")
    ax4.legend()

    ax5.hist(np.log10(np.array(self.eiso_short) / 1e52), bins=28, histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax5.step(eisox_lana, eisoy_lana, color="red", label="Model")
    ax5.set(title="Eiso distributions", xlabel="Log10(Eiso/1e52) (erg)", ylabel="Number of GRB", xscale="linear", yscale="linear")
    ax5.legend()

    ax6.hist(self.thetaj_short, bins=np.linspace(0, 15, 30), histtype="step", color="blue", weights=[1000 / n_sample] * n_sample, label="Distribution")
    ax6.step(thx_lana, thy_lana, color="red", label="Model")
    ax6.set(title="thetaj distributions", xlabel="Thetaj (°)", ylabel="Number of GRB", xscale="linear", yscale="linear")
    ax6.legend()

  def short_distri(self):
    """
    Compare the distribution of the created quatities and the seed distributions
    """
    yscale = "log"
    nbin = 50
    gbmduty = 0.587
    # gbmcorrec = 1 / gbmduty / 10
    gbmcorrec = 2
    # n_sample = len(self.z_long)
    comp_fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(27, 12))
    ax1.hist(self.z_short, bins=nbin, histtype="step", color="blue", label=f"Sample, {len(self.z_short)} GRB", weights=[1 / self.n_year] * len(self.z_short))
    ax1.set(title="Redshift distributions", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax1.legend()

    ax2.hist(self.liso_short, bins=np.logspace(np.log10(self.lmin), np.log10(self.lmax), nbin), histtype="step", color="blue", label=f"Sample, {len(self.liso_short)} GRB", weights=[1 / self.n_year] * len(self.liso_short))
    ax2.set(title="Lbol distributions", xlabel="Bolometric luminosity (erg/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax2.legend()

    ax3.hist(self.ep_short, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="blue", label=f"Sample, {len(self.ep_short)} GRB", weights=[1 / self.n_year] * len(self.ep_short))
    ax3.hist(self.short_gbm_epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="red", label=f"GBM, {len(self.short_gbm_epeak)} GRB", weights=[gbmcorrec] * len(self.short_gbm_epeak))
    ax3.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax3.legend()

    ph_short_fluence = np.array(self.mean_flux_short)*np.array(self.t90_short)
    ph_min, ph_max = np.min(ph_short_fluence), np.max(ph_short_fluence)
    ax4.hist(ph_short_fluence, bins=np.logspace(np.log10(ph_min), np.log10(ph_max), nbin), histtype="step", color="blue", label=f"Sample, {len(ph_short_fluence)} GRB", weights=[1 / self.n_year] * len(ph_short_fluence))
    ax4.hist(self.short_gbm_ph_fluence, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="red", label=f"GBM, {len(self.short_gbm_ph_fluence)} GRB", weights=[gbmcorrec] * len(self.short_gbm_ph_fluence))
    ax4.set(title="Fluence distributions", xlabel="Photon fluence (photon/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax4.legend()

    ax5.hist(self.mean_flux_short, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="blue", label=f"Sample, {len(self.mean_flux_short)} GRB", weights=[1 / self.n_year] * len(self.mean_flux_short))
    ax5.hist(self.short_gbm_ph_flux, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="red", label=f"GBM, {len(self.short_gbm_ph_flux)} GRB", weights=[gbmcorrec] * len(self.short_gbm_ph_flux))
    ax5.set(title="Mean flux distributions", xlabel="Photon flux (photon/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax5.legend()

    ax6.hist(self.t90_short, bins=nbin, histtype="step", color="blue", label=f"Sample, {len(self.t90_short)} GRB", weights=[1 / self.n_year] * len(self.t90_short))
    ax6.hist(self.short_gbm_t90, bins=nbin, histtype="step", color="red", label=f"GBM, {len(self.short_gbm_t90)} GRB", weights=[gbmcorrec] * len(self.short_gbm_t90))
    ax6.set(title="T90 distributions", xlabel="T90 (s)", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax6.legend()

    # index = np.where(ph_short_fluence > 52.64808276890968, True, False)
    # comp_fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(27, 12))
    # ax1.hist(np.array(self.z_short)[index], bins=nbin, histtype="step", color="blue", label=f"Sample, {len(np.array(self.z_short)[index])} GRB")
    # ax1.set(title="Redshift distributions", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    # ax1.legend()
    #
    # ax2.hist(np.array(self.liso_short)[index], bins=np.logspace(np.log10(self.lmin), np.log10(self.lmax), nbin), histtype="step", color="blue", label=f"Sample, {len(np.array(self.liso_short)[index])} GRB")
    # ax2.set(title="Lbol distributions", xlabel="Bolometric luminosity (erg/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    # ax2.legend()
    #
    # ax3.hist(np.array(self.ep_short)[index], bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="blue", label=f"Sample, {len(np.array(self.ep_short)[index])} GRB")
    # ax3.hist(self.short_gbm_epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="red", label=f"GBM, {len(self.short_gbm_epeak)} GRB", weights=[1 / gbmduty] * len(self.short_gbm_epeak))
    # ax3.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    # ax3.legend()
    #
    # ph_short_fluence = np.array(self.mean_flux_short)*np.array(self.t90_short)
    # ph_min, ph_max = np.min(ph_short_fluence), np.max(ph_short_fluence)
    # ax4.hist(np.array(ph_short_fluence)[index], bins=np.logspace(np.log10(ph_min), np.log10(ph_max), nbin), histtype="step", color="blue", label=f"Sample, {len(np.array(ph_short_fluence)[index])} GRB")
    # ax4.hist(self.short_gbm_ph_fluence, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="red", label=f"GBM, {len(self.short_gbm_ph_fluence)} GRB", weights=[1 / gbmduty] * len(self.short_gbm_ph_fluence))
    # ax4.set(title="Fluence distributions", xlabel="Photon fluence (photon/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    # ax4.legend()
    #
    # ax5.hist(np.array(self.mean_flux_short)[index], bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="blue", label=f"Sample, {len(np.array(self.mean_flux_short)[index])} GRB")
    # ax5.hist(self.short_gbm_ph_flux, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="red", label=f"GBM, {len(self.short_gbm_ph_flux)} GRB", weights=[1 / gbmduty] * len(self.short_gbm_ph_flux))
    # ax5.set(title="Mean flux distributions", xlabel="Photon flux (photon/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    # ax5.legend()
    #
    # ax6.hist(np.array(self.t90_short)[index], bins=nbin, histtype="step", color="blue", label=f"Sample, {len(np.array(self.t90_short)[index])} GRB")
    # ax6.hist(self.short_gbm_t90, bins=nbin, histtype="step", color="red", label=f"GBM, {len(self.short_gbm_t90)} GRB", weights=[1 / gbmduty] * len(self.short_gbm_t90))
    # ax6.set(title="T90 distributions", xlabel="T90 (s)", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    # ax6.legend()


  def long_distri(self):
    """
    Compare the distribution of the created quatities and the seed distributions
    """
    yscale = "log"
    nbin = 50
    gbmduty = 0.587
    # n_sample = len(self.z_long)
    # gbmcorrec = 1 / gbmduty / 10
    gbmcorrec = 2
    comp_fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(27, 12))
    ax1.hist(self.z_long, bins=nbin, histtype="step", color="blue", label=f"Sample, {len(self.z_long)} GRB", weights=[1 / self.n_year] * len(self.z_long))
    ax1.set(title="Redshift distributions", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax1.legend()

    ax2.hist(self.lbol_long, bins=np.logspace(np.log10(self.lmin), np.log10(self.lmax), nbin), histtype="step", color="blue", label=f"Sample, {len(self.lbol_long)} GRB", weights=[1 / self.n_year] * len(self.lbol_long))
    ax2.set(title="Lbol distributions", xlabel="Bolometric luminosity (erg/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax2.legend()

    ax3.hist(self.ep_long, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="blue", label=f"Sample, {len(self.ep_long)} GRB", weights=[1 / self.n_year] * len(self.ep_long))
    ax3.hist(self.long_gbm_epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="red", label=f"GBM, {len(self.long_gbm_epeak)} GRB", weights=[gbmcorrec] * len(self.long_gbm_epeak))
    ax3.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax3.legend()

    ax4.hist(self.band_low_long, bins=np.linspace(np.min(self.long_gbm_alpha), np.max(self.long_gbm_alpha), nbin), histtype="step", color="blue", label=f"Sample, {len(self.band_low_long)} GRB", weights=[1 / self.n_year] * len(self.band_low_long))
    ax4.hist(self.long_gbm_alpha, bins=np.linspace(np.min(self.long_gbm_alpha), np.max(self.long_gbm_alpha), nbin), histtype="step", color="red", label=f"GBM, {len(self.long_gbm_alpha)} GRB", weights=[gbmcorrec] * len(self.long_gbm_alpha))
    ax4.set(title="Band low energy index", xlabel="Alpha", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax4.legend()

    ax5.hist(self.band_high_long, bins=np.linspace(np.min(self.long_gbm_beta), np.max(self.long_gbm_beta), nbin), histtype="step", color="green", label=f"Sample, {len(self.band_high_long)} GRB", weights=[1 / self.n_year] * len(self.band_high_long))
    ax5.hist(self.long_gbm_beta, bins=np.linspace(np.min(self.long_gbm_beta), np.max(self.long_gbm_beta), nbin), histtype="step", color="orange", label=f"GBM, {len(self.long_gbm_beta)} GRB", weights=[gbmcorrec] * len(self.long_gbm_beta))
    ax5.set(title="Band high energy index", xlabel="Beta", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax5.legend()

    ph_long_fluence = np.array(self.mean_flux_long)*np.array(self.t90_long)
    ph_min, ph_max = np.min(ph_long_fluence), np.max(ph_long_fluence)
    ax6.hist(ph_long_fluence, bins=np.logspace(np.log10(ph_min), np.log10(ph_max), nbin), histtype="step", color="blue", label=f"Sample, {len(ph_long_fluence)} GRB", weights=[1 / self.n_year] * len(ph_long_fluence))
    ax6.hist(self.long_gbm_ph_fluence, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="red", label=f"GBM, {len(self.long_gbm_ph_fluence)} GRB", weights=[gbmcorrec] * len(self.long_gbm_ph_fluence))
    ax6.set(title="Fluence distributions", xlabel="Photon fluence (photon/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax6.legend()

    ax7.hist(self.mean_flux_long, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="blue", label=f"Sample, {len(self.mean_flux_long)} GRB", weights=[1 / self.n_year] * len(self.mean_flux_long))
    ax7.hist(self.long_gbm_ph_flux, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="red", label=f"GBM, {len(self.long_gbm_ph_flux)} GRB", weights=[gbmcorrec] * len(self.long_gbm_ph_flux))
    ax7.set(title="Mean flux distributions", xlabel="Photon flux (photon/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    ax7.legend()

    ax8.hist(self.t90_long, bins=nbin, histtype="step", color="blue", label=f"Sample, {len(self.t90_long)} GRB", weights=[1 / self.n_year] * len(self.t90_long))
    ax8.hist(self.long_gbm_t90, bins=nbin, histtype="step", color="red", label=f"GBM, {len(self.long_gbm_t90)} GRB", weights=[gbmcorrec] * len(self.long_gbm_t90))
    ax8.set(title="T90 distributions", xlabel="T90 (s)", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    ax8.legend()

    # index = np.where(ph_long_fluence > 3498.2773030363187, True, False)
    # comp_fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(27, 12))
    # ax1.hist(np.array(self.z_long)[index], bins=nbin, histtype="step", color="blue", label=f"Sample, {len(np.array(self.z_long)[index])} GRB")
    # ax1.set(title="Redshift distributions", xlabel="Redshift", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    # ax1.legend()
    #
    # ax2.hist(np.array(self.lbol_long)[index], bins=np.logspace(np.log10(self.lmin), np.log10(self.lmax), nbin), histtype="step", color="blue", label=f"Sample, {len(np.array(self.lbol_long)[index])} GRB")
    # ax2.set(title="Lbol distributions", xlabel="Bolometric luminosity (erg/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    # ax2.legend()
    #
    # ax3.hist(np.array(self.ep_long)[index], bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="blue", label=f"Sample, {len(np.array(self.ep_long)[index])} GRB")
    # ax3.hist(self.long_gbm_epeak, bins=np.logspace(np.log10(self.epmin), np.log10(self.epmax), nbin), histtype="step", color="red", label=f"GBM, {len(self.long_gbm_epeak)} GRB", weights=[1 / gbmduty] * len(self.long_gbm_epeak))
    # ax3.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    # ax3.legend()
    #
    # ax4.hist(np.array(self.band_low_long)[index], bins=np.linspace(np.min(self.long_gbm_alpha), np.max(self.long_gbm_alpha), nbin), histtype="step", color="blue", label=f"Sample, {len(np.array(self.band_low_long)[index])} GRB")
    # ax4.hist(self.long_gbm_alpha, bins=np.linspace(np.min(self.long_gbm_alpha), np.max(self.long_gbm_alpha), nbin), histtype="step", color="red", label=f"GBM, {len(self.long_gbm_alpha)} GRB", weights=[1 / gbmduty] * len(self.long_gbm_alpha))
    # ax4.set(title="Band low energy index", xlabel="Alpha", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    # ax4.legend()
    #
    # ax5.hist(np.array(self.band_high_long)[index], bins=np.linspace(np.min(self.long_gbm_beta), np.max(self.long_gbm_beta), nbin), histtype="step", color="green", label=f"Sample, {len(np.array(self.band_high_long)[index])} GRB")
    # ax5.hist(self.long_gbm_beta, bins=np.linspace(np.min(self.long_gbm_beta), np.max(self.long_gbm_beta), nbin), histtype="step", color="orange", label=f"GBM, {len(self.long_gbm_beta)} GRB", weights=[1 / gbmduty] * len(self.long_gbm_beta))
    # ax5.set(title="Band high energy index", xlabel="Beta", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    # ax5.legend()
    #
    # ph_long_fluence = np.array(self.mean_flux_long)*np.array(self.t90_long)
    # ph_min, ph_max = np.min(ph_long_fluence), np.max(ph_long_fluence)
    # ax6.hist(np.array(ph_long_fluence)[index], bins=np.logspace(np.log10(ph_min), np.log10(ph_max), nbin), histtype="step", color="blue", label=f"Sample, {len(np.array(ph_long_fluence)[index])} GRB")
    # ax6.hist(self.long_gbm_ph_fluence, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="red", label=f"GBM, {len(self.long_gbm_ph_fluence)} GRB", weights=[1 / gbmduty] * len(self.long_gbm_ph_fluence))
    # ax6.set(title="Fluence distributions", xlabel="Photon fluence (photon/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    # ax6.legend()
    #
    # ax7.hist(np.array(self.mean_flux_long)[index], bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="blue", label=f"Sample, {len(np.array(self.mean_flux_long)[index])} GRB")
    # ax7.hist(self.long_gbm_ph_flux, bins=np.logspace(np.log10(np.min(self.mean_flux_long)), np.log10(np.max(self.mean_flux_long)), nbin), histtype="step", color="red", label=f"GBM, {len(self.long_gbm_ph_flux)} GRB", weights=[1 / gbmduty] * len(self.long_gbm_ph_flux))
    # ax7.set(title="Mean flux distributions", xlabel="Photon flux (photon/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
    # ax7.legend()
    #
    # ax8.hist(np.array(self.t90_long)[index], bins=nbin, histtype="step", color="blue", label=f"Sample, {len(np.array(self.t90_long)[index])} GRB")
    # ax8.hist(self.long_gbm_t90, bins=nbin, histtype="step", color="red", label=f"GBM, {len(self.long_gbm_t90)} GRB", weights=[1 / gbmduty] * len(self.long_gbm_t90))
    # ax8.set(title="T90 distributions", xlabel="T90 (s)", ylabel="Number of GRB", xscale="linear", yscale=yscale)
    # ax8.legend()

