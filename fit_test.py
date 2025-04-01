###### Finding the beta and alpha distribution for short and long burst
# beta is common to short and long GRBs according to Ghirlanda, 2016
# alpha is different for the 2 populations, then a different one should be considered
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import xscale
from funcsample import acc_reject
from time import time
import os

from catalog import Catalog
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm
import matplotlib as mpl

mpl.use("Qt5Agg")

gbm_cat = Catalog("GBM/allGBM.txt", [4, '\n', 5, '|', 4000], "GBM/rest_frame_properties.txt")

np.random.seed(os.getpid() + int(time() * 1000) % 2**32)


def gauss(x, amp, mu, sig):
  # print(type(amp), type(mu), type(sig))
  # print(amp * np.random.normal(loc=mu, scale=sig))
  return amp * norm.pdf(x, loc=mu, scale=sig)


def double_gauss(x, amp1, mu1, sig1, amp2, mu2, sig2):
  # print(type(amp), type(mu), type(sig))
  # print(amp * np.random.normal(loc=mu, scale=sig))
  return gauss(x, amp1, mu1, sig1) + gauss(x, amp2, mu2, sig2)


indexes = False
durations = False
limits = True
distribution_display = False

if indexes:
  # model = df_used["flnc_best_fitting_model"].values
  alpha_band = []
  beta_band = []
  alpha_comp = []
  alpha_sbpl = []
  beta_sbpl = []
  alpha_plaw = []

  # cattype = "short"
  # cattype = "long"
  cattype = "all"
  if cattype == "short":
    df_used = gbm_cat.df[gbm_cat.df.t90 < 2]
  elif cattype == "long":
    df_used = gbm_cat.df[gbm_cat.df.t90 >= 2]
  else:
    df_used = gbm_cat.df

  df_used.index = range(0, len(df_used), 1)

  for ite, model in enumerate(df_used["flnc_best_fitting_model"].values):
    if model == "flnc_band":
      alpha_band.append(df_used.flnc_band_alpha[ite])
      beta_band.append(df_used.flnc_band_beta[ite])
    elif model == "flnc_comp":
      alpha_comp.append(df_used.flnc_comp_index[ite])
    elif model == "flnc_sbpl":
      alpha_sbpl.append(df_used.flnc_sbpl_indx1[ite])
      beta_sbpl.append(df_used.flnc_sbpl_indx2[ite])
    elif model == "flnc_plaw":
      alpha_plaw.append(df_used.flnc_plaw_index[ite])

  full_alpha = alpha_band + alpha_comp + alpha_sbpl
  full_beta = beta_band + beta_sbpl

  kde_a = gaussian_kde(full_alpha)
  kde_b = gaussian_kde(full_beta)
  xa = np.random.random(1000) * (4 + 1) - 4
  ya = kde_a.pdf(xa) * 350
  xb = np.random.random(1000) * (5 - 0.9) - 5
  yb = kde_b.pdf(xb) * 200
  popt_a, pcov_a = curve_fit(gauss, xdata=xa, ydata=ya, bounds=((0, -np.inf, 0), (np.inf, np.inf, np.inf)))
  popt_b, pcov_b = curve_fit(gauss, xdata=xb, ydata=yb, bounds=((0, -np.inf, 0), (np.inf, np.inf, np.inf)))
  print(popt_a, pcov_a)
  print(popt_b, pcov_b)
  yaf = gauss(xa, *popt_a)
  ybf = gauss(xb, *popt_b)

  nbin = 30
  binsa = np.linspace(-4, 1, nbin)
  binsb = np.linspace(-5, -0.9, nbin)
  dis, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
  ax1.hist(alpha_band, bins=binsa, histtype="step", label="Alpha band")
  ax1.hist(alpha_comp, bins=binsa, histtype="step",  label="Alpha comp")
  ax1.hist(alpha_sbpl, bins=binsa, histtype="step",  label="Alpha sbpl")
  # ax1.hist(alpha_plaw, bins=binsa, histtype="step",  label = "Alpha plaw")
  # ax1.scatter(xa, ya, s=1, label="kde")
  # ax1.scatter(xa, yaf, s=1, label=f"gaussian fit\nmu={popt_a[1]:.2f}+-{pcov_a[1][1]:.2f}\nsig={popt_a[2]:.2f}+-{pcov_a[2][2]:.2f}")
  ax1.hist(full_alpha, bins=binsa, histtype="step",  label = "Alpha all", color="black")
  ax1.set(xlabel=r"$\alpha$", ylabel="Number of values")
  ax1.legend()

  ax2.hist(beta_band, bins=binsb, histtype="step",  label="Beta band")
  ax2.hist(beta_sbpl, bins=binsb, histtype="step",  label="Beta sbpl")
  # ax2.scatter(xb, yb, s=1, label="kde")
  # ax2.scatter(xb, ybf, s=1, label=f"gaussian fit\nmu={popt_b[1]:.2f}+-{pcov_b[1][1]:.2f}\nsig={popt_b[2]:.2f}+-{pcov_b[2][2]:.2f}")
  ax2.hist(full_beta, bins=binsb, histtype="step", label="Beta all", color="black")
  ax2.set(xlabel=r"$\beta$", ylabel="Number of values")
  ax2.legend()

  plt.suptitle(f"{cattype} GRB")
  plt.show()

# Fitting the T90 distribution
if durations:
  times = gbm_cat.df.t90.values
  times_long = gbm_cat.df.t90[gbm_cat.df.t90 > 2].values
  times_short = gbm_cat.df.t90[gbm_cat.df.t90 <= 2].values

  nbin = 24
  bins = np.logspace(-3, 3, nbin)
  xlist = np.logspace(-3, 3, 1000)
  timedis, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

  hist_long = ax1.hist(times_long, bins=bins, histtype="step", label="lGRBs")
  x_long, y_long = np.log10((hist_long[1][1:] + hist_long[1][:-1])/2), hist_long[0]
  popt_long, pcov_long = curve_fit(gauss, xdata=x_long, ydata=y_long, bounds=((100, np.log10(2), 0.1), (np.inf, 3, 3)))
  ax1.plot(xlist, gauss(np.log10(xlist), *popt_long), label="Gaussian fit")
  made_long = []
  while len(made_long) < len(times_long):
    temp_time = 10**np.random.normal(1.4875, 0.45669)
    if temp_time > 2:
      made_long.append(temp_time)
  ax1.hist(made_long, bins=bins, histtype="step", label="Made lGRBs")
  ax1.set(xlabel="Time (s)", ylabel="Number of GRB", xscale="log", yscale="linear")
  ax1.legend()
  print(popt_long)

  hist_short = ax2.hist(times_short, bins=bins, histtype="step", label="sGRBs")
  x_short, y_short = np.log10((hist_short[1][1:] + hist_short[1][:-1])/2), hist_short[0]
  popt_short, pcov_short = curve_fit(gauss, xdata=x_short, ydata=y_short, bounds=((50, -3, 0.1), (np.inf, np.log10(2), 3)))
  ax2.plot(xlist, gauss(np.log10(xlist), *popt_short), label="Gaussian fit")
  made_short = []
  while len(made_short) < len(times_short):
    temp_time = 10**np.random.normal(-0.025, 0.631)
    if temp_time <= 2:
      made_short.append(temp_time)

  ax2.hist(made_short, bins=bins, histtype="step", label="Made sGRBs")
  ax2.set(xlabel="Time (s)", ylabel="Number of GRB", xscale="log", yscale="linear")
  ax2.legend()
  print(popt_short)

  hist_all = ax3.hist(times, bins=bins, histtype="step", label="All GRBs")
  x_all, y_all = np.log10((hist_all[1][1:] + hist_all[1][:-1])/2), hist_all[0]
  popt_all, pcov_all = curve_fit(double_gauss, xdata=x_all, ydata=y_all, p0=[popt_long[0], popt_long[1], popt_long[2], popt_short[0], popt_short[1], popt_short[2]])#bounds=((100, np.log10(2), 0.1, 50, -3, 0.1), (np.inf, 3, 3, np.inf, np.log10(2), 3)))
  ax3.plot(xlist, double_gauss(np.log10(xlist), *popt_all), label="Double gaussian fit")
  made_all = made_long + made_short
  ax3.hist(made_all, bins=bins, histtype="step", label="Made all GRBs")

  ax3.set(xlabel="Time (s)", ylabel="Number of GRB", xscale="log", yscale="linear")
  ax3.legend()
  print("== Fit result ==")
  print(f"amplitude long : {popt_all[0]}, mean long : {popt_all[1]}, stdev long : {popt_all[2]}")
  print(f"amplitude short : {popt_all[3]}, mean short : {popt_all[4]}, stdev short : {popt_all[5]}")
  for ite in range(len(pcov_all)):
    print("error : ", pcov_all[ite][ite])

  # kept :
  # amplitude long : 467, mean long : 1.4875, stdev long : 0.45669
  # amplitude short : 137.5, mean short : -0.025, stdev short : 0.631

  plt.show()

if limits:
  from funcmod import use_scipyquad
  from astropy.cosmology import FlatLambdaCDM
  cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

  def redshift_1(red, red0, n1, n2, z1):
    """
    Version
      redshift distribution for long GRBs
      Function and associated parameters and cases are taken from Sarah Antier's thesis
    :param red: float or array of float containing redshifts
    """
    if type(red) is float or type(red) is int:
      if red <= z1:
        dist = red0 * (1 + red) ** n1
      else:
        dist = red0 * (1 + z1) ** (n1 - n2) * (1 + red) ** n2
    elif type(red) is np.ndarray:
      dist = np.where(red <= z1, red0 * (1 + red) ** n1, red0 * (1 + z1) ** (n1 - n2) * (1 + red) ** n2)
    else:
      raise TypeError("Please use a correct type for red, only accepted are float or numpy ndarray")
    vol_com = cosmo.differential_comoving_volume(red).to_value("Gpc3 / sr")  # Change from Mpc3 / sr to Gpc3 / sr
    return 4 * np.pi * dist / (1 + red) * vol_com


  def redshift_2(red, n0, na, nb, zm):
    """
    Version
      redshift distribution for long GRBs
      Function and associated parameters and cases are taken from Jesse Palmerio k05-A-nF
    :param red: float or array of float containing redshifts
    """

    if type(red) is float or type(red) is int:
      if red < zm:
        dist = n0 * np.exp(na * red)
      else:
        dist = n0 * np.exp(nb * red) * np.exp((na - nb) * zm)
    elif type(red) is np.ndarray:
      dist = np.where(red < zm, n0 * np.exp(na * red), n0 * np.exp(nb * red) * np.exp((na - nb) * zm))
    else:
      raise TypeError("Please use a correct type for red, only accepted are float or numpy ndarray")
    vol_com = cosmo.differential_comoving_volume(red).to_value("Gpc3 / sr")  # Change from Mpc3 / sr to Gpc3 / sr
    return 4 * np.pi * dist / (1 + red) * vol_com


  def redshift_3(red, red0, n1, n2, z1):
    """
    Version
      redshift distribution for long GRBs
      Function and associated parameters and cases are taken from Lan G., 2019
    :param red: float or array of float containing redshifts
    """
    # n1, n2, z1 = 3.85, -1.07, 2.33
    if type(red) is float or type(red) is int:
      if red <= z1:
        dist = red0 * (1 + red) ** n1
      else:
        dist = red0 * (1 + z1) ** (n1 - n2) * (1 + red) ** n2
    elif type(red) is np.ndarray:
      dist = np.where(red <= z1, red0 * (1 + red) ** n1, red0 * (1 + z1) ** (n1 - n2) * (1 + red) ** n2)
    else:
      raise TypeError("Please use a correct type for red, only accepted are float or numpy ndarray")
    vol_com = cosmo.differential_comoving_volume(red).to_value("Gpc3 / sr")  # Change from Mpc3 / sr to Gpc3 / sr
    return 4 * np.pi * dist / (1 + red) * vol_com


  def redshift_short(red, rate0, p1, p2, zp):
    """
    Version
      redshift distribution for short GRBs
      Function and associated parameters and cases are taken from Ghirlanda et al. 2016
    :param red: float or array of float containing redshifts
    """
    # p1, zp, p2 = 2.8, 2.3, 3.5
    # p1, zp, p2 = 3.1, 2.5, 3.6 # test
    if type(red) is float or type(red) is int or type(red) is np.ndarray:
      dist = (1 + p1 * red) / (1 + (red / zp) ** p2)
    else:
      raise TypeError("Please use a correct type for red, only accepted are float or numpy ndarray")
    vol_com = cosmo.differential_comoving_volume(red).to_value("Gpc3 / sr")  # Change from Mpc3 / sr to Gpc3 / sr
    return rate0 * 4 * np.pi * dist / (1 + red) * vol_com


  def plot_redshift_count(func, range1, range2, range3, range4):
    """

    """
    nvals = 100
    xlist1 = np.linspace(range1[0], range1[2], nvals)
    xlist2 = np.linspace(range2[0], range2[2], nvals)
    xlist3 = np.linspace(range3[0], range3[2], nvals)
    xlist4 = np.linspace(range4[0], range4[2], nvals)
    var1 = [nyear * use_scipyquad(func, 0, 10, func_args=(value, range2[1], range3[1], range4[1]), x_logscale=False)[0] for value in xlist1]
    var2 = [nyear * use_scipyquad(func, 0, 10, func_args=(range1[1], value, range3[1], range4[1]), x_logscale=False)[0] for value in xlist2]
    var3 = [nyear * use_scipyquad(func, 0, 10, func_args=(range1[1], range2[1], value, range4[1]), x_logscale=False)[0] for value in xlist3]
    var4 = [nyear * use_scipyquad(func, 0, 10, func_args=(range1[1], range2[1], range3[1], value), x_logscale=False)[0] for value in xlist4]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    ax1.plot(xlist1, var1)
    ax1.axvline(range1[1])
    ax1.set(xlabel="Rate0", ylabel="Number of GRB")

    ax2.plot(xlist2, var2)
    ax2.axvline(range2[1])
    ax2.set(xlabel="Index1", ylabel="Number of GRB")

    ax3.plot(xlist3, var3)
    ax3.axvline(range3[1])
    ax3.set(xlabel="Index2", ylabel="Number of GRB")

    ax4.plot(xlist4, var4)
    ax4.axvline(range4[1])
    ax4.set(xlabel="z cut", ylabel="Number of GRB")

    plt.show()

  nyear = 10
  # zdist1 = np.array([acc_reject(redshift_distribution_long, [], 0, 10) for i in range(1000)])
  red0_1, n1_1, n2_1, z1_1 = 0.42, 2.07, -0.7, 3.6
  nz1 = nyear * use_scipyquad(redshift_1, 0, 10, func_args=(red0_1, n1_1, n2_1, z1_1), x_logscale=False)[0]
  print(nz1)
  # n0_2, na_2, nb_2, zm_2 = 1.8, 1.24, -0.21, 2.11
  # nz1 = nyear * use_scipyquad(redshift_2, 0, 10, func_args=(n0_2, na_2, nb_2, zm_2), x_logscale=False)[0]
  # print(nz1)
  # red0_3, n1_3, n2_3, z1_3 = 1.49, 3.85, -1.07, 2.33
  # nz1 = nyear * use_scipyquad(redshift_3, 0, 10, func_args=(red0_3, n1_3, n2_3, z1_3), x_logscale=False)[0]
  # print(nz1)
  # # + de 47000 sur 5 ans
  # long_rate = 1.49  # +0.63 -0.64   # initial value is 1.49 (divided by 6.7) but this division was needed for a realistic ratio of Long/short GRBs
  # # long_rate = 0.22  # +0.63 -0.64   # initial value is 1.49 (divided by 6.7) but this division was needed for a realistic ratio of Long/short GRBs
  # ind1_z_l = 3.85  # +0.48 -0.45
  # ind2_z_l = -1.07  # +0.98 -1.12
  # zb_l = 2.33  # +0.39 -0.24
  # from funcsample import red_rate_long
  # nz1 = nyear * use_scipyquad(red_rate_long, 0, 10, func_args=(long_rate, ind1_z_l, ind2_z_l, zb_l), x_logscale=False)[0]
  # print(nz1)

  range1_1 = [0.2, 0.42, 0.6]
  range2_1 = [1.5, 2.07, 2.5]
  range3_1 = [-0.5, -0.7, -0.9]
  range4_1 = [3, 3.6, 4.2]
  plot_redshift_count(redshift_1, range1_1, range2_1, range3_1, range4_1)

  # range1_2 = [1.6, 1.8, 2]
  # range2_2 = [1.09, 1.24, 1.41]
  # range3_2 = [-0.27, -0.21, -0.17]
  # range4_2 = [2.04, 2.11, 2.18]
  # plot_redshift_count(redshift_2, range1_2, range2_2, range3_2, range4_2)
  #
  # range1_3 = [0.85, 1.49, 2.12]
  # range2_3 = [3.4, 3.85, 4.33]
  # range3_3 = [-2.19, -1.07, -0.9]
  # range4_3 = [2.09, 2.33, 2.72]
  # plot_redshift_count(redshift_3, range1_3, range2_3, range3_3, range4_3)

  short_rate = 0.20  # 0.8 +0.3 -0.15 [Gpc-3.yr-1] # Ghirlanda 2016 initial value : 0.2 +0.04 -0.07  but 0.8 gives better results in terms of distribution and total number of sGRB
  ind1_z_s = 2.8
  ind2_z_s = 2.3
  zb_s = 3.5
  nz1 = nyear * use_scipyquad(redshift_short, 0, 10, func_args=(short_rate, ind1_z_s, ind2_z_s, zb_s), x_logscale=False)[0]
  print(nz1)
  range1_3 = [0.13, 0.20, 0.24]
  range2_3 = [0.59, 2.8, 3.7]
  range3_3 = [0.94, 3.5, 4]
  range4_3 = [1.7, 2.3, 3.2]
  plot_redshift_count(redshift_short, range1_3, range2_3, range3_3, range4_3)

  short_rate = 0.80  # 0.8 +0.3 -0.15 [Gpc-3.yr-1] # Ghirlanda 2016 initial value : 0.2 +0.04 -0.07  but 0.8 gives better results in terms of distribution and total number of sGRB
  ind1_z_s = 3.1
  ind2_z_s = 3.6
  zb_s = 2.5
  nz1 = nyear * use_scipyquad(redshift_short, 0, 10, func_args=(short_rate, ind1_z_s, ind2_z_s, zb_s), x_logscale=False)[0]
  print(nz1)
  # nz1 = nyear * use_scipyquad(redshift_short, 0, 10, func_args=(short_rate, ind1_z_s, ind2_z_s, zb_s), x_logscale=False)[0]
  # print(nz1)
  range1_3 = [0.65, 0.8, 1.1]
  range2_3 = [0.51, 3.1, 4.1]
  range3_3 = [1.1, 3.6, 3.7]
  range4_3 = [1.1, 2.5, 3.3]
  plot_redshift_count(redshift_short, range1_3, range2_3, range3_3, range4_3)

  # self.nlong_min = 5000
  # self.nlong_max = 100000
  # self.nshort_min = 1500
  # self.nshort_max = 25000
  # self.long_short_rate_min = 4
  # self.long_short_rate_max = 6.5
  #
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
  # self.s_ind2_z_min = 1
  # self.s_ind2_z_max = 4
  # self.s_zb_min = 1.1
  # self.s_zb_max = 3.3
  #
  # # broken_plaw
  # # Luminosity
  # # self.ind1_l = -0.65    -1.14+-0.02    -1.5    -1.5:-1.66,-1.18    -0.74:-2.16,0.62     -1.2
  # # self.ind2_l = -3       -1.7 +-0.03    -3.2    -2.32:-2.64,-1.55   -1.92:-2.06,-1.81     -2.4
  # # self.lb_l = 1.12e52     1.43e51        1e51   3.8:1.1,10.1e52     5.5:2.1,12.4e50      3.16e52
  #
  # self.l_ind1_min = -1.6
  # self.l_ind1_max = -0.7
  # self.l_ind2_min = -2.6
  # self.l_ind2_max = -1.6
  # self.l_lb_min = 2e51
  # self.l_lb_max = 6e52
  #
  # # self.ind1_s = -0.53   -1 -0.39
  # # self.ind2_s = -3.4    -3.7 -1.7
  # # self.lb_s = 2.8e52    0.91 3.4 e52
  # self.s_ind1_min = -1
  # self.s_ind1_max = -0.39
  # self.s_ind2_min = -3.7
  # self.s_ind2_max = -1.7
  # self.s_lb_min = 0.91e52
  # self.s_lb_max = 3.4e52

if distribution_display:
  from time import time
  from funcsample import red_rate_long, red_rate_short, broken_plaw, yonetoku_reverse_short, pick_normal_alpha_beta, norm_band_spec_calc, yonetoku_reverse_long
  from funcmod import extract_lc
  from astropy.cosmology import FlatLambdaCDM

  def transfo_broken_plaw(ind1, ind2, val_b, inf_lim, sup_lim):
    ral = val_b / (ind1 + 1)
    pal = (inf_lim / val_b)**(ind1 + 1)
    rbe = val_b / (ind2 + 1)
    pbe = (sup_lim / val_b)**(ind2 + 1)
    ampl = 1 / (ral * (1 - pal) + rbe * (pbe - 1))

    rb = ampl * ral * (1 - pal)

    rand_val = np.random.random()

    # print(rb, rand_val)

    # print(rand_val, rb)
    if rand_val <= rb:
      return val_b * (rand_val / ampl / ral + pal)**(1/(ind1 + 1))
    else:
      return val_b * (rand_val / ampl / rbe + 1 + ral / rbe * (pal - 1))**(1/(ind2 + 1))

  def int_red_long(r0, n1, n2, z_b, inf_lim, sup_lim):
    rat1 = r0 / (n1 + 1)
    rat2 = r0 * (1 + z_b)**(n1 - n2) / (n2 + 1)
    pb1 = (1 + z_b)**(n1 + 1)
    pb2 = (1 + z_b)**(n2 + 1)
    pinf = (1 + inf_lim)**(n1 + 1)
    ampl = 1 / (rat1 * (pb1 - pinf) + rat2 * ((1 + sup_lim)**(n2 + 1) - pb2))

    rb = ampl * rat1 * (pb1 - pinf)

    # if valtest <= z_b:
    #   return ampl * rat1 * ((1 + valtest)**(n1 + 1) - pinf)
    # else:
    #   return ampl * (rat1 * (pb1 - pinf) + rat2 * ((1 + valtest)**(n2 + 1) - pb2))

    rand_val = np.random.random()

    # print(rb, rand_val)

    # print(rand_val, rb)
    if rand_val <= rb:
      return (rand_val / ampl / rat1 + pinf)**(1/(n1 + 1)) - 1
    else:
      return (rand_val / ampl / rat2 + rat1 / rat2 * (pinf - pb1) + pb2)**(1/(n2 + 1)) - 1


  def get_short(short_rate, ind1_z_s, ind2_z_s, zb_s, ind1_s, ind2_s, lb_s):
    """
    Creates the quatities of a short burst according to distributions
    Based on Lana Salmon's thesis and Ghirlanda et al, 2016
    """
    zmin = 0
    zmax = 10
    lmin = 1e49  # erg/s
    lmax = 3e54
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    band_low_s_mu, band_low_s_sig = -0.57, 0.32
    band_high_s_mu, band_high_s_sig = -2.17, 0.31

    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    timelist = []
    init_tot = time()
    init_time = time()
    z_obs_temp = acc_reject(red_rate_short, [short_rate, ind1_z_s, ind2_z_s, zb_s], zmin, zmax)
    timelist.append(time() - init_time)
    init_time = time()
    # lpeak_rest_temp = acc_reject(broken_plaw, [ind1_s, ind2_s, lb_s], lmin, lmax)
    lpeak_rest_temp = transfo_broken_plaw(ind1_s, ind2_s, lb_s, lmin, lmax)
    timelist.append(time() - init_time)

    ep_rest_temp = yonetoku_reverse_short(lpeak_rest_temp)

    init_time = time()
    band_low_obs_temp, band_high_obs_temp = pick_normal_alpha_beta(band_low_s_mu, band_low_s_sig, band_high_s_mu, band_high_s_sig)
    timelist.append(time() - init_time)
    init_time = time()
    t90_obs_temp = 1000
    while t90_obs_temp > 2:
      t90_obs_temp = 10 ** np.random.normal(-0.025, 0.631)
    timelist.append(time() - init_time)

    dl_obs_temp = cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc

    ##################################################################################################################
    # Calculation of spectrum and data saving
    ##################################################################################################################
    init_time = time()
    ener_range = np.logspace(1, 3, 10001)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    timelist.append(time() - init_time)
    # print(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range)
    print(f"Time total : {time() - init_tot}")
    for times in timelist:
      print(f"Time taken : {times:8.6f}s making {times/(time() - init_tot)*100:5.2f}% of the run")
    return z_obs_temp, lpeak_rest_temp
    # return [temp_mean_flux, temp_peak_flux, temp_mean_flux * t90_obs_temp]

  def get_long(long_rate, ind1_z_l, ind2_z_l, zb_l, ind1_l, ind2_l, lb_l):
    """
    Creates the quatities of a long burst according to distributions
    Based on Sarah Antier's thesis
    """
    zmin = 0
    zmax = 10
    lmin = 1e49  # erg/s
    lmax = 3e54
    band_low_l_mu, band_low_l_sig = -0.95, 0.31
    band_high_l_mu, band_high_l_sig = -2.17, 0.30
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    ##################################################################################################################
    # picking according to distributions
    ##################################################################################################################
    timelist = []
    init_tot = time()
    init_time = time()
    z_obs_temp = acc_reject(red_rate_long, [long_rate, ind1_z_l, ind2_z_l, zb_l], zmin, zmax)
    timelist.append(time() - init_time)
    init_time = time()
    # lpeak_rest_temp = acc_reject(broken_plaw, [ind1_l, ind2_l, lb_l], lmin, lmax)
    lpeak_rest_temp = transfo_broken_plaw(ind1_l, ind2_l, lb_l, lmin, lmax)
    timelist.append(time() - init_time)
    init_time = time()
    band_low_obs_temp, band_high_obs_temp = pick_normal_alpha_beta(band_low_l_mu, band_low_l_sig, band_high_l_mu, band_high_l_sig)
    timelist.append(time() - init_time)
    init_time = time()
    t90_obs_temp = 0
    while t90_obs_temp <= 2:
      t90_obs_temp = 10 ** np.random.normal(1.4875, 0.45669)
    timelist.append(time() - init_time)

    dl_obs_temp = cosmo.luminosity_distance(z_obs_temp).value / 1000  # Gpc

    ep_rest_temp = yonetoku_reverse_long(lpeak_rest_temp)

    # ep_obs_temp = ep_rest_temp / (1 + z_obs_temp)
    # eiso_rest_temp = amati_long(ep_rest_temp)

    ##################################################################################################################
    # Calculation of spectrum and data saving
    ##################################################################################################################
    init_time = time()
    ener_range = np.logspace(1, 3, 10001)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range, verbose=False)
    timelist.append(time() - init_time)
    print(f"Time total : {time() - init_tot}")
    for times in timelist:
      print(f"Time taken : {times:8.6f}s making {times/(time() - init_tot)*100:5.2f}% of the run")
    # return [temp_mean_flux, temp_peak_flux, temp_mean_flux * t90_obs_temp]
    # print(band_low_obs_temp, band_high_obs_temp, z_obs_temp, dl_obs_temp, ep_rest_temp, lpeak_rest_temp, ener_range)
    return z_obs_temp, lpeak_rest_temp

  list_l_int = []
  list_s_int = []
  t_list_l = []
  t_list_s = []
  z_l_list = []
  z_s_list = []
  lum_l_list = []
  lum_s_list = []
  nloop = 10
  for ite in range(nloop):
    print(f"ite {ite}", end="\r")
    init_time = time()
    ret = get_long(0.22, 3.85, -1.07, 2.33, -0.65, -3, 10 ** 52.05)
    z_l_list.append(ret[0])
    lum_l_list.append(ret[1])
    t_list_l.append(time()-init_time)
    ener_range = np.logspace(1, 3, 10001)
    # norm_val, spec, temp_peak_flux = norm_band_spec_calc(-1.0956234599865278, -2.655249785650843, 0.9461736188566983, 6.170977971400905, 362.547485870984, 9.498256988623784e+51, ener_range, verbose=False)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(-1.0559965573344643, -2.847270974209892, 1.7716263185651893, 13.392046775949256, 20.61997919717707, 3.0724906209678515e+49, ener_range, verbose=False)
    # norm_val, spec, temp_peak_flux = norm_band_spec_calc(-0.8657421762646083, -2.4980196601377957, 2.7572524555304447, 22.963918019167913, 337.47815629183833, 8.230106513335245e+51, ener_range, verbose=False)
    list_l_int.append(temp_peak_flux)
  print("====")
  for ite in range(nloop):
    print(f"ite {ite}", end="\r")
    init_time = time()
    ret = get_short(0.80, 2.8, 2.3, 3.5, -0.53, -3.4, 2.8e52)
    z_s_list.append(ret[0])
    lum_s_list.append(ret[1])
    t_list_s.append(time()-init_time)
    ener_range = np.logspace(1, 3, 10001)
    # norm_val, spec, temp_peak_flux = norm_band_spec_calc(0.01104318770787227, -2.1118239854006995, 3.5215692496371878, 30.807868469315167, 669.0295582100977, 9.094407462093352e+51, ener_range, verbose=False)
    norm_val, spec, temp_peak_flux = norm_band_spec_calc(-0.4069282792400512, -2.5896491442117315, 0.4021245668290496, 2.185791838541577, 1170.6133049957748, 1.7701994693279716e+52, ener_range, verbose=False)
    # norm_val, spec, temp_peak_flux = norm_band_spec_calc(-0.8404556401447851, -2.9380325693867535, 1.2898913594893413, 9.053594384219412, 56.16773071803093, 4.762900600827777e+50, ener_range, verbose=False)
    list_s_int.append(temp_peak_flux)

  # vals = np.linspace(0, 10, 101)
  # ys = [int_red_long(val, 0.22, 3.85, -1.07, 2.33, 0, 10) for val in vals]
  # # for val in vals:
  # #   print(int_broken_plaw(val, -0.65, -3, 10 ** 52.05, 1e49, 3e54), "\n")
  # figtt, ax = plt.subplots(1, 1)
  # ax.plot(vals, ys, label="Luminosity")
  # ax.set(xlabel="Redshift", ylabel="r", xscale="linear", yscale="linear")
  # ax.legend()
  # plt.show()

  if nloop > 10:
    from funcsample import redshift_distribution_long
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.hist(z_l_list, bins=30, histtype="step", label="long")
    ax1.plot(np.linspace(0, 10, 101), redshift_distribution_long(np.linspace(0, 10, 101), 0.22, 3.85, -1.07, 2.33))
    ax2.hist(z_s_list, bins=30, histtype="step", label="short")
    ax3.hist(lum_l_list, bins=np.logspace(49, np.log10(3e54), 30), histtype="step", label="long")
    ax3.axvline(np.quantile(lum_l_list, 0.99998), label="quantile 99.999%")
    ax4.hist(lum_s_list, bins=np.logspace(49, np.log10(3e54), 30), histtype="step", label="short")
    ax4.axvline(np.quantile(lum_s_list, 0.99998), label="quantile 99.999%")
    ax1.set(xlabel="Redshift", ylabel="number obtained", xscale="linear", yscale="log")
    ax2.set(xlabel="Redshift", ylabel="number obtained", xscale="linear", yscale="log")
    ax3.set(xlabel="Peak luminosity", ylabel="number obtained", xscale="log", yscale="log")
    ax4.set(xlabel="Peak luminosity", ylabel="number obtained", xscale="log", yscale="log")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.show()

    figt, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(t_list_l, bins=30, histtype="step", label="long")
    ax2.hist(t_list_s, bins=30, histtype="step", label="short")
    ax1.set(xlabel="Computation time", ylabel="number obtained")
    ax2.set(xlabel="Computation time", ylabel="number obtained")
    ax1.legend()
    ax2.legend()
    plt.show()


  # fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
  # ax1.hist(list_l_int, bins=30, histtype="step", label="long")
  # ax2.hist(list_s_int, bins=30, histtype="step", label="short")
  # ax3.hist(t_list_l, bins=30, histtype="step", label="long")
  # ax4.hist(t_list_s, bins=30, histtype="step", label="short")
  # ax1.set(xlabel="Pflux", ylabel="number obtained")
  # ax2.set(xlabel="Pflux", ylabel="number obtained")
  # ax3.set(xlabel="Computation time", ylabel="number obtained")
  # ax4.set(xlabel="Computation time", ylabel="number obtained")
  # ax1.legend()
  # ax2.legend()
  # ax3.legend()
  # ax4.legend()
  # plt.show()
