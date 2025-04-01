import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from funcmod import comp, band, plaw, sbpl, calc_flux_gbm, gauss, double_gauss
from scipy.optimize import curve_fit

# Version 2, Created by Adrien Laviron, updated by Nathan Franel


def treat_item(item_ev, item):
  """
  Convert the event of an item to float, only strip the string otherwise, if the item has no value the value is set to None
  :param item_ev: str, item event
  :param item: str, name of the item treated
  """
  striped_item_ev = item_ev.strip()
  try:
    if item == "bcat_detector_mask" or item == "scat_detector_mask":
      return str(striped_item_ev)
    else:
      return float(striped_item_ev)
  except ValueError:
    if striped_item_ev == "":
      return np.nan
    else:
      if item == "ra" or item == "dec":
        return np.sum(np.array(striped_item_ev.split(" ")).astype(float) / [1, 60, 3600])
      else:
        return striped_item_ev


class Catalog:
  def __init__(self, datafile=None, sttype=None, rest_frame_file=None):
    """
    Instanciates a catalog
    :param datafile: None or string, data to put in the catalog
    :param sttype: See Catalog.fill
    """
    self.cat_type = "GBM"
    self.df = None
    self.datafile = None
    self.sttype = None
    self.rest_frame_file = None
    self.length = 0
    if not (datafile is None or sttype is None or rest_frame_file is None):
      self.sttype = sttype
      self.datafile = datafile
      self.rest_frame_file = rest_frame_file
      self.fill()

  def __len__(self):
    """
    Makes use of built-in len function
    """
    return self.length

  def formatsttype(self):
    """
    Formats self.sttype, the standardized type of text data file
    sttype: iterable of len 5:
      first header event (int)
      event separator (str)
      first event (int)
      item separator (str)
      last event (int) OR list of the sources wanted (list)
    """
    for i in range(5):
      if self.sttype[i] == "n":
        self.sttype[i] = "\n"
      if self.sttype[i] == "t":
        self.sttype[i] = "\t"
      if i % 2 == 0:
        if type(self.sttype[i]) is str and self.sttype[i].startswith('['):
          self.sttype[i] = self.sttype[i][1:-1].split(',')
        else:
          self.sttype[i] = int(self.sttype[i])

  def fill(self):
    """
    Fills a Catalog with data
    """
    # Opening GBM data and removing undesired lines
    self.formatsttype()
    with open(self.datafile) as f:
      d = f.read().split(self.sttype[1])
    if type(self.sttype[4]) is int:
      events = d[self.sttype[2]:self.sttype[4]]
      if events[-1] == '':
        events = events[:-1]
    elif type(self.sttype[4]) is list:
      events = d[self.sttype[2]:]
      if events[-1] == '':
        events = events[:-1]
      events = [event for event in events if event.split(self.sttype[3])[1] in self.sttype[4]]
    else:
      events = []
    self.length = len(events)
    if events[-1] == "":
      self.length -= 1
    header = d[self.sttype[0]]
    # Selecting item names
    items_gbm = [i.strip() for i in header.split(self.sttype[3])]
    if items_gbm[0] == "":
      items_gbm = items_gbm[1:]
    if items_gbm[-1] == "":
      items_gbm = items_gbm[:-1]

    # Opening GBM rest frame data and removing undesired lines and columns
    with open(self.rest_frame_file) as f:
      d = f.read().split("\n")
    rf_events = d[54:]
    while rf_events[-1] == '':
      rf_events = rf_events[:-1]
    # Selecting item names
    items_rest_frame = ["rest_name", 'rest_k_comp', 'rest_eiso_comp', 'rest_eiso_err_comp', 'rest_liso_comp', 'rest_liso_err_comp', 'rest_epeak_comp', 'rest_epeak_err_comp',
                        'rest_k_band', 'rest_eiso_band', 'rest_eiso_err_band', 'rest_liso_band', 'rest_liso_err_band', 'rest_epeak_band', 'rest_epeak_err_band']
    rest_names = []
    rest_values = []
    for ev in rf_events:
      line = ev.split("|")[2:-4]
      rest_names.append(f"GRB{line[0]}")
      rest_values.append(list(map(float, line[1:])))
    items = items_gbm + items_rest_frame[1:]
    index_name = items_gbm.index("name")
    data_tab = []
    for ev in events:
      line = ev.split(self.sttype[3])
      if line[0] == "":
        line = line[1:]
      if line[-1] == "":
        line = line[:-1]
      grb_name = line[index_name]
      data_row = []
      for item_ite in range(len(items_gbm)):
        data_row.append(treat_item(line[item_ite], items_gbm[item_ite]))
      if grb_name in rest_names:
        # Searching for the index in case the 2 list don't have the same order or if there is some GRB in rest data not in GBM data
        id_grb = rest_names.index(grb_name)
        data_row += rest_values[id_grb]
      else:
        data_row += [np.nan] * 14
      data_tab.append(data_row)
    self.df = pd.DataFrame(data=data_tab, columns=items)
    self.set_fluxes()

  def set_fluxes(self):
    """

    """
    pht_mflx_list = []
    pht_pflx_list = []
    for ite in range(len(self.df)):
      model = self.df.flnc_best_fitting_model[ite]
      pht_mflx = float(self.df[f"{model}_phtflux"][ite])

      pfluxmodel = self.df.pflx_best_fitting_model[ite]
      if type(pfluxmodel) == str:
        pht_pflx = float(self.df[f"{pfluxmodel}_phtflux"][ite])
      else:
        if np.isnan(pfluxmodel):
          pht_pflx = np.nan
        else:
          raise ValueError("A value for pflx_best_fitting_model is not set properly")
      pht_mflx_list.append(pht_mflx)
      pht_pflx_list.append(pht_pflx)

    self.df["mean_flux"] = np.array(pht_mflx_list)
    self.df["peak_flux"] = np.array(pht_pflx_list)

  def grb_map_plot(self, mode="no_cm"):
    """
    Display the catalog GRBs position in the sky
    :param mode: no_cm or t90, use t90 to give a color to the points based on the GRB duration
    """
    # Extracting dec and ra from catalog and transforms decimal degrees into degrees into the right frame
    thetap = self.df.dec.values
    phip = np.mod(np.array(self.df.ra.values) + 180, 360) - 180

    plt.subplot(111, projection="aitoff")
    plt.xlabel("RA (°)")
    plt.ylabel("DEC (°)")
    plt.grid(True)
    plt.title("Map of GRB")
    if mode == "no_cm":
      plt.scatter(phip, thetap, s=12, marker="*")
    elif mode == "t90":
      sc = plt.scatter(phip, thetap, s=12, marker="*", c=self.df.t90.values, norm=colors.LogNorm())
      cbar = plt.colorbar(sc)
      cbar.set_label("GRB Duration - T90 (s)", rotation=270, labelpad=20)
    plt.show()

  def spectral_information(self, nbins=50):
    """
    Displays the spectral information of the GRBs including the proportion of different best fit models and the
    corresponding parameters
    :param nbins: number of bins for the histograms of spectra parameters
    """
    # Containers for the different model parameters
    df_band = self.df.loc[self.df.flnc_best_fitting_model == "flnc_band"]
    df_comp = self.df.loc[self.df.flnc_best_fitting_model == "flnc_comp"]
    df_sbpl = self.df.loc[self.df.flnc_best_fitting_model == "flnc_sbpl"]
    df_plaw = self.df.loc[self.df.flnc_best_fitting_model == "flnc_plaw"]
    # Band model
    band_ampl = df_band.flnc_band_ampl.values
    band_alpha = df_band.flnc_band_alpha.values
    band_beta = df_band.flnc_band_beta.values
    band_epeak = df_band.flnc_band_epeak.values
    # Comptonized model
    comp_ampl = df_comp.flnc_comp_ampl.values
    comp_index = df_comp.flnc_comp_index.values
    comp_epeak = df_comp.flnc_comp_epeak.values
    comp_pivot = df_comp.flnc_comp_pivot.values
    # Smoothly broken powerlaw model
    sbpl_ampl = df_sbpl.flnc_sbpl_ampl.values
    sbpl_indx1 = df_sbpl.flnc_sbpl_indx1.values
    sbpl_indx2 = df_sbpl.flnc_sbpl_indx2.values
    sbpl_brken = df_sbpl.flnc_sbpl_brken.values
    sbpl_brksc = df_sbpl.flnc_sbpl_brksc.values
    sbpl_pivot = df_sbpl.flnc_sbpl_pivot.values
    # Powerlaw model
    plaw_ampl = df_plaw.flnc_plaw_ampl.values
    plaw_index = df_plaw.flnc_plaw_index.values
    plaw_pivot = df_plaw.flnc_plaw_pivot.values

    # Plot the proportion of the different models
    prop, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels = ["plaw", "comp", "band", "sbpl"]
    values = [len(plaw_ampl), len(comp_ampl), len(band_ampl), len(sbpl_ampl)]
    ax.pie(values, labels=labels, autopct=lambda x: int(19.28*x))
    plt.show()

    # Plot the distributions of the models' parameters
    plaw_plot, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 6))
    plt.suptitle("plaw parameters")
    ax1.hist(plaw_ampl, bins=nbins)
    ax2.hist(plaw_index, bins=nbins)
    ax3.hist(plaw_pivot, bins=nbins)
    print(f"plaw mean ampl : {np.mean(plaw_ampl)}")
    print(f"plaw mean index : {np.mean(plaw_index)}")
    print(f"plaw mean pivot : {np.mean(plaw_pivot)}")

    ax1.set(xlabel="Amplitude (photon/cm2/s/keV)", ylabel="Number of GRBs")
    ax2.set(xlabel="Index", ylabel="Number of GRBs")
    ax3.set(xlabel="Pivot energy (keV)", ylabel="Number of GRBs")

    plt.show()

    # Plot the distributions of the models' parameters
    comp_plot, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    plt.suptitle("comp parameters")
    ax1.hist(comp_ampl, bins=nbins)
    ax2.hist(comp_index, bins=nbins)
    ax3.hist(comp_epeak, bins=nbins)
    ax4.hist(comp_pivot, bins=nbins)
    print(f"comp mean ampl : {np.mean(comp_ampl)}")
    print(f"comp mean index : {np.mean(comp_index)}")
    print(f"comp mean epeak : {np.mean(comp_epeak)}")
    print(f"comp mean pivot : {np.mean(comp_pivot)}")

    ax1.set(xlabel="Amplitude (photon/cm2/s/keV)", ylabel="Number of GRBs")
    ax2.set(xlabel="Index", ylabel="Number of GRBs")
    ax3.set(xlabel="Peak energy (keV)", ylabel="Number of GRBs")
    ax4.set(xlabel="Pivot energy (keV)", ylabel="Number of GRBs")

    plt.show()

    # Plot the distributions of the models' parameters
    band_plot, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    plt.suptitle("band parameters")
    ax1.hist(band_ampl, bins=nbins)
    ax2.hist(band_alpha, bins=nbins)
    ax3.hist(band_beta, bins=nbins)
    ax4.hist(band_epeak, bins=nbins)
    print(f"band mean ampl : {np.mean(band_ampl)}")
    print(f"band mean alpha : {np.mean(band_alpha)}")
    print(f"band mean beta : {np.mean(band_beta)}")
    print(f"band mean pivot : {np.mean(band_epeak)}")

    ax1.set(xlabel="Amplitude (photon/cm2/s/keV)", ylabel="Number of GRBs")
    ax2.set(xlabel="Alpha index", ylabel="Number of GRBs")
    ax3.set(xlabel="Beta index", ylabel="Number of GRBs")
    ax4.set(xlabel="Peak energy (keV)", ylabel="Number of GRBs")

    plt.show()

    # Plot the distributions of the models' parameters
    sbpl_plot, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(27, 12))
    plt.suptitle("sbpl parameters")
    ax1.hist(sbpl_ampl, bins=nbins)
    ax2.hist(sbpl_indx1, bins=nbins)
    ax3.hist(sbpl_indx2, bins=nbins)
    ax4.hist(sbpl_brken, bins=nbins)
    ax5.hist(sbpl_brksc, bins=nbins)
    ax6.hist(sbpl_pivot, bins=nbins)
    print(f"sbpl mean ampl : {np.mean(sbpl_ampl)}")
    print(f"sbpl mean indx1 : {np.mean(sbpl_indx1)}")
    print(f"sbpl mean indx2 : {np.mean(sbpl_indx2)}")
    print(f"sbpl mean brken : {np.mean(sbpl_brken)}")
    print(f"sbpl mean brksc : {np.mean(sbpl_brksc)}")
    print(f"sbpl mean pivot : {np.mean(sbpl_pivot)}")

    ax1.set(xlabel="Amplitude (photon/cm2/s/keV)", ylabel="Number of GRBs")
    ax2.set(xlabel="Index 1", ylabel="Number of GRBs")
    ax3.set(xlabel="Index 2", ylabel="Number of GRBs")
    ax4.set(xlabel="Break energy (keV)", ylabel="Number of GRBs")
    ax5.set(xlabel="Break scale (keV)", ylabel="Number of GRBs")
    ax6.set(xlabel="Pivot energy (keV)", ylabel="Number of GRBs")

    plt.show()

  def spectra_GBM_mean_param(self):
    """
    Plots the spectra comp, band, plaw, sbpl with GBM mean best fitting parameters
    """
    comp_mean_ampl = 0.016340622602951507
    comp_mean_index = -0.8693237216017569
    comp_mean_epeak = 363.1025391426563
    comp_mean_pivot = 100.0

    band_mean_ampl = 0.03930144912844037
    band_mean_alpha = -0.8282043258256881
    band_mean_beta = -2.2592566376146785
    band_mean_pivot = 209.26818935779812

    plaw_mean_ampl = 0.004935109612015504
    plaw_mean_index = -1.569128538372093
    plaw_mean_pivot = 100.0

    sbpl_mean_ampl = 0.012727944041666668
    sbpl_mean_indx1 = -0.9551795631944445
    sbpl_mean_indx2 = -2.1827798513888887
    sbpl_mean_brken = 162.59191256944445
    sbpl_mean_brksc = 0.3
    sbpl_mean_pivot = 100.0

    tx = np.logspace(0, 4, 1000)

    # e, ampl, index_l, ep, pivot=100
    ty1 = comp(tx, comp_mean_ampl, comp_mean_index, comp_mean_epeak, comp_mean_pivot)
    tyy1 = [tx[ite1]**2 * ty1[ite1] for ite1 in range(len(tx))]
    # e, ampl, alpha, beta, ep, pivot=100
    ty2 = [band(valx, band_mean_ampl, band_mean_alpha, band_mean_beta, band_mean_pivot) for valx in tx]
    tyy2 = [tx[ite2]**2 * ty2[ite2] for ite2 in range(len(tx))]
    # e, ampl, index_l, pivot=100
    ty3 = plaw(tx, plaw_mean_ampl, plaw_mean_index, plaw_mean_pivot)
    tyy3 = [tx[ite3]**2 * ty3[ite3] for ite3 in range(len(tx))]
    # e, ampl, l1, l2, eb, delta, pivot=100
    ty4 = sbpl(tx, sbpl_mean_ampl, sbpl_mean_indx1, sbpl_mean_indx2, sbpl_mean_brken, sbpl_mean_brksc, sbpl_mean_pivot)
    tyy4 = [tx[ite4]**2 * ty4[ite4] for ite4 in range(len(tx))]

    spec, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle("GBM mean spectra (ph/cm2/keV/s)")
    ax1.plot(tx, ty1)
    ax1.axvline(comp_mean_epeak)
    ax1.set(title="Comptonized spectrum", xscale="log", yscale="log")
    ax2.plot(tx, ty2)
    ax2.axvline(band_mean_pivot)
    ax2.set(title="Band spectrum", xscale="log", yscale="log")
    ax3.plot(tx, ty3)
    ax3.axvline(plaw_mean_pivot)
    ax3.set(title="Power law spectrum", xscale="log", yscale="log")
    ax4.plot(tx, ty4)
    ax4.axvline(sbpl_mean_brken)
    ax4.set(title="Smoothly broken power law spectrum", xscale="log", yscale="log")
    plt.show()

    pow_spec, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle("GBM mean power density spectra (keV/cm2/s)")
    ax1.plot(tx, tyy1)
    ax1.axvline(comp_mean_epeak)
    ax1.set(title="Comptonized spectrum", xscale="log", yscale="log")
    ax2.plot(tx, tyy2)
    ax2.axvline(band_mean_pivot)
    ax2.set(title="Band spectrum", xscale="log", yscale="log")
    ax3.plot(tx, tyy3)
    ax3.axvline(plaw_mean_pivot)
    ax3.set(title="Power law spectrum", xscale="log", yscale="log")
    ax4.plot(tx, tyy4)
    ax4.axvline(sbpl_mean_brken)
    ax4.set(title="Smoothly broken power law spectrum", xscale="log", yscale="log")
    plt.show()

  def grb_distribution(self):
    gbm_ph_flux = []
    long_gbm_ph_flux = []
    short_gbm_ph_flux = []
    all_df = self.df.loc[np.logical_not(np.isnan(self.df.flnc_band_epeak))]
    long_df = all_df.loc[all_df.t90 > 2]
    short_df = all_df.loc[all_df.t90 <= 2]
    for ite_gbm, gbm_ep in enumerate(self.df.flnc_band_epeak):
      if not np.isnan(gbm_ep):
        ph_flux = calc_flux_gbm(self, ite_gbm, (10, 1000))
        gbm_ph_flux.append(ph_flux)
        if self.df.t90[ite_gbm] > 2:
          long_gbm_ph_flux.append(ph_flux)
        else:
          short_gbm_ph_flux.append(ph_flux)
    gbm_ph_flux = np.array(gbm_ph_flux)
    long_gbm_ph_flux = np.array(long_gbm_ph_flux)
    short_gbm_ph_flux = np.array(short_gbm_ph_flux)

    dist_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(27, 12))
    nbin = 60
    gbmcorrec = 1 / 0.587 / 10
    lenall = len(all_df)
    lenlong = len(long_df)
    lenshort = len(short_df)

    epmin, epmax = np.min(all_df.flnc_band_epeak.values), np.max(all_df.flnc_band_epeak.values)
    bins1 = np.logspace(np.log10(epmin), np.log10(epmax), nbin)
    ax1.hist(all_df.flnc_band_epeak.values, bins=bins1, histtype="step", color="blue", label=f"GBM, {lenall} GRB", weights=[gbmcorrec] * lenall)
    ax1.hist(long_df.flnc_band_epeak.values, bins=bins1, histtype="step", color="red", label=f"GBM long, {lenlong} GRB", weights=[gbmcorrec] * lenlong)
    ax1.hist(short_df.flnc_band_epeak.values, bins=bins1, histtype="step", color="green", label=f"GBM short, {lenshort} GRB", weights=[gbmcorrec] * lenshort)
    ax1.set(title="Ep distributions", xlabel="Peak energy (keV)", ylabel="Number of GRB", xscale="log", yscale="log")
    ax1.legend()

    ph_fluence_min, ph_fluence_max = np.min(gbm_ph_flux * all_df.t90.values), np.max(gbm_ph_flux * all_df.t90.values)
    bins2 = np.logspace(np.log10(ph_fluence_min), np.log10(ph_fluence_max), nbin)
    ax2.hist(gbm_ph_flux * all_df.t90.values, bins=bins2, histtype="step", color="blue", label=f"GBM, {lenall} GRB", weights=[gbmcorrec] * lenall)
    ax2.hist(long_gbm_ph_flux * long_df.t90.values, bins=bins2, histtype="step", color="red", label=f"GBM long, {lenlong} GRB", weights=[gbmcorrec] * lenlong)
    ax2.hist(short_gbm_ph_flux * short_df.t90.values, bins=bins2, histtype="step", color="green", label=f"GBM short, {lenshort} GRB", weights=[gbmcorrec] * lenshort)
    ax2.set(title="Fluence distributions", xlabel="Photon fluence (photon/cm²)", ylabel="Number of GRB", xscale="log", yscale="log")
    ax2.legend()

    ph_flux_min, ph_flux_max = np.min(gbm_ph_flux), np.max(gbm_ph_flux)
    bins3 = np.logspace(np.log10(ph_flux_min), np.log10(ph_flux_max), nbin)
    ax3.hist(gbm_ph_flux, bins=bins3, histtype="step", color="blue", label=f"GBM, {lenall} GRB", weights=[gbmcorrec] * lenall)
    ax3.hist(long_gbm_ph_flux, bins=bins3, histtype="step", color="red", label=f"GBM long, {lenlong} GRB", weights=[gbmcorrec] * lenlong)
    ax3.hist(short_gbm_ph_flux, bins=bins3, histtype="step", color="green", label=f"GBM short, {lenshort} GRB", weights=[gbmcorrec] * lenshort)
    ax3.set(title="Mean flux distributions", xlabel="Photon flux (photon/cm²/s)", ylabel="Number of GRB", xscale="log", yscale="log")
    ax3.legend()

    t90_min, t90_max = np.min(all_df.t90.values), np.max(all_df.t90.values)
    bins4 = np.logspace(np.log10(t90_min), np.log10(t90_max), nbin)
    ax4.hist(all_df.t90.values, bins=bins4, histtype="step", color="blue", label=f"GBM, {lenall} GRB", weights=[gbmcorrec] * lenall)
    ax4.hist(long_df.t90.values, bins=bins4, histtype="step", color="red", label=f"GBM long, {lenlong} GRB", weights=[gbmcorrec] * lenlong)
    ax4.hist(short_df.t90.values, bins=bins4, histtype="step", color="green", label=f"GBM short, {lenshort} GRB", weights=[gbmcorrec] * lenshort)
    ax4.set(title="T90 distributions", xlabel="T90 (s)", ylabel="Number of GRB", xscale="log", yscale="log")
    ax4.legend()

    plt.show()

  def T90_hardness_graphs(self, show_fit_stats=False):
    """

    """
    mpl.use("Qt5Agg")

    # Extracting and creating the values I need
    temp_df = self.df[["name", "t90"]].copy()
    ergcut1 = (10, 100)
    ergcut2 = (100, 300)
    temp_df["flux_low"] = [calc_flux_gbm(self, source_ite, ergcut1) for source_ite in range(len(self.df))]
    temp_df["flux_high"] = [calc_flux_gbm(self, source_ite, ergcut2) for source_ite in range(len(self.df))]
    temp_df["log_HR"] = np.log10(temp_df.flux_high.values / temp_df.flux_low.values)
    temp_df["log_t90"] = np.log10(temp_df.t90.values)
    temp_df["type"] = ["Short" if t90 <= 2 else "Long" for t90 in temp_df.t90.values]

    # print(temp_df)

    # Fitting the T90 distribution
    bins = np.logspace(-3, 3, 30)
    log_bin_centroids = np.log10((bins[1:] + bins[:-1]) / 2)  # Value is logged because the fit is in the log phase diagram
    short_hist = np.histogram(temp_df[temp_df.t90 <= 2].t90.values, bins=bins)[0]
    popt_short, pcov_short = curve_fit(gauss, log_bin_centroids, short_hist)[:2]

    long_hist = np.histogram(temp_df[temp_df.t90 > 2].t90.values, bins=bins)[0]
    popt_long, pcov_long = curve_fit(gauss, log_bin_centroids, long_hist)[:2]

    energies = np.logspace(-3, 3, 1000)
    energies_log = np.log10(energies)

    fig_t90, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(temp_df.t90.values, bins=bins, label="T90 distribution")
    ax.plot(energies, gauss(energies_log, *popt_short), label="T90 distribution of short GRBs")
    ax.plot(energies, gauss(energies_log, *popt_long), label="T90 distribution of long GRBs")
    # ax.plot(energies, gauss(energies_log, *popt_short) + gauss(energies_log, *popt_long), label="T90 distribution of all GRBs")
    ax.set(xlabel="T90 (s)", ylabel="Number of GRB over the 10 year GBM catalog", xscale="log", yscale="linear", ylim=(0.7, 500))

    nshort = 320
    lshort = []
    short_loc, short_scale = round(popt_short[1], 4), round(popt_short[2], 4)
    for ite in range(nshort):
      tshort = 1000
      while tshort > 2:
        tshort = 10 ** np.random.normal(short_loc, short_scale)
      lshort.append(tshort)
    nlong = 2000
    llong = []
    long_loc, long_scale = round(popt_long[1], 4), round(popt_long[2], 4)
    for ite in range(nlong):
      tlong = 0
      while tlong <= 2:
        tlong = 10 ** np.random.normal(long_loc, long_scale)
      llong.append(tlong)
    ltimes = lshort + llong
    ax.hist(ltimes, bins=bins, histtype="step", label=f"T90 drawn distribution \nshort fit : $\mu$ = {short_loc} $\pm$ {pcov_short[1][1]:.4f}, $\sigma$ = {short_scale} $\pm$ {pcov_short[2][2]:.4f}\nlong fit : $\mu$ = {long_loc} $\pm$ {pcov_long[1][1]:.4f}, $\sigma$ = {long_scale} $\pm$ {pcov_long[2][2]:.4f}")

    ax.legend(loc="upper left")
    plt.show()

    fig_hr, ax_hr = plt.subplots(1, 1, figsize=(10, 6))
    sns.scatterplot(data=temp_df, x="log_t90", y="log_HR", ax=ax_hr, s=10)
    sns.kdeplot(data=temp_df, x="log_t90", y="log_HR", ax=ax_hr, thresh=.1)

  def spectral_index_graphs(self):
    """

    """

    cat_type_list = ["short", "long", "all"]

    for cat_type in cat_type_list:
      if cat_type == "short":
        df_used = self.df[self.df.t90 < 2]
      elif cat_type == "long":
        df_used = self.df[self.df.t90 >= 2]
      else:
        df_used = self.df

      df_used.index = range(0, len(df_used), 1)

      alpha_band = []
      beta_band = []
      alpha_comp = []
      alpha_sbpl = []
      beta_sbpl = []
      alpha_plaw = []
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

      nbin = 30
      binsa = np.linspace(-4, 1, nbin)
      binsb = np.linspace(-5, -0.9, nbin)

      ya, xa = np.histogram(full_alpha, bins=binsa)[:2]
      yb, xb = np.histogram(full_beta, bins=binsb)[:2]
      xa = (xa[1:] + xa[:-1])/2
      xb = (xb[1:] + xb[:-1])/2
      popt_a, pcov_a = curve_fit(gauss, xdata=xa, ydata=ya, bounds=((0, -np.inf, 0.1), (np.inf, np.inf, 1)))
      popt_b, pcov_b = curve_fit(gauss, xdata=xb, ydata=yb, bounds=((0, -np.inf, 0.1), (np.inf, np.inf, 1)))

      dis_a, ax1 = plt.subplots(1, 1, figsize=(10, 6))
      plt.suptitle(f"Alpha distribution {cat_type} GRB")
      ax1.hist(alpha_band, bins=binsa, histtype="step", label="BAND", color="blue")
      ax1.hist(alpha_comp, bins=binsa, histtype="step", label="COMP", color="green")
      ax1.hist(alpha_sbpl, bins=binsa, histtype="step", label="SBPL", color="red")
      ax1.hist(alpha_plaw, bins=binsa, histtype="step",  label="PLAW", color="orange")
      ax1.plot(np.linspace(-4, 1, 1000), gauss(np.linspace(-4, 1, 1000), *popt_a), label=f"Gaussian fit\n$\mu$ = {popt_a[1]:.4f} $\pm$ {pcov_a[1][1]:.4f}\n$\sigma$ = {popt_a[2]:.4f} $\pm$ {pcov_a[2][2]:.4f}", color="purple")
      ax1.hist(full_alpha, bins=binsa, histtype="step", label="BAND + COMP + SBPL", color="black")
      ax1.set(xlabel=r"$\alpha$", ylabel="Number of values")
      ax1.legend(loc="upper left")
      plt.show()

      dis_a, ax2 = plt.subplots(1, 1, figsize=(10, 6))
      plt.suptitle(f"Beta distribution {cat_type} GRB")
      ax2.hist(beta_band, bins=binsb, histtype="step", label="BAND", color="blue")
      ax2.hist(beta_sbpl, bins=binsb, histtype="step", label="SBPL", color="red")
      ax2.plot(np.linspace(-5, -0.9, 1000), gauss(np.linspace(-5, -0.9, 1000), *popt_b), label=f"Gaussian fit\n$\mu$ = {popt_b[1]:.4f} $\pm$ {pcov_b[1][1]:.4f}\n$\sigma$ = {popt_b[2]:.4f} $\pm$ {pcov_b[2][2]:.4f}", color="purple")
      ax2.hist(full_beta, bins=binsb, histtype="step", label="BAND + SBPL", color="black")
      ax2.set(xlabel=r"$\beta$", ylabel="Number of values")
      ax2.legend()

      plt.show()


class SampleCatalog:
  def __init__(self, datafile=None, sttype=None):
    """
    Instanciates a catalog
    :param datafile: None or string, data to put in the catalog
    """
    self.cat_type = "sampled"
    self.df = None
    self.datafile = None
    self.sttype = None
    self.length = 0
    self.columns = ["name", "z_obs", "dl", "ep_rest", "alpha", "beta", "liso", "eiso", "thetaj", "mean_flux", "peak_flux", "t90", "fluence", "lc"]

    # Catalog attributes
    if not (datafile is None or sttype is None):
      self.sttype = sttype
      self.datafile = datafile
      self.fill()


  def __len__(self):
    """
    Makes use of built-in len function
    """
    return self.length

  def formatsttype(self):
    """
    Formats self.sttype, the standardized type of text data file
    """
    for i in range(5):
      if self.sttype[i] == "n":
        self.sttype[i] = "\n"
      if self.sttype[i] == "t":
        self.sttype[i] = "\t"
      if i % 2 == 0:
        if type(self.sttype[i]) is str and self.sttype[i].startswith('['):
          self.sttype[i] = self.sttype[i][1:-1].split(',')
        else:
          self.sttype[i] = int(self.sttype[i])

  def fill(self):
    """
    Fills a Catalog with data
    :param datafile: string, data file name
    :param sttype: iterable of len 5:
      first header event (int)
      event separator (str)
      first event (int)
      item separator (str)
      last event (int) OR list of the sources wanted (list)
    """
    self.formatsttype()
    with open(self.datafile) as f:
      lines = f.read().split(self.sttype[1])  # Getting rid of the header
    if type(self.sttype[4]) is int:
      events = lines[self.sttype[2]:self.sttype[4]]
      if events[-1] == '':
        events = events[:-1]
    elif type(self.sttype[4]) is list:
      events = lines[self.sttype[2]:]
      if events[-1] == '':
        events = events[:-1]
      events = [event for event in events if event.split(self.sttype[3])[1] in self.sttype[4]]
    else:
      events = []
    self.length = len(events)
    if events[-1] == "":
      self.length -= 1
    data_tab = []
    for line in events:
      data = line.split("|")
      row = [data[0], float(data[6]), float(data[10]), float(data[9]), float(data[7]), float(data[8]), float(data[11]), float(data[12]),
             float(data[13]), float(data[4]), float(data[5]), float(data[1]), float(data[3]), data[2]]
      data_tab.append(row)
    self.df = pd.DataFrame(data=data_tab, columns=self.columns)

  def items(self):
    """
    List all knowns items
    """
    return list(self.__dict__.keys())[3:]
