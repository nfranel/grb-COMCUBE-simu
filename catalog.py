import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from funcmod import comp, band, plaw, sbpl

# Version 2, Created by Adrien Laviron, updated by Nathan Franel


class Catalog:
  def __init__(self, data=None, sttype=None):
    """
    Instanciates a catalog
    :param data: None or string, data to put in the catalog
    :param sttype: See Catalog.fill
    """
    self.cat_type = "GBM"
    # Fields added for some clarity
    # Peak parameters
    self.pflx_best_fitting_model = None
    self.pflx_plaw_phtflux = None
    self.pflx_comp_phtflux = None
    self.pflx_band_phtflux = None
    self.pflx_sbpl_phtflux = None
    # Spectra parameters
    self.flnc_best_fitting_model = None
    self.flnc_plaw_ampl = None
    self.flnc_plaw_index = None
    self.flnc_plaw_pivot = None
    self.flnc_comp_ampl = None
    self.flnc_comp_index = None
    self.flnc_comp_epeak = None
    self.flnc_comp_pivot = None
    self.flnc_band_ampl = None
    self.flnc_band_alpha = None
    self.flnc_band_beta = None
    self.flnc_band_epeak = None
    self.flnc_sbpl_ampl = None
    self.flnc_sbpl_indx1 = None
    self.flnc_sbpl_indx2 = None
    self.flnc_sbpl_brken = None
    self.flnc_sbpl_brksc = None
    self.flnc_sbpl_pivot = None
    # Lightcurve calculation attributes
    self.t90_start = None
    self.duration_energy_low = None
    self.duration_energy_high = None
    self.back_interval_low_start = None
    self.back_interval_low_stop = None
    self.back_interval_high_start = None
    self.back_interval_high_stop = None
    self.bcat_detector_mask = None
    self.scat_detector_mask = None
    self.flnc_spectrum_start = None
    self.flnc_spectrum_stop = None

    # Catalog attributes
    self.name = None
    self.t90 = None
    self.length = 0
    self.data = None
    self.sttype = None
    self.dec = None
    self.ra = None
    if not (data is None or sttype is None):
      self.fill(data, sttype)

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

  def tofloat(self, item, default=0):
    """
    Convert an item of all events to float
    :param item: str, item
    :param default: default value, default=0
    """
    if not (hasattr(self, item)):
      raise AttributeError("Catalog does not contain item {}".format(item))
    for i in range(self.length):
      try:
        getattr(self, item)[i] = float(getattr(self, item)[i])
      except ValueError:
        getattr(self, item)[i] = default

  def tofloats(self, items, defaults=0):
    """
    Convert several items of all events to float
    :param items: list of str, items
    :param defaults: value or list of values, default values, default=0
    """
    if not (hasattr(defaults, "__iter__")):
      defaults = np.full(len(items), defaults)
    for item, default in zip(items, defaults):
      self.tofloat(item, default)

  def fill(self, data, sttype):
    """
    Fills a Catalog with data
    :param data: string, data file name
    :param sttype: iterable of len 5:
      first header event (int)
      event separator (str)
      first event (int)
      item separator (str)
      last event (int) OR list of the sources wanted (list)
    """
    self.data = data
    self.sttype = sttype
    self.formatsttype()
    with open(data) as f:
      d = f.read().split(sttype[1])
    if type(sttype[4]) is int:
      events = d[sttype[2]:sttype[4]]
      if events[-1] == '':
        events = events[:-1]
    elif type(sttype[4]) is list:
      events = d[sttype[2]:]
      if events[-1] == '':
        events = events[:-1]
      events = [event for event in events if event.split(sttype[3])[1] in sttype[4]]
    else:
      events = []
    self.length = len(events)
    if events[-1] == "":
      self.length -= 1
    header = d[sttype[0]]
    items = [i.strip() for i in header.split(sttype[3])]
    c = 0  # Compteur d'Empty
    for i in range(len(items)):
      if items[i] == "":
        items[i] = "Empty{}".format(c)
        c += 1
    for item in items:
      setattr(self, item, list())
    for e in events:
      for item, value in zip(items, e.split(sttype[3])):
        getattr(self, item).append(value)

  def items(self):
    """
    List all knowns items
    """
    return list(self.__dict__.keys())[3:]

  def grb_map_plot(self, mode="no_cm"):
    """
    Display the catalog GRBs position in the sky
    :param mode: no_cm or t90, use t90 to give a color to the pointsbased on the GRB duration
    """
    # Extracting dec and ra from catalog and transforms decimal degrees into degrees into the right frame
    thetap = [np.sum(np.array(dec.split(" ")).astype(float) / [1, 60, 3600]) if len(dec.split("+")) == 2 else np.sum(np.array(dec.split(" ")).astype(float) / [1, -60, -3600]) for dec in self.dec]
    thetap = np.deg2rad(np.array(thetap))
    phip = [np.sum(np.array(ra.split(" ")).astype(float) / [1, 60, 3600]) if len(ra.split("+")) == 2 else np.sum(np.array(ra.split(" ")).astype(float) / [1, -60, -3600]) for ra in self.ra]
    phip = np.mod(np.deg2rad(np.array(phip)) + np.pi, 2 * np.pi) - np.pi

    plt.subplot(111, projection="aitoff")
    plt.xlabel("RA (°)")
    plt.ylabel("DEC (°)")
    plt.grid(True)
    plt.title("Map of GRB")
    if mode == "no_cm":
      plt.scatter(phip, thetap, s=12, marker="*")
    elif mode == "t90":
      self.tofloat("t90")
      sc = plt.scatter(phip, thetap, s=12, marker="*", c=self.t90, norm=colors.LogNorm())
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
    # Powerlaw model
    plaw_ampl = []
    plaw_index = []
    plaw_pivot = []
    # Comptonized model
    comp_ampl = []
    comp_index = []
    comp_epeak = []
    comp_pivot = []
    # Band model
    band_ampl = []
    band_alpha = []
    band_beta = []
    band_epeak = []
    # Smoothly broken powerlaw model
    sbpl_ampl = []
    sbpl_indx1 = []
    sbpl_indx2 = []
    sbpl_brken = []
    sbpl_brksc = []
    sbpl_pivot = []
    for ite in range(len(self.name)):
      model = self.flnc_best_fitting_model[ite].rstrip()
      if model == "flnc_plaw":
        self.tofloat('flnc_plaw_ampl')
        self.tofloat('flnc_plaw_index')
        self.tofloat('flnc_plaw_pivot')
        plaw_ampl.append(self.flnc_plaw_ampl[ite])
        plaw_index.append(self.flnc_plaw_index[ite])
        plaw_pivot.append(self.flnc_plaw_pivot[ite])

      elif model == "flnc_comp":
        self.tofloat('flnc_comp_ampl')
        self.tofloat('flnc_comp_index')
        self.tofloat('flnc_comp_epeak')
        self.tofloat('flnc_comp_pivot')
        comp_ampl.append(self.flnc_comp_ampl[ite])
        comp_index.append(self.flnc_comp_index[ite])
        comp_epeak.append(self.flnc_comp_epeak[ite])
        comp_pivot.append(self.flnc_comp_pivot[ite])

      elif model == "flnc_band":
        self.tofloat('flnc_band_ampl')
        self.tofloat('flnc_band_alpha')
        self.tofloat('flnc_band_beta')
        self.tofloat('flnc_band_epeak')
        band_ampl.append(self.flnc_band_ampl[ite])
        band_alpha.append(self.flnc_band_alpha[ite])
        band_beta.append(self.flnc_band_beta[ite])
        band_epeak.append(self.flnc_band_epeak[ite])

      elif model == "flnc_sbpl":
        self.tofloat('flnc_sbpl_ampl')
        self.tofloat('flnc_sbpl_indx1')
        self.tofloat('flnc_sbpl_indx2')
        self.tofloat('flnc_sbpl_brken')
        self.tofloat('flnc_sbpl_brksc')
        self.tofloat('flnc_sbpl_pivot')
        sbpl_ampl.append(self.flnc_sbpl_ampl[ite])
        sbpl_indx1.append(self.flnc_sbpl_indx1[ite])
        sbpl_indx2.append(self.flnc_sbpl_indx2[ite])
        sbpl_brken.append(self.flnc_sbpl_brken[ite])
        sbpl_brksc.append(self.flnc_sbpl_brksc[ite])
        sbpl_pivot.append(self.flnc_sbpl_pivot[ite])

    # Plot the proportion of the different models
    prop, ax = plt.subplots(1, 1)
    plt.get_current_fig_manager().window.showMaximized()
    labels = ["plaw", "comp", "band", "sbpl"]
    values = [len(plaw_ampl), len(comp_ampl), len(band_ampl), len(sbpl_ampl)]
    ax.pie(values, labels=labels, autopct=lambda x: int(19.28*x))
    plt.show()

    # Plot the distributions of the models' parameters
    plaw, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.get_current_fig_manager().window.showMaximized()
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
    plaw, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    plt.get_current_fig_manager().window.showMaximized()
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
    band, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    plt.get_current_fig_manager().window.showMaximized()
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
    sbpl, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    plt.get_current_fig_manager().window.showMaximized()
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


class SampleCatalog:
  def __init__(self, datafile=None):
    """
    Instanciates a catalog
    :param datafile: None or string, data to put in the catalog
    """
    self.cat_type = "sampled"
    self.name = []
    self.red = []
    self.dl = []
    self.ep = []
    self.band_low = []
    self.band_high = []
    self.liso = []
    self.eiso = []
    self.thetaj = []
    self.mean_flux = []
    self.t90 = []
    self.fluence = []
    self.lc = []

    # Catalog attributes
    self.length = 0
    if datafile is not None:
      self.fill(datafile)

  def __len__(self):
    """
    Makes use of built-in len function
    """
    return self.length

  def fill(self, datafile):
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
    with open(datafile) as f:
      lines = f.read().split("\n")[3:-1]  # 3 first lines are header
    self.length = len(lines)
    for line in lines:
      data = line.split("|")
      self.name.append(data[0])
      self.t90.append(float(data[1]))
      self.lc.append(data[2])
      self.fluence.append(float(data[3]))
      self.mean_flux.append(float(data[4]))
      self.red.append(float(data[5]))
      self.band_low.append(float(data[6]))
      self.band_high.append(float(data[7]))
      self.ep.append(float(data[8]))
      self.dl.append(float(data[9]))
      self.liso.append(float(data[10]))
      self.eiso.append(float(data[11]))
      self.thetaj.append(float(data[12]))
    self.name = np.array(self.name)
    self.red = np.array(self.red)
    self.dl = np.array(self.dl)
    self.ep = np.array(self.ep)
    self.band_low = np.array(self.band_low)
    self.band_high = np.array(self.band_high)
    self.liso = np.array(self.liso)
    self.eiso = np.array(self.eiso)
    self.thetaj = np.array(self.thetaj)
    self.mean_flux = np.array(self.mean_flux)
    self.t90 = np.array(self.t90)
    self.fluence = np.array(self.fluence)
    self.lc = np.array(self.lc)

  def items(self):
    """
    List all knowns items
    """
    return list(self.__dict__.keys())[3:]
