import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Version 2, Created by Adrien Laviron, updated by Nathan Franel

class Catalog:

  def __init__(self, data=None, sttype=None):
    """
    Instanciates a catalog
    :param data: None or string, data to put in the catalog
    :param sttype: See Catalog.fill
    """
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
        if type(self.sttype[i]) == str and self.sttype[i].startswith('['):
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
    :param sttype: iterable of len 5: first header event (int), event separator (str), first event (int), item separator (str), last event (int) OR list of the sources wanted (list)
    """
    self.data = data
    self.sttype = sttype
    self.formatsttype()
    with open(data) as f:
      d = f.read().split(sttype[1])
    if type(sttype[4]) == int:
      events = d[sttype[2]:sttype[4]]
      if events[-1] == '':
        events = events[:-1]
    elif type(sttype[4]) == list:
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
    """
    # Extracting dec and ra from catalog and transforms decimal degrees into degrees into the right frame
    thetap = [np.sum(np.array(dec.split(" ")).astype(np.float) / [1, 60, 3600]) if len(dec.split("+")) == 2 else np.sum(np.array(dec.split(" ")).astype(np.float) / [1, -60, -3600]) for dec in self.dec]
    thetap = np.deg2rad(np.array(thetap))
    phip = [np.sum(np.array(ra.split(" ")).astype(np.float) / [1, 60, 3600]) if len(ra.split("+")) == 2 else np.sum(np.array(ra.split(" ")).astype(np.float) / [1, -60, -3600]) for ra in self.ra]
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
    """
    ## Containers for the different model parameters
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

    ax1.set(xlabel="Amplitude (photon/cm2/s/keV)", ylabel="Number of GRBs")
    ax2.set(xlabel="Index 1", ylabel="Number of GRBs")
    ax3.set(xlabel="Index 2", ylabel="Number of GRBs")
    ax4.set(xlabel="Break energy (keV)", ylabel="Number of GRBs")
    ax5.set(xlabel="Break scale (keV)", ylabel="Number of GRBs")
    ax6.set(xlabel="Pivot energy (keV)", ylabel="Number of GRBs")

    plt.show()
