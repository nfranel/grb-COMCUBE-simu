# Autor Nathan Franel
# Date 15/03/2023
# Version 1 :
# Creation of the class and modules, mainly based on work from Adrien Laviron

from funcmod import *
from catalogext import Catalog
from scipy.optimize import curve_fit
from inspect import signature
import matplotlib.pyplot as plt
import subprocess
import matplotlib as mpl
import matplotlib.colors as colors
# import numpy as np
# import gzip

class Fit:
  """
  Fit container
  :field f:       function, function fitted to data
  :field x:       np.array, data parameter
  :field y:       np.array, data
  :field popt:    np.array, optimum parameters
  :field pcov:    np.array, covariance matrix
  :field comment: str,      comment on the fit (ex: type, human readable name of function, ...)
  :field q2:      float,    Q^2 value of the fit
  :field nparam:  int,      number of parameters of the function
  """

  def __init__(self, f, x, y, bounds=None, comment=""):
    """
    Instanciates a Fit
    :param f:       function, function fitted to data
    :param x:       np.array, data parameter
    :param y:       np.array, data
    :param comment: str,      comment on the fit (ex: type, human readable name of function, ...)
    :returns:       Correctly instanciated Fit
    """
    self.f = f
    self.x = x
    self.y = y
    if bounds is None:
      self.popt, self.pcov = curve_fit(f, x, y)[:2]
    else:
      self.popt, self.pcov = curve_fit(f, x, y, bounds=bounds)[:2]
    self.comment = comment
    yf = f(x, *self.popt)
    self.q2 = np.sum((y - yf) ** 2)
    self.nparam = len(signature(f).parameters) - 1

  def disp(self):
    """
    Some statistical magic happens here
    """
    if self.comment == "modulation":
      print("\nPolarization analysis:")
      pa = (self.popt[0] + (90 if self.popt[1] < 0 else 0)) % 180
      print("\tModulation        :  {}+-{}".format(abs(self.popt[1]), np.sqrt(self.pcov[1][1])))
      print("\tPolarization angle: ({}+-{}) deg".format(pa, np.sqrt(self.pcov[0][0])))
      print("\tSource flux       :  {}+-{}".format(self.popt[2], np.sqrt(self.pcov[2][2])))
      print("\tFit goodness      : {}\n".format(self.q2 / (len(self.x) - self.nparam)))
    elif self.comment == "constant":
      print("\nConstant fit:")
      print("\tFit goodness      : {}\n".format(self.q2 / (len(self.x) - self.nparam)))
    else:
      print("\n{}: Unknown fit type - displaying raw results".format(self.comment))
      print(self.popt)
      print(np.sqrt(np.diag(self.pcov)))
      print(self.pcov)
      print("Q^2 / ndof: {}\n".format(self.q2 / (len(self.x) - self.nparam)))


class Polarigram(list):
  """
  Polarigram is a container for usual data regarding polarization measurements and usefull functions for data analysis
  :self: Heritates from list, list of angles in deg
  :field bins:  iterable, bins with which histograms are calculated in deg
  :field theta: float,    source position's polar angle theta in rad (in sat frame)
  :field phi:   float,    source position's azimuthal angle theta in rad (in sat frame)
  :field PA:    float,    polarization angle of the source as defined in cosima's source file for "RelativeY" polarization definition in deg (in sat frame)
  :field fits:  list,     list of Fit instances applied to the Polarigram
  """

  def __init__(self, data, theta, phi, pa, bins=np.arange(-180, 181, 18), corr=False, ergcut=None):
    """
    Correctly instanciates a Polarigram, filled with data
    :param data:  list or str, list of angles in deg or name of .tra file
    :param theta: float,       source position's polar angle theta in deg (will be converted to rad)
    :param phi:   float,       source position's azimuthal angle theta in deg (will be converted to rad)
    :param pa:    float,       polarization angle of the source as defined in cosima's source file for "RelativeY" polarization definition in deg
    :param bins:  iterable,    bins with which histograms are calculated in deg, default = -180° to 180° with 18° steps
    :param corr:  bool,        wether to correct for the source sky position and cosima's "RelativeY" polarization definition or not, default=False
    :param ergcut: couple (Emin, Emax) or None, energy range in which to perform polarization analysis, default=None(=all data available)
    :returns:     Polarigram instance
    """
    self.bins = bins
    self.theta = theta * np.pi / 180
    self.phi = phi * np.pi / 180
    self.expected_pa = pa
    self.fits = []
    self.polar_angles = []
    self.azim_angle_corrected = False
    if type(data) == list:
      list.__init__(self, data)
    elif type(data) == str:
      angle_lists = analyzetra(data, self.theta, self.phi, self.expected_pa, corr=corr, ergcut=ergcut)
      list.__init__(self, angle_lists[0])
      self.polar_angles = angle_lists[1]
      self.behave()
      self.azim_angle_corrected = corr

  def cor(self):
    """
    Calculates the angle to correct for the source sky position and cosima's "RelativeY" polarization definition
    :returns: float, angle ni deg
    Warning : That's actually minus the correction angle (so that the correction uses a + instead of a - ...)
    """
    return np.arctan(np.cos(self.theta) * np.tan(self.phi)) * 180 / np.pi + self.expected_pa

  def behave(self, width=360):
    """
    Make angles be between the beginning of the first bin and the beginning of the first bin plus the width parameter
    Calculi are made in-place
    :param width: float, width of the polarigram in deg, default=360, SHOULD BE 360
    """
    for i in range(len(self)):
      self[i] = self[i] % width + self.bins[0]

  def corr(self):
    """
    Corrects the angles from the source sky position and cosima's "RelativeY" polarization definition
    """
    if self.azim_angle_corrected:
      print(" Impossible to correct the azimuthal compton scattering angles, the correction has already been made")
    else:
      cor = self.cor()
      for i in range(len(self)):
        self[i] += cor
      self.behave()
      self.azim_angle_corrected = True

  def anticorr(self):
    """
    Undo the corr operation
    """
    if self.azim_angle_corrected:
      cor = self.cor()
      for i in range(len(self)):
        self[i] -= cor
      self.behave()
      self.azim_angle_corrected = False
    else:
      print(" Impossible to undo the correction of the azimuthal compton scattering angles : no correction were made")

  def fit(self, unpoldata=None, fit_bounds=None):
    """
    Fits first a modulation function and then a constant function to the polarigram
    :param unpoldata: Polarigram or None, Polarigram used for geometry correction, default=None
    """
    var_x = .5 * (self.bins[1:] + self.bins[:-1])
    binw = self.bins[1:] - self.bins[:-1]
    p = np.histogram(self, self.bins)[0] / binw
    if unpoldata is not None:
      unpol = np.histogram(unpoldata, self.bins)[0] / binw
      if 0. in unpol:
        print("Unpolarized data do not allow a fit : a bin is empty")
        self.fits.append(None)
      else:
        #p = p / unpol * np.mean(unpol)
        p /= unpol
        self.fits.append(Fit(modulation_func, var_x, p, bounds=fit_bounds, comment="modulation"))
        self.fits.append(Fit(lambda x, a: a * x / x, var_x, p, comment="constant"))
    else:
      self.fits.append(Fit(modulation_func, var_x, p, bounds=fit_bounds, comment="modulation"))
      self.fits.append(Fit(lambda x, a: a * x / x, var_x, p, comment="constant"))


  def clf(self):
    """
    Clears the fit list
    """
    self.fits = []

  def show(self, unpoldata=None, fit=False, disp=True, plot=True, plotfit=None, show=True, ret=True):
    """
    Plots and show a polarigram, and also does all the statistical analysis (indev)
    :param unpoldata: Polarigram,  unpolarised data to correct for geometrical effects
    :param fit:       bool,        whether or not to fit the polarigram,                  default=False
    :param disp:      bool,        whether or not to print fit results,                   default=True
    :param plot:      bool,        whether or not to plot the polarigram and fit results, default=True
    :param plotfit:   list of int, which fit(s) to plot (None is none),                   default=[-2]
    :param show:      bool,        whether or not to show fit results,                    default=True
    :param ret:       bool,        whether or not to return the result,                   default=True
    :returns:         couple of np.ndarray or None
    """
    if plotfit is None:
      plotfit = [-2]
    x = .5 * (self.bins[1:] + self.bins[:-1])
    binw = self.bins[1:] - self.bins[:-1]
    p = np.histogram(self, self.bins)[0] / binw
    ylabel = "Number of counts (per degree)"
    if unpoldata is not None:
      unpol = np.histogram(unpoldata, self.bins)[0] / binw
      #p = p / unpol * np.mean(unpol)
      p /= unpol
      ylabel = "Corrected number of count"
    print(p, unpol)
    if fit:
      self.fit(unpoldata)
    if plot:
      plt.step(x, p, "g", where="mid")
      if plotfit is not None:
        xfit = np.arange(self.bins[0] - binw[0], self.bins[-1] + binw[-1], 1)
        for i in plotfit:
          if disp:
            self.fits[i].disp()
          plt.plot(xfit, self.fits[i].f(xfit, *self.fits[i].popt), "r--")
      plt.xlabel("Azimuthal scatter angle (degree)")
      plt.ylabel(ylabel)
      if show:
        plt.show()
    if ret:
      return p


class BkgContainer:
  """
  Class containing the information for 1 background file
  """

  def __init__(self, datafile, sim_duration, opt_items=None, opt_analysis=None, ergcut=None):
    """
    -data_list : list of 1 or 2 files (pol or pol+unpol) from which extract the data
    """
    if opt_analysis is None:
      opt_analysis = [treatCE]
    if opt_items is None:
      opt_items = ["CE"]
    self.triggers = 0
    self.calor = 0
    self.dsssd = 0
    self.side = 0
    self.single = 0
    self.compton = 0
    self.CE = 0
    self.CE_sum = 0
    self.polar_from_energy = []
    self.cr = 0

    self.dec, self.ra = datafile.split("_")[-2:]
    self.dec = float(self.dec)
    self.ra = float(self.ra.split(".inc")[0])

    if datafile.endswith(".tra"):
      with open(datafile) as f:
        lines = f.read().split("\n")
    elif datafile.endswith(".tra.gz"):
      with gzip.open(datafile, "rt") as f:
        lines = [e[:-1] for e in f]
    else:
      raise TypeError(f"{datafile} has unknown extension (known: .tra ou .tra.gz)")

    for item in opt_items:
      setattr(self, item, [])
    for i, line in enumerate(lines):
      if line.startswith("  Number of triggered events:"):
        self.triggers = int(line.split(" ")[-1])
      elif line.startswith("      TCeBr3Det:"):
        self.calor = int(line.split(" ")[-1])
      elif line.startswith("      TSiSDDet:"):
        self.dsssd = int(line.split(" ")[-1])
      elif line.startswith("      TSideDetector:"):
        self.side = int(line.split(" ")[-1])
      elif line.startswith("       Single-site"):
        self.single = int(line[58:66])
      elif line.startswith("       Compton"):
        self.compton = int(line[58:66])
      for item in opt_items:
        if line.startswith(item):
          getattr(self, item).append(line.split(" ")[1:] if len(line.split(" ")[1:]) > 1 else line.split(" ")[1])
    for item, f in zip(opt_items, opt_analysis):
      if f is not None:
        setattr(self, item, list(map(f, getattr(self, item))))
    if "CE" in opt_items:
      self.CE = np.array(self.CE)
      self.CE_sum = np.sum(self.CE, axis=1)
      # print(1/self.CE[:, 1])
      # print((1 - m_elec * c_light**2 / charge_elem / 1000 * (1/self.CE[:, 1] - 1/(self.CE_sum))))
      # print(np.rad2deg(np.arccos(1 - m_elec * c_light**2 / charge_elem / 1000 * (1/self.CE[:, 1] - 1/(self.CE_sum)))))
    self.compton = np.sum(inwindow(self.CE_sum, ergcut))
    self.cr = self.compton / sim_duration


class FormatedData:
  """
  Class containing the data for 1 GRB, for 1 sim, and 1 satellite
  """

  def __init__(self, data_list, sat_info, num_sat, sim_duration, opt_items=None, opt_analysis=None, corr=False,
               ergcut=None):
    """
    -data_list : list of 1 or 2 files (pol or pol+unpol) from which extract the data
    """
    if opt_analysis is None:
      opt_analysis = [treatCE]
    if opt_items is None:
      opt_items = ["CE"]
    self.num_sat = num_sat
    if sat_info is None:
      self.b_rate = 0
    else:
      self.b_rate = sat_info[-1]
    self.triggers = 0
    self.calor = 0
    self.dsssd = 0
    self.side = 0
    self.single = 0
    self.compton = 0
    self.CE = 0
    self.CE_sum = 0
    self.polar_from_energy = []
    self.cr = 0
    self.pol = []
    self.unpol = []
    self.n_sat_detect = 1
    self.s_eff = None
    self.mu100 = None
    self.pa = None
    self.fit_cr = None
    self.mdp = None
    self.snr = None
    self.pa_err = None
    self.mu100_err = None
    self.fit_cr_err = None
    self.fit_goodness = None
    self.dec_world_frame = None
    self.ra_world_frame = None
    self.dec_sat_frame = None
    self.ra_sat_frame = None
    self.expected_pa = None

    if len(data_list) == 0:
      for item in opt_items:
        setattr(self, item, [])
      self.n_sat_detect = 0
    else:
      # Photometric analysis
      dat = data_list[0]
      if dat.endswith(".tra"):
        with open(dat) as f:
          lines = f.read().split("\n")
      elif dat.endswith(".tra.gz"):
        with gzip.open(dat, "rt") as f:
          lines = [e[:-1] for e in f]
      else:
        raise TypeError(f"{dat} has unknown extension (known: .tra ou .tra.gz)")

      for item in opt_items:
        setattr(self, item, [])
      for i, line in enumerate(lines):
        if line.startswith("  Number of triggered events:"):
          self.triggers = int(line.split(" ")[-1])
        elif line.startswith("      TCeBr3Det:"):
          self.calor = int(line.split(" ")[-1])
        elif line.startswith("      TSiSDDet:"):
          self.dsssd = int(line.split(" ")[-1])
        elif line.startswith("      TSideDetector:"):
          self.side = int(line.split(" ")[-1])
        elif line.startswith("       Single-site"):
          self.single = int(line[58:66])
        elif line.startswith("       Compton"):
          self.compton = int(line[58:66])
        for item in opt_items:
          if line.startswith(item):
            getattr(self, item).append(line.split(" ")[1:] if len(line.split(" ")[1:]) > 1 else line.split(" ")[1])
      for item, f in zip(opt_items, opt_analysis):
        if f is not None:
          setattr(self, item, list(map(f, getattr(self, item))))
      if "CE" in opt_items:
        self.CE = np.array(self.CE)
        self.CE_sum = np.sum(self.CE, axis=1)
        # print(1 / self.CE[:, 1])
        # print((1 - m_elec * c_light ** 2 / charge_elem / 1000 * (1 / self.CE[:, 1] - 1 / (self.CE_sum))))
        # print(np.rad2deg(np.arccos(1 - m_elec * c_light**2 / charge_elem / 1000 * (1/self.CE[:, 1] - 1/(self.CE_sum)))))
      self.compton = np.sum(inwindow(self.CE_sum, ergcut))
      self.cr = self.compton / sim_duration

      self.dec_world_frame, self.ra_world_frame = fname2decra(data_list[0])
      self.dec_sat_frame, self.ra_sat_frame, self.expected_pa = decra2tpPA(self.dec_world_frame, self.ra_world_frame,
                                                                           sat_info[:3])
      if len(data_list) == 2:
        # Polarization analysis
        self.pol = Polarigram(data_list[0], self.dec_sat_frame, self.ra_sat_frame, self.expected_pa, corr=corr,
                              ergcut=ergcut)
        self.unpol = Polarigram(data_list[1], self.dec_sat_frame, self.ra_sat_frame, self.expected_pa, corr=corr,
                                ergcut=ergcut)
      elif len(data_list) == 1:
        self.pol = Polarigram(data_list[0], self.dec_sat_frame, self.ra_sat_frame, self.expected_pa, corr=corr,
                              ergcut=ergcut)
        self.unpol = None

  def get_keys(self):
    print("======================================================================")
    print("    Attributes")
    print(" Number of the satellite (or satellites if constellation):                .num_sat")
    print(" Number of triggers in the detectors                                      .triggers")
    print(" Number of triggers in the colorimeters                                   .calor")
    print(" Number of triggers in the dsssds                                         .dsssd")
    print(" Number of triggers in the side detectors                                 .side")
    print(" Number of single hits                                                    .single")
    print(" Number of compton events recorded                                        .compton")
    print(" Number of compton events recorded (from the field CE)                    .CE")
    print(" Compton events count rate of the source for the satellite/constellation  .cr")
    print(" List of the azimuthal compton angles for all Compton events )pol data)   .pol")
    print(" List of the azimuthal compton angles for all Compton events (unpol data) .unpol")
    print(" Number of sat detecting the source, != 1 only for constellations         .n_sat_detect")
    print(" mu100 for the satellite/constellation                                    .mu100")
    print(" Polarization angle obtained from the polarigram                          .pa")
    print(" Compton events count rate of the source from the fit                     .fit_cr")
    print(" Minimum detectable polarization calculated with Compton events           .mdp")
    print(" Signal to noise ratio calculated from a choosen field (ex :CE)           .snr")
    print(" Declination of the source in the world frame                             .dec_world_frame")
    print(" Right ascention of the source in the world frame                         .ra_world_frame")
    print(" Declination of the source in the satelitte(s) frame                      .dec_sat_frame")
    print(" Right ascention of the source in the satelitte(s) frame                  .ra_sat_frame")
    print(" Expected polarization angle for the satelittes(s) (set by sim)           .expected_pa")
    print(" Polarization angle error from the fit                                    .pa_err")
    print(" mu100 error from the fit                                                 .mu100_err")
    print(" Count rate error from the fit                                            .fit_cr_err")
    print(" Fit goodness                                                             .fit_goodness")

    print("======================================================================")
    print("    Methods")
    print("======================================================================")

  def analyze(self, source_duration, source_fluence, source_with_bkg=True, fit_bounds=None):
    """
    Proceeds to the data analysis to get mu100, pa, compton cr, mdp and snr
    mdp has physical significance between 0 and 1
    """
    if source_fluence is None:
      self.s_eff = None
    else:
      self.s_eff = self.compton / source_fluence
    if self.unpol is not None:
      self.pol.fit(self.unpol, fit_bounds=fit_bounds)
      # self.pol.fit(self.unpol, fit_bounds=([-np.inf, -np.inf, (len(self.pol)-1)/100], [np.inf, np.inf, (len(self.pol)+1)/100]))
      if self.pol.fits[0] is not None:
        self.pa, self.mu100, self.fit_cr = self.pol.fits[-2].popt
        if self.mu100 < 0:
          self.pa = (self.pa + 90) % 180
          self.mu100 = - self.mu100
        else:
          self.pa = self.pa % 180
        self.pa_err = np.sqrt(self.pol.fits[-2].pcov[0][0])
        self.mu100_err = np.sqrt(self.pol.fits[-2].pcov[1][1])
        self.fit_cr_err = np.sqrt(self.pol.fits[-2].pcov[2][2])
        self.fit_goodness = self.pol.fits[-2].q2 / (len(self.pol.fits[-2].x) - self.pol.fits[-2].nparam)

        if source_with_bkg:
          print("MDP calculation may not work if source is simulated with the background")
          self.mdp = MDP((self.cr - self.b_rate) * source_duration, self.b_rate * source_duration, self.mu100)
          self.snr = SNR(self.cr * source_duration, self.b_rate * source_duration)
        else:
          self.mdp = MDP(self.cr * source_duration, self.b_rate * source_duration, self.mu100)
          self.snr = SNR((self.cr + self.b_rate) * source_duration, self.b_rate * source_duration)


class AllSatData(list):
  """
  Class containing all the data for 1 simulation of 1 GRB (or other source) for a full set of trafiles
  """

  def __init__(self, source_prefix, num_sim, pol_analysis, sat_info, sim_duration, options):
    temp_list = []
    self.n_sat_det = 0
    self.n_sat = len(sat_info)
    self.dec_world_frame = None
    self.ra_world_frame = None
    self.pol_analysis = True
    loading_count = 0
    for num_sat in range(self.n_sat):
      flist = subprocess.getoutput("ls {}_sat{}_{:04d}_*".format(source_prefix, num_sat, num_sim)).split("\n")
      if len(flist) == 2:
        temp_list.append(FormatedData(flist, sat_info[num_sat], num_sat, sim_duration, *options))
        self.n_sat_det += 1
        loading_count += 2
      elif len(flist) == 1:
        if flist[0].startswith("ls: cannot access"):
          temp_list.append(None)
        elif pol_analysis:
          temp_list.append(FormatedData(flist, sat_info[num_sat], sim_duration, num_sat, *options))
          self.n_sat_det += 1
          self.pol_analysis = False
          loading_count += 1
          raise Warning(f'Polarization analysis is expected but the wrong number of trafile has been found, no polarization data were extracted : {flist}')
        else:
          temp_list.append(FormatedData(flist, sat_info[num_sat], sim_duration, num_sat, *options))
          self.n_sat_det += 1
          self.pol_analysis = False
          loading_count += 1
      if not flist[0].startswith("ls: cannot access") and self.dec_world_frame is None:
        self.dec_world_frame, self.ra_world_frame = fname2decra(flist[0])
    print(f"Total of {loading_count} files loaded for simulation {num_sim}")
    list.__init__(self, temp_list)
    self.const_data = None

    def get_keys(self):
      print("======================================================================")
      print("    Attributes")
      print(" Number of satellites detecting the source :           .n_sat_det")
      print(" Number of satellites in the simulation :              .n_sat")
      print(" Declination of the source (world frame) :             .dec_world_frame")
      print(" Right ascention of the source (world frame) :         .ra_world_frame")
      print(" Whether or not a polarization analysis is possible    .pol_analysis")
      print(" ===== Attribute that needs to be handled + 2 cases (full FoV or full sky)")
      print(" Extracted data but for a given set of satellites      .const_data")
      print("======================================================================")
      print("    Methods")
      print("======================================================================")

  def analyze(self, source_duration, source_fluence, source_with_bkg=True, fit_bounds=None, const_analysis=True):
    """
    Proceed to the analysis of polarigrams for all satellites and constellation (unless specified)
    """
    for sat in self:
      if sat is not None:
        sat.analyze(source_duration, source_fluence, source_with_bkg=source_with_bkg, fit_bounds=fit_bounds)
    if self.const_data is not None and const_analysis:
      self.const_data.analyze(source_duration, source_fluence, source_with_bkg=source_with_bkg, fit_bounds=fit_bounds)
    else:
      print("Constellation not set : please use make_const method if you want to analyze the constellation's results")

  def make_const(self, options, const=None):
    if const is None:
      const = np.array(range(self.n_sat))
    considered_sat = const[np.where(np.array(self) == None, False, True)]
    self.const_data = FormatedData([], None, None, None, *options)
    for item in self.const_data.__dict__.keys():
      if item.startswith("dec") or item.startswith("ra"):
        setattr(self.const_data, item, [])
        for num_sat in considered_sat:
          getattr(self.const_data, item).append(getattr(self[num_sat], item))
      elif item == "pol":
        temp_list = []
        for num_sat in considered_sat:
          temp_list = temp_list + getattr(self[num_sat], item)
        setattr(self.const_data, item, Polarigram(temp_list, 0, 0, 0, corr=options[2], ergcut=options[3]))
      elif item == "unpol":
        if self.pol_analysis:
          temp_list = []
          for num_sat in considered_sat:
            temp_list = temp_list + getattr(self[num_sat], item)
          setattr(self.const_data, item, Polarigram(temp_list, 0, 0, 0, corr=options[2], ergcut=options[3]))
        else:
          setattr(self.const_data, item, None)
      elif item == "num_sat" or item == "expected_pa":
        setattr(self.const_data, item, [])
        for num_sat in considered_sat:
          getattr(self.const_data, item).append(getattr(self[num_sat], item))
      elif item == "CE_sum":
        setattr(self.const_data, item, np.array([]))
        for num_sat in considered_sat:
          setattr(self.const_data, item, np.concatenate((getattr(self.const_data, item), getattr(self[num_sat], item))))
      elif item == "CE":
        setattr(self.const_data, item, np.array([[0, 0]]))
        for num_sat in considered_sat:
          print(getattr(self.const_data, item), getattr(self[num_sat], item))
          setattr(self.const_data, item, np.concatenate((getattr(self.const_data, item), getattr(self[num_sat], item))))
        setattr(self.const_data, item, getattr(self.const_data, item)[1:])
      else:
        if item not in ['s_eff', 'mu100', 'pa', 'fit_cr', 'mdp', 'snr', 'pa_err', 'mu100_err', 'fit_cr_err', 'fit_goodness']:
          for num_sat in considered_sat:
            setattr(self.const_data, item, getattr(self.const_data, item) + getattr(self[num_sat], item))


class AllSimData(list):
  """
  Class containing all the data for 1 GRB (or other source) for a full set of trafiles
  """

  def __init__(self, sim_prefix, source_ite, cat_data, mode, n_sim, sat_info, pol_analysis, sim_duration, options):
    temp_list = []
    self.n_sim_det = 0
    if type(cat_data) == list:
      self.source_name = cat_data[0][source_ite]
      self.source_duration = float(cat_data[1][source_ite])
      self.p_flux = None
      self.best_fit_model = None
    else:
      self.source_name = cat_data.name[source_ite]
      self.source_duration = float(cat_data.t90[source_ite])
      self.best_fit_model = getattr(cat_data, f"{mode}_best_fitting_model")[source_ite].rstrip()
      self.p_flux = float(getattr(cat_data, f"{self.best_fit_model}_phtflux")[source_ite])
    self.proba_detec_fov = None
    self.proba_compton_image_fov = None
    self.const_proba_detec_fov = None
    self.const_proba_compton_image_fov = None
    self.proba_detec_sky = None
    self.proba_compton_image_sky = None
    self.const_proba_detec_sky = None
    self.const_proba_compton_image_sky = None
    source_prefix = f"{sim_prefix}_{self.source_name}"
    flist = subprocess.getoutput("ls {}_*".format(source_prefix)).split("\n")
    if flist[0].startswith("ls: cannot access"):
      print(f"No file to be loaded for source {self.source_name}")
    else:
      print(f"{len(flist)} files to be loaded for source {self.source_name} : ")
    for num_sim in range(n_sim):
      flist = subprocess.getoutput("ls {}_*_{:04d}_*".format(source_prefix, num_sim)).split("\n")
      if len(flist) >= 2:
        temp_list.append(AllSatData(source_prefix, num_sim, pol_analysis, sat_info, sim_duration, options))
        self.n_sim_det += 1
      elif len(flist) == 1:
        if flist[0].startswith("ls: cannot access"):
          temp_list.append(None)
        else:
          temp_list.append(AllSatData(source_prefix, num_sim, pol_analysis, sat_info, sim_duration, options))
          self.n_sim_det += 1
    list.__init__(self, temp_list)

  def get_keys(self):
    print("======================================================================")
    print("    Attributes")
    print(" Number of simulations detected by the constellation : .n_sim_det")
    print(" Name of the source :                                  .source_name")
    print(" Duration of the source (t90 for GRB) :                .source_duration")
    print(" ===== Attribute that needs to be handled + 2 cases (full FoV or full sky)")
    print(" Probability of having a detection                     .proba_detec")
    print(" Probability of being able to construct an image :     .proba_compton_image")
    print("======================================================================")
    print("    Methods")
    print("======================================================================")

  def set_probabilities(self, n_sat, snr_min=5, n_image_min=50):
    """
    Calculates detection probability and probability of having a correct compton image
    """
    temp_proba_detec = np.zeros(n_sat)
    temp_proba_compton_image = np.zeros(n_sat)
    temp_const_proba_detec = 0
    temp_const_proba_compton_image = 0
    for sim in self:
      if sim is not None:
        for sat_ite, sat in enumerate(sim):
          if sat is not None:
            if sat.snr >= snr_min:
              temp_proba_detec[sat_ite] += 1
            if sat.compton >= n_image_min:
              temp_proba_compton_image[sat_ite] += 1
        if sim.const_data.snr >= snr_min:
          temp_const_proba_detec += 1
        if sim.const_data.compton >= n_image_min:
          temp_const_proba_compton_image += 1

    self.proba_detec_fov = temp_proba_detec / self.n_sim_det
    self.proba_detec_sky = temp_proba_detec / len(self)
    self.proba_compton_image_fov = temp_proba_compton_image / self.n_sim_det
    self.proba_compton_image_sky = temp_proba_compton_image / len(self)
    self.const_proba_detec_fov = temp_const_proba_detec / self.n_sim_det
    self.const_proba_detec_sky = temp_const_proba_detec / len(self)
    self.const_proba_compton_image_fov = temp_const_proba_compton_image / self.n_sim_det
    self.const_proba_compton_image_sky = temp_const_proba_compton_image / len(self)



class AllSourceData:
  """
  Class containing all the data for a full set of trafiles
  """

  def __init__(self, bkg_prefix, param_file, erg_cut=(100, 460)):
    """
    Initiate the class AllData using
    - bkg_prefix : str, the prefix for background files
    - param_file : str, the path to the parameter file (.par) used for the simulation
    - erg_cut    : tuple of len 2, the lower and uppuer bounds of the energy window considered

    Extract from the parameters and from the files the information needed for the analysis
    Makes some basic tests on filenames to reduce the risk of unseen errors

    FAIRE OPTION AVEC PARAM FILE QUI EST UNE LISTE (evite d'avoir a faire un .par si on veut juste étudier 1 seule simu)
    Ceci est le cas de base pour les simulations, le modifier pour permettre des sources moins habituelles
    """

    self.bkg_prefix = bkg_prefix
    self.param_file = param_file
    self.erg_cut = erg_cut

    #### A CODER AUTREMENT AVEC LECTURE D'UN FICHIER DE PARAMETRE POUR LES BACKGROUNDS
    self.bkg_sim_duration = 3600
    opt_items = ["CE", "PE"]
    opt_analysis = [treatCE, None]
    # opt_items = None
    # opt_analysis = None
    corr = True
    self.options = [opt_items, opt_analysis, corr, self.erg_cut]
    self.pol_data = False
    self.sat_info = []
    with open(self.param_file) as f:
      lines = f.read().split("\n")
    for line in lines:
      if line.startswith("@prefix"):
        self.sim_prefix = line.split(" ")[1]
      elif line.startswith("@cosimasourcefile"):
        self.source_file = line.split(" ")[1]
      elif line.startswith("@revancfgfile"):
        self.revan_file = line.split(" ")[1]
      elif line.startswith("@geometry"):
        self.geometry = line.split(" ")[1]
      elif line.startswith("@type"):
        self.sim_type = line.split(" ")[1]
      elif line.startswith("@instrument"):
        self.instrument = line.split(" ")[1]
      elif line.startswith("@mode"):
        self.mode = line.split(" ")[1]
      elif line.startswith("@sttype"):
        self.sttype = line.split(" ")[1:]
      elif line.startswith("@file"):
        self.cat_file = line.split(" ")[1]
      elif line.startswith("@spectrafilepath"):
        self.spectra_path = line.split(" ")[1]
      elif line.startswith("@simulationsperevent"):
        self.n_sim = int(line.split(" ")[1])
      elif line.startswith("@position"):
        self.position_allowed_sim = np.array(line.split(" ")[1:], dtype=float)
      elif line.startswith("@satellite"):
        temp = [float(e) for e in line.split(" ")[1:]]
        if len(temp) == 3:  # satellite pointing
          dat = [temp[0], temp[1], horizonAngle(temp[2])]
        else:  # satellite orbital parameters
          inclination, ohm, omega = map(np.deg2rad, temp[:3])
          thetasat = np.arccos(np.sin(inclination) * np.sin(omega))  # rad
          phisat = np.arctan2(
            (np.cos(omega) * np.sin(ohm) + np.sin(omega) * np.cos(inclination) * np.cos(ohm)),
            (np.cos(omega) * np.cos(ohm) - np.sin(omega) * np.cos(inclination) * np.sin(ohm)))  # rad
          dat = [thetasat, phisat, horizonAngle(temp[3])]
        self.sat_info.append(dat)
    self.n_sat = len(self.sat_info)

    with open(self.source_file) as f:
      lines = f.read().split("\n")
    self.source_with_bkg = False
    if len(lines) > 50:
      self.source_with_bkg = True
    duration_source = []
    sim_name = ""
    source_name = ""
    for line in lines:
      if line.startswith("Geometry"):
        if line.split("Geometry")[1].strip() != self.geometry:
          raise Warning("Different geometry files in parfile dans sourcefile")
      elif line.startswith("Run"):
        sim_name = line.split(" ")[1]
      elif line.startswith(f"{sim_name}.Time"):
        duration_source.append(float(line.split("Time")[1].strip()))
      elif line.startswith(f"{sim_name}.Source"):
        source_name = line.split(" ")[1]
      elif line.startswith(f"{source_name}.Polarization") and not self.pol_data:
        self.pol_data = True
      elif line.startswith(f"{source_name}.Polarization") and self.pol_data:
        raise Warning("Sourcefile contains 2 polarized sources")
    if (np.array(duration_source) / duration_source[0] != np.ones((len(duration_source)))).all():
      raise Warning("Simulations in sourcefile seem to have different duration")
    self.sim_duration = duration_source[0]

    self.bkgdata = []
    flist = subprocess.getoutput("ls {}_*".format(bkg_prefix)).split("\n")
    for bkgfile in flist:
      self.bkgdata.append(BkgContainer(bkgfile, self.bkg_sim_duration, opt_items=opt_items, opt_analysis=opt_analysis,
                                       ergcut=self.erg_cut))

    for sat_ite in range(len(self.sat_info)):
      self.sat_info[sat_ite].append(closest_bkg_rate(self.sat_info[sat_ite][0], self.bkgdata))

    if self.cat_file=="None":
      cat_data = self.extract_sources(self.sim_prefix)
      self.namelist = cat_data[0]
      self.n_source = len(self.namelist)
      self.fluence = None
    else:
      cat_data = Catalog(self.cat_file, self.sttype)
      self.namelist = cat_data.name
      self.n_source = len(self.namelist)
      self.fluence = [calc_fluence(cat_data, source_index, erg_cut)*self.sim_duration for source_index in range(self.n_source)]
    self.s_eff = None
    self.alldata = [
      AllSimData(self.sim_prefix, source_ite, cat_data, self.mode, self.n_sim, self.sat_info, self.pol_data, self.sim_duration,
                 self.options) for source_ite in range(self.n_source)]

    self.cat_duration = 10
    self.com_duty = 1
    self.gbm_duty = 0.6
    ### Implementer une maniere automatique de calculer le fov de comcube
    self.com_fov = 1
    self.gbm_fov = (1 - np.cos(np.deg2rad(horizonAngle(565)))) / 2
    self.weights = 1/self.n_sim/self.cat_duration*self.com_duty/self.gbm_duty*self.com_fov/self.gbm_fov

  def get_keys(self):
    print("======================================================================")
    print("    Files and paths")
    print(" background files prefix :            .bkg_prefix")
    print(" Parameter file used for simulation : .param_file")
    print(" Simulated data prefix :              .sim_prefix")
    print(" Source file path :                   .source_file")
    print(" Revan cfg file path :                .revan_file")
    print(" Geometry file path :                 .geometry")
    print(" Catalog file path :                  .cat_file")
    print(" Path of spectra :                    .spectra_path")
    print("======================================================================")
    print("    Simulation parameters")
    print(" Type of simulation from parfile :         .sim_type")  # Might be usefull to handle different types of sim
    print(" Instrument fiel from parfile    :         .instrument")
    print(" Mode used to handle catalog information : .mode")
    print(" Formated str to extract catalog sources : .sttype")  # Might put in an other field ?
    print(" Area of the sky allowed for simulations : .position_allowed_sim")
    print("======================================================================")
    print("    Data analysis options")
    print(" Energy window considered for the analysis :           .erg_cut")
    print(" Data extraction options :                             .options")
    print("   [opt_items, opt_analysis, corr, erg_cut]")
    print("    opt_items : to get another fiels from trafiles")
    print("    opt_analysis : to handle the new field with a specific function")
    print("    corr : to correct the polarization angle")
    print(" Whether or not polarized simulations were done :       .pol_data")
    print(" Whether or not bkg simulated with the source :         .source_with_bkg")
    print("======================================================================")
    print("    Data and simulation information")
    print(" Information on satellites' position :   .sat_info")
    print(" Number of satellites :                  .n_sat")
    print(" Number of simulation performed :        .n_sim")
    print(" Duration of simulations :               .sim_duration")
    print(" List of source names :                  .namelist")
    print(" Number of sources simulated :           .n_source")
    print(" Data extracted from simulation files :  .alldata")
    print("======================================================================")
    print("    Methods")
    print("======================================================================")

  def extract_sources(self, prefix, duration=None):
    """

    """
    if duration is None:
      duration = self.sim_duration
    flist = subprocess.getoutput("ls {}_*".format(prefix)).split("\n")
    source_names = []
    if len(flist) >= 1 and not flist[0].startswith("ls: cannot access"):
      temp_list = []
      for file in flist:
        temp_list.append(file.split("_")[1])
      source_names = list(set(temp_list))
    return [source_names, [duration]*len(source_names)]

  def azi_angle_corr(self):
    """

    """
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            for sat in sim:
              if sat is not None:
                sat.pol.corr()
                if sat.unpol is not None:
                  sat.unpol.corr()

  def azi_angle_anticorr(self):
    """

    """
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            for sat in sim:
              if sat is not None:
                sat.pol.anticorr()
                if sat.unpol is not None:
                  sat.unpol.anticorr()

  def analyze(self, fit_bounds=None, const_analysis=True):
    """
    Proceed to the analysis of polarigrams for all satellites and constellation (unless specified) for all data
    """
    for source_ite, source in enumerate(self.alldata):
      if source is not None:
        for sim in source:
          if sim is not None:
            if self.fluence is None:
              sim.analyze(source.source_duration, self.fluence, source_with_bkg=self.source_with_bkg, fit_bounds=fit_bounds, const_analysis=const_analysis)
            else:
              sim.analyze(source.source_duration, self.fluence[source_ite], source_with_bkg=self.source_with_bkg, fit_bounds=fit_bounds, const_analysis=const_analysis)
        source.set_probabilities(n_sat=self.n_sat, snr_min=5, n_image_min=50)

  def make_const(self, const=None):
    """
    This function is used to combine results from different satellites
    Results are then stored in the key const_data
    """
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            sim.make_const(self.options, const=const)

  def effective_area(self, sat=0):
    """
    sat is the number of the satellite considered
    This method is supposed to be working with a set of 40 satellites with a lot of simulations
    The results obtained with this method are meaningful only is there is no background simulated
    """
    if self.source_with_bkg:
      raise Warning("The source has been simulated with a background, the calculation has not been done as this would lead to biased results")
    else:
      list_dec = []
      list_s_eff = []
      for source in self.alldata:
        if source is not None:
          temp_dec = []
          temp_s_eff = []
          for num_sim, sim in enumerate(source):
            if sim is not None:
              if sim[sat] is None:
                print(f"The satellite {sat} selected didn't detect the source '{source.source_name}' for the simulation number {num_sim}.")
              else:
                temp_dec.append(sim[sat].dec_sat_frame)
                temp_s_eff.append(sim[sat].s_eff)
          list_dec.append(temp_dec)
          list_s_eff.append(temp_s_eff)

      figure, ax = plt.subplots(2, 2, figsize=(16, 12))
      figure.suptitle("Effective area as a function of detection angle")
      for graph in range(4):
        for ite in range(graph * 10, min(graph * 10 + 10, len(list_dec))):
          ax[int(graph / 2)][graph % 2].scatter(list_dec[ite], list_s_eff[ite],
                                                label=f"Fluence : {np.around(self.fluence[ite], decimals=1)} ph/cm²")
        ax[int(graph / 2)][graph % 2].set(xlabel="GRB zenith angle (rad)",
                                          ylabel="Effective area (cm²)")  # , yscale="linear")
        ax[int(graph / 2)][graph % 2].legend()
      plt.show()

  def viewing_angle_study(self):
    """

    """
    pass

  def fov_const(self, num_val=500, mode="polarization", show=True, save=False):
    """
    Plots a map of the sensibility (s_eff) over the sky
    Mode is the mode used to obtain the sensibility :
      Polarization gives the sensibility to polarization
      Spectrometry gives the sensibility to spectrometry (capacity of detection)
    """
    phi_world = np.linspace(0, 2 * np.pi, num_val)
    # theta will be converted in sat coord with decra2tp, which takes dec in world coord with 0 being north pole and 180 the south pole !
    theta_world = np.linspace(0, np.pi, num_val)
    detection_pola = np.zeros((self.n_sat, num_val, num_val))
    detection_spectro = np.zeros((self.n_sat, num_val, num_val))

    for ite in range(self.n_sat):
      # detection[ite] = np.array([[eff_area_func(trafile.decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="FoV") for phi in phi_world] for theta in theta_world])
      detection_pola[ite] = np.array([[eff_area_pola_func(decra2tp(theta, phi, self.sat_info[ite], unit="rad")[0],
                                                          self.sat_info[ite][2], func_type="cos") for phi in phi_world] for
                                      theta in theta_world])
      detection_spectro[ite] = np.array([[eff_area_spectro_func(
        decra2tp(theta, phi, self.sat_info[ite], unit="rad")[0], self.sat_info[ite][2], func_type="data") for phi in
                                          phi_world] for theta in theta_world])

    detec_sum_pola = np.sum(detection_pola, axis=0)
    detec_sum_spectro = np.sum(detection_spectro, axis=0)

    phi_plot, theta_plot = np.meshgrid(phi_world, theta_world)
    detec_min_pola = int(np.min(detec_sum_pola))
    detec_max_pola = int(np.max(detec_sum_pola))
    detec_min_spectro = int(np.min(detec_sum_spectro))
    detec_max_spectro = int(np.max(detec_sum_spectro))
    cmap_pola = mpl.cm.Greens_r
    cmap_spectro = mpl.cm.Oranges_r

    # Eff_area plots for polarimetry
    # levels_pola = range(int(detec_min_pola / 2) * 2, detec_max_pola + 1)
    levels_pola = range(int(detec_min_pola), int(detec_max_pola) + 1,
                        int((int(detec_max_pola) + 1 - int(detec_min_pola)) / 15))

    plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_pola, cmap=cmap_pola)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_pola)
    cbar.set_label("Effective area at for polarisation (cm²)", rotation=270, labelpad=20)
    plt.savefig("figtest")
    if save:
      plt.savefig("eff_area_noproj_pola")
    if show:
      plt.show()

    plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum_pola, cmap=cmap_pola)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_pola)
    cbar.set_label("Effective area for polarisation (cm²)", rotation=270, labelpad=20)
    if save:
      plt.savefig("eff_area_proj_pola")
    if show:
      plt.show()

    # Eff_area plots for spectroscopy
    # levels_spectro = range(int(detec_min_spectro / 2) * 2, detec_max_spectro + 1)
    levels_spectro = range(int(detec_min_spectro), int(detec_max_spectro) + 1,
                           int((int(detec_max_spectro) + 1 - int(detec_min_spectro)) / 15))

    plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_spectro, cmap=cmap_spectro)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_spectro)
    cbar.set_label("Effective area for spectrometry (cm²)", rotation=270, labelpad=20)
    if save:
      plt.savefig("eff_area_noproj_spectro")
    if show:
      plt.show()

    plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum_spectro, cmap=cmap_spectro)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_spectro)
    cbar.set_label("Effective area for spectrometry (cm²)", rotation=270, labelpad=20)
    if save:
      plt.savefig("eff_area_proj_spectro")
    if show:
      plt.show()

    print(f"La surface efficace moyenne pour la polarisation est de {np.mean(np.mean(detec_sum_pola, axis=1))} cm²")
    print(f"La surface efficace moyenne pour la spectrométrie est de {np.mean(np.mean(detec_sum_spectro, axis=1))} cm²")

  def grb_map_plot(self, mode="no_cm"):
    """
    Display the catalog GRBs position in the sky
    """
    if self.cat_file == "None":
      print("No cat file has been given, the GRBs' position cannot be displayed")
    else:
      cat_data = Catalog(self.cat_file, self.sttype)
      # Extracting dec and ra from catalog and transforms decimal degrees into degrees into the right frame
      thetap = [np.sum(np.array(dec.split(" ")).astype(np.float)/[1, 60, 3600]) if len(dec.split("+")) == 2 else np.sum(np.array(dec.split(" ")).astype(np.float)/[1, -60, -3600]) for dec in cat_data.dec]
      thetap = np.deg2rad(np.array(thetap))
      phip = [np.sum(np.array(ra.split(" ")).astype(np.float)/[1, 60, 3600]) if len(ra.split("+")) == 2 else np.sum(np.array(ra.split(" ")).astype(np.float)/[1, -60, -3600]) for ra in cat_data.ra]
      phip = np.mod(np.deg2rad(np.array(phip))+np.pi, 2*np.pi)-np.pi

      plt.subplot(111, projection="aitoff")
      plt.xlabel("RA (°)")
      plt.ylabel("DEC (°)")
      plt.grid(True)
      plt.title("Map of GRB")
      if mode == "no_cm":
        plt.scatter(phip, thetap, s=12, marker="*")
      elif mode == "t90":
        sc = plt.scatter(phip, thetap, s=12, marker="*", c=cat_data.t90, norm=colors.LogNorm())
        plt.colorbar(sc)
      plt.show()


  def mdp_histogram(self, selected_sat="const", mdp_threshold=1, cumul=True, n_bins=30, y_scale="log"):
    """
    Display and histogram representing the number of grb of a certain mdp per year
    """
    if self.cat_file == "longGBM.txt":
      grb_type = "lGRB"
    elif self.cat_file == "shortGRB.txt":
      grb_type = "sGRB"
    else:
      grb_type = "undefined source"

    mdp_list = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if sim.const_data.mdp <= mdp_threshold:
                mdp_list.append(sim.const_data.mdp*100)
            else:
              if sim[selected_sat].mdp <= mdp_threshold:
                mdp_list.append(sim[selected_sat].mdp*100)
    plt.hist(mdp_list, bins=n_bins, cumulative=cumul, histtype="step", weights=[self.weights] * len(mdp_list))
    if cumul:
      plt.title(f"Cumulative distribution of the MDP - {grb_type}")
    else:
      plt.title(f"Distribution of the MDP - {grb_type}")
    plt.xlabel("MPD (%)")
    plt.ylabel("Number of detection per year")
    plt.grid()
    plt.yscale(y_scale)
    plt.show()


  def hits_vs_energy(self, num_grb, num_sim, selected_sat, n_bins=30):
    """

    """
    hits_energy = []
    if self.alldata[num_grb] is not None:
      if self.alldata[num_grb][num_sim] is not None:
        if type(selected_sat) == int:
          hits_energy = self.alldata[num_grb][num_sim][selected_sat].CE_sum
        elif selected_sat == "const":
          hits_energy = self.alldata[num_grb][num_sim].const_data.CE_sum

    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle("Energy distribution of photons for a GRB")
    ax1.hist(hits_energy, bins=n_bins, cumulative=0, histtype="step")
    ax1.set(xlabel="Energy (keV)", ylabel="Number of photon detected", yscale="linear")
    plt.show()

  def peak_flux_distri(self, selected_sat="const", snr_min=5, n_bins=30, y_scale="log"):
    """

    """
    hist_pflux = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if sim.const_data.snr > snr_min:
                hist_pflux.append(source.p_flux)
            else:
              if sim[selected_sat].snr > snr_min:
                hist_pflux.append(source.p_flux)
    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle("Peak flux distribution of detected long GRB")
    ax1.hist(hist_pflux, bins=np.logspace(int(np.log10(min(hist_pflux))) - 1, int(np.log10(max(hist_pflux))), n_bins),
             cumulative=False, histtype="step", weights=[self.weights] * len(hist_pflux))
    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Number of detection per year", xscale='log', yscale=y_scale)
    ax1.legend()
    plt.show()

  def det_proba_vs_flux(self, selected_sat="const"):
    """
    sat contains either the number of the satellite selected or "const"
    """
    p_flux_list = []
    det_prob_fov_list = []
    det_prob_sky_list = []
    for source in self.alldata:
      if source is not None:
        p_flux_list.append(source.p_flux)
        if selected_sat == "const":
          det_prob_fov_list.append(source.const_proba_detec_fov)
          det_prob_sky_list.append(source.const_proba_detec_sky)
        else:
          det_prob_fov_list.append(source.proba_detec_fov[selected_sat])
          det_prob_sky_list.append(source.proba_detec_sky[selected_sat])

    distrib, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    distrib.suptitle("Detection probability vs peak flux of detected long GRB - GRB in the whole sky (left) and only in the FoV (right)")
    ax1.scatter(p_flux_list, det_prob_sky_list, s=2)
    ax2.scatter(p_flux_list, det_prob_fov_list, s=2)

    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Detection probability", xscale='log', )
    ax1.legend()
    ax2.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Detection probability", xscale='log', )
    ax2.legend()
    plt.show()

  def compton_im_proba_vs_flux(self, selected_sat="const"):
    """
    sat contains either the number of the satellite selected or "const"
    """
    p_flux_list = []
    comp_im_prob_fov_list = []
    comp_im_prob_sky_list = []
    for source in self.alldata:
      if source is not None:
        p_flux_list.append(source.p_flux)
        if selected_sat == "const":
          comp_im_prob_fov_list.append(source.const_proba_compton_image_fov)
          comp_im_prob_sky_list.append(source.const_proba_compton_image_sky)
        else:
          comp_im_prob_fov_list.append(source.proba_compton_image_fov[selected_sat])
          comp_im_prob_sky_list.append(source.proba_compton_image_sky[selected_sat])
    distrib, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    distrib.suptitle("Compton Image probability vs peak flux of detected long GRB - GRB in the whole sky (left) and only in the FoV (right)")
    ax1.scatter(p_flux_list, comp_im_prob_sky_list, s=2)
    ax2.scatter(p_flux_list, comp_im_prob_fov_list, s=2)

    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Compton image probability", xscale='log', )
    ax1.legend()
    ax2.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Compton image probability", xscale='log', )
    ax2.legend()
    plt.show()

  def mu100_distri(self, selected_sat="const", n_bins=30, y_scale="log"):
    """

    """
    mu_100_list = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if sim.const_data.mu100 is not None:
                mu_100_list.append(sim.const_data.mu100)
            else:
              if sim[selected_sat].mu100 is not None:
                mu_100_list.append(sim[selected_sat].mu100)
    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle("mu100 distribution of detected GRB")
    ax1.hist(mu_100_list, bins=n_bins, cumulative=0, histtype="step", weights=[self.weights] * len(mu_100_list))
    ax1.set(xlabel="mu100 (dimensionless)", ylabel="Number of detection per year", yscale=y_scale)
    plt.show()

  def pa_distribution(self, selected_sat="const", n_bins=30, y_scale="log"):
    """

    """
    pa_list = []
    for source in self.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if selected_sat == "const":
              if sim.const_data.pa is not None:
                pa_list.append(sim.const_data.pa)
            else:
              if sim[selected_sat].pa is not None:
                pa_list.append(sim[selected_sat].pa)
    distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    distrib.suptitle("Polarization angle distribution of detected GRB")
    ax1.hist(pa_list, bins=n_bins, cumulative=0, histtype="step", weights=[self.weights] * len(pa_list))
    ax1.set(xlabel="Polarization angle (°)", ylabel="Number of detection per year", yscale=y_scale)
    plt.show()


  def mdp99_vs_fluence(self, selected_sat="const", mdp_threshold=1, n_bins=30, y_scale="log"):
    """

    """
    if self.fluence is not None:
      mdp_list = []
      fluence_list = []
      mdp_count = 0
      no_detec_fluence = []
      for source_ite, source in enumerate(self.alldata):
        if source is not None:
          for sim in source:
            if sim is not None:
              if selected_sat == "const":
                if sim.const_data.mdp <= mdp_threshold:
                  mdp_list.append(sim.const_data.mdp * 100)
                  fluence_list.append(self.fluence[source_ite])
                else:
                  no_detec_fluence.append(self.fluence[source_ite])
              else:
                if sim[selected_sat].mdp <= mdp_threshold:
                  mdp_list.append(sim[selected_sat].mdp * 100)
                  fluence_list.append(self.fluence[source_ite])
                else:
                  no_detec_fluence.append(self.fluence[source_ite])
              mdp_count += 1

      distrib, ax1 = plt.subplots(1, 1, figsize=(8, 6))
      distrib.suptitle("MDP99 as a functin of fluence of detected GRB")
      for val in np.unique(no_detec_fluence):
        ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
      ax1.scatter(fluence_list, mdp_list, s=3, label=f'Detected GRB \nRatio of detectable polarization : {len(mdp_list) / mdp_count}')
      ax1.set(xlabel="fluence (erg.cm-2)", ylabel="MDP99 (%)", yscale='linear', xscale='log',
              xlim=(10 ** (int(np.log10(np.min(fluence_list))) - 1), 10 ** (int(np.log10(np.max(fluence_list))))))
      ax1.legend()
      plt.show()


bkg = "./backgrounds/bkg"  # _background_sat0_0000_90.0_0.0.inc1.id1.extracted.tra"
param = "./test/polGBM.par"
erg = (100, 460)
test = AllSourceData(bkg, param, erg)
test.make_const()
test.analyze()

#bkg = "./backgrounds/bkg"  # _background_sat0_0000_90.0_0.0.inc1.id1.extracted.tra"
#param = "./test/polGBM.par"
#erg = (100, 460)
#test = AllSourceData(bkg, param, erg)

# class SimulationData(SimData):
#   """
#   Class extracting and containing the data from the .tra files
#   """
#
#   def __init__(self, source_prefix, bkg_prefix, param_file, erg_cut, duration):
#     super.__init__()
#     pass
