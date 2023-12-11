# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# Developped modules imports
from launch_mu100_sim import read_par
from funcmod import *
from MFit import Fit

# Ploting adjustments
mpl.use('Qt5Agg')
# plt.rcParams.update({'font.size': 20})


class Mu100Container(list):
  """
  Class containing the information for mu100 files
  """

  def __init__(self, mu100parfile, ergcut=(100, 460), armcut=180):
    """
    :param bkgparfile : background parameter file
    :param save_pos : True if the interaction positions are to be saved
    :param save_time : True if the interaction times are to be saved
    :param ergcut : energy cut to apply
    """
    geom, revanf, mimrecf, source_base, spectra, bandparam, poltime, unpoltime, decs, ras = read_par(mu100parfile)
    self.geometry = geom       # To compare with data/mu100 and make sure everything works with the same softs
    self.revanfile = revanf    # To compare with data/mu100 and make sure everything works with the same softs
    self.mimrecfile = mimrecf  # To compare with data/mu100 and make sure everything works with the same softs
    self.bandparam = bandparam
    self.poltime = poltime
    self.unpoltime = unpoltime
    self.decs = decs
    self.ras = ras
    self.bins = set_bins("fixed")
    self.ergcut = ergcut
    func = lambda x: band(x, self.bandparam[0], self.bandparam[1], self.bandparam[2], self.bandparam[3], self.bandparam[4])
    self.fluence = quad(func, self.ergcut[0], self.ergcut[1])[0] * self.poltime

    geom_name = geom.split(".geo.setup")[0].split("/")[-1]
    saving = f"mu-seff-saved_{geom_name}_{np.min(self.decs):.0f}-{np.max(self.decs):.0f}-{len(self.decs):.0f}_{np.min(self.ras):.0f}-{np.max(self.ras):.0f}-{len(self.ras):.0f}.txt"
    if ergcut is None:
      ergname = "ergcut-None"
    else:
      ergname = f"ergcut-{ergcut[0]}-{ergcut[1]}"
    if armcut is None:
      armname = "armcut-None"
    else:
      armname = f"armcut-{armcut[0]}-{armcut[1]}"
    cond_saving = f"condensed-saved_{geom_name}_{np.min(self.decs):.0f}-{np.max(self.decs):.0f}-{len(self.decs):.0f}_{np.min(self.ras):.0f}-{np.max(self.ras):.0f}-{len(self.ras):.0f}_{ergname}_{armname}.txt"

    if not saving in os.listdir(f"./mu100/sim_{geom_name}"):
      self.save_fulldata(f"./mu100/sim_{geom_name}/{saving}")
      self.save_condenseddata(f"./mu100/sim_{geom_name}/{saving}", f"./mu100/sim_{geom_name}/{cond_saving}", ergcut, armcut)
    else:
      if not cond_saving in os.listdir(f"./mu100/sim_{geom_name}"):
        self.save_condenseddata(f"./mu100/sim_{geom_name}/{saving}", f"./mu100/sim_{geom_name}/{cond_saving}", ergcut, armcut)
    list.__init__(self, self.read_data(f"./mu100/sim_{geom_name}/{cond_saving}"))

  def save_fulldata(self, file):
    """
    Function used to save the mu100/seff data into a txt file
    """
    with open(file, "w") as f:
      f.write("# File containing raw mu100 data for : \n")
      f.write(f"# Geometry : {self.geometry}\n")
      f.write(f"# Revan file : {self.revanfile}\n")
      f.write(f"# Mimrec file : {self.mimrecfile}\n")
      f.write(f"# Pol simulation time : {self.poltime}\n")
      f.write(f"# Unpol simulation time : {self.unpoltime}\n")
      f.write(f"# dec min-max-number of value : {np.min(self.decs)}-{np.max(self.decs)}-{len(self.decs)}\n")
      f.write(f"# ra min-max-number of value : {np.min(self.ras)}-{np.max(self.ras)}-{len(self.ras)}\n")
      # Keys if EVERYTHING is saved
      # f.write("# Keys : dec | ra | compton_ener_pol | compton_ener_unpol | compton_second_pol | compton_second_unpol | single_ener_pol | single_ener_unpol | compton_firstpos_pol |compton_firstpos_unpol | compton_secpos_pol | compton_secpos_unpol | single_pos_pol | single_pos_unpol | compton_time_pol | compton_time_unpol | single_time_pol | single_time_unpol | single_pol | single_unpol | single_cr_pol | single_cr | compton | compton_cr | calor | dsssd | side\n")
      # Keys if the usefull data are saved
      f.write("# Keys : dec | ra | compton_ener_pol | compton_ener_unpol | compton_second_pol | compton_second_unpol | single_ener_pol\n")
      for dec in self.decs:
        for ra in self.ras:
          #  The commented parts are the ones that may not be useful
          geom_name = self.geometry.split(".geo.setup")[0].split("/")[-1]
          polsname = f"./mu100/sim_{geom_name}/sim/mu100_{dec:.1f}_{ra:.1f}pol.inc1.id1.extracted.tra"
          unpolsname = f"./mu100/sim_{geom_name}/sim/mu100_{dec:.1f}_{ra:.1f}pol.inc1.id1.extracted.tra"
          datapol = readfile(polsname)
          dataunpol = readfile(unpolsname)
          compton_second_pol = []
          compton_second_unpol = []
          compton_ener_pol = []
          compton_ener_unpol = []
          # compton_time_pol = []
          # compton_time_unpol = []
          compton_firstpos_pol = []
          compton_firstpos_unpol = []
          compton_secpos_pol = []
          compton_secpos_unpol = []
          single_ener_pol = []
          # single_ener_unpol = []
          # single_time_pol = []
          # single_time_unpol = []
          # single_pos_pol = []
          # single_pos_unpol = []
          for event in datapol:
            reading = readevt(event, None)
            if len(reading) == 5:
              compton_second_pol.append(reading[0])
              compton_ener_pol.append(reading[1])
              # compton_time_pol.append(reading[2])
              # compton_firstpos_pol.append(reading[3])
              # compton_secpos_pol.append(reading[4])
            elif len(reading) == 3:
              single_ener_pol.append(reading[0])
              # single_time_pol.append(reading[1])
              # single_pos_pol.append(reading[2])
          for event in dataunpol:
            reading = readevt(event, None)
            if len(reading) == 5:
              compton_second_unpol.append(reading[0])
              compton_ener_unpol.append(reading[1])
              # compton_time_unpol.append(reading[2])
              # compton_firstpos_unpol.append(reading[3])
              # compton_secpos_unpol.append(reading[4])
            # elif len(reading) == 3:
            #   single_ener_unpol.append(reading[0])
              # single_time_unpol.append(reading[1])
              # single_pos_unpol.append(reading[2])
          f.write("NewPos\n")
          f.write(f"{dec}\n")
          f.write(f"{ra}\n")
          for ite in range(len(compton_second_pol) - 1):
            f.write(f"{compton_second_pol[ite]}|")
          f.write(f"{compton_second_pol[-1]}\n")
          for ite in range(len(compton_second_unpol) - 1):
            f.write(f"{compton_second_unpol[ite]}|")
          f.write(f"{compton_second_unpol[-1]}\n")

          for ite in range(len(compton_ener_pol) - 1):
            f.write(f"{compton_ener_pol[ite]}|")
          f.write(f"{compton_ener_pol[-1]}\n")
          for ite in range(len(compton_ener_unpol) - 1):
            f.write(f"{compton_ener_unpol[ite]}|")
          f.write(f"{compton_ener_unpol[-1]}\n")

          # for ite in range(len(compton_time_pol) - 1):
          #   f.write(f"{compton_time_pol[ite]}|")
          # f.write(f"{compton_time_pol[-1]}\n")
          # for ite in range(len(compton_time_unpol) - 1):
          #   f.write(f"{compton_time_unpol[ite]}|")
          # f.write(f"{compton_time_unpol[-1]}\n")

          for ite in range(len(compton_firstpos_pol) - 1):
            string = f"{compton_firstpos_pol[ite][0]}_{compton_firstpos_pol[ite][1]}_{compton_firstpos_pol[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_firstpos_pol[-1][0]}_{compton_firstpos_pol[-1][1]}_{compton_firstpos_pol[-1][2]}"
          f.write(f"{string}\n")
          for ite in range(len(compton_firstpos_unpol) - 1):
            string = f"{compton_firstpos_unpol[ite][0]}_{compton_firstpos_unpol[ite][1]}_{compton_firstpos_unpol[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_firstpos_unpol[-1][0]}_{compton_firstpos_unpol[-1][1]}_{compton_firstpos_unpol[-1][2]}"
          f.write(f"{string}\n")

          for ite in range(len(compton_secpos_pol) - 1):
            string = f"{compton_secpos_pol[ite][0]}_{compton_secpos_pol[ite][1]}_{compton_secpos_pol[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_secpos_pol[-1][0]}_{compton_secpos_pol[-1][1]}_{compton_secpos_pol[-1][2]}"
          f.write(f"{string}\n")
          for ite in range(len(compton_secpos_unpol) - 1):
            string = f"{compton_secpos_unpol[ite][0]}_{compton_secpos_unpol[ite][1]}_{compton_secpos_unpol[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_secpos_unpol[-1][0]}_{compton_secpos_unpol[-1][1]}_{compton_secpos_unpol[-1][2]}"
          f.write(f"{string}\n")

          for ite in range(len(single_ener_pol) - 1):
            f.write(f"{single_ener_pol[ite]}|")
          f.write(f"{single_ener_pol[-1]}\n")
          # for ite in range(len(single_ener_unpol) - 1):
          #   f.write(f"{single_ener_unpol[ite]}|")
          # f.write(f"{single_ener_unpol[-1]}\n")

          # for ite in range(len(single_time_pol) - 1):
          #   f.write(f"{single_time_pol[ite]}|")
          # f.write(f"{single_time_pol[-1]}\n")
          # for ite in range(len(single_time_unpol) - 1):
          #   f.write(f"{single_time_unpol[ite]}|")
          # f.write(f"{single_time_unpol[-1]}\n")

          # for ite in range(len(single_pos_pol) - 1):
          #   string = f"{single_pos_pol[ite][0]}_{single_pos_pol[ite][1]}_{single_pos_pol[ite][2]}"
          #   f.write(f"{string}|")
          # string = f"{single_pos_pol[-1][0]}_{single_pos_pol[-1][1]}_{single_pos_pol[-1][2]}"
          # f.write(f"{string}\n")
          # for ite in range(len(single_pos_unpol) - 1):
          #   string = f"{single_pos_unpol[ite][0]}_{single_pos_unpol[ite][1]}_{single_pos_unpol[ite][2]}"
          #   f.write(f"{string}|")
          # string = f"{single_pos_unpol[-1][0]}_{single_pos_unpol[-1][1]}_{single_pos_unpol[-1][2]}"
          # f.write(f"{string}\n")

  def save_condenseddata(self, fullfile, condfile, ergcut, armcut):
    """
    Function used to save the mu100/seff data into a txt file
    """
    var_x = .5 * (self.bins[1:] + self.bins[:-1])
    binw = self.bins[1:] - self.bins[:-1]
    with open(fullfile, "r") as fullf:
      fulldata = fullf.read().split("NewPos\n")
    with open(condfile, "w") as f:
      f.write("# File containing mu100 and seff data for : \n")
      f.write(f"# Geometry : {self.geometry}\n")
      f.write(f"# Revan file : {self.revanfile}\n")
      f.write(f"# Mimrec file : {self.mimrecfile}\n")
      f.write(f"# Pol simulation time : {self.poltime}\n")
      f.write(f"# Unpol simulation time : {self.unpoltime}\n")
      f.write(f"# dec min-max-number of value : {np.min(self.decs)}-{np.max(self.decs)}-{len(self.decs)}\n")
      f.write(f"# ra min-max-number of value : {np.min(self.ras)}-{np.max(self.ras)}-{len(self.ras)}\n")
      f.write("# Keys : dec | ra | mu100 | mu100_err | pa | pa_err | fit_goodness | seff_compton | seff_single\n")
      for filedata in fulldata[1:]:
        lines = filedata.split("\n")
        # Extraction of position
        dec = float(lines[0])
        ra = float(lines[1])
        # Extraction of compton energies for pol and unpol events
        compton_second_pol = np.array(lines[2].split("|"), dtype=float)
        compton_second_unpol = np.array(lines[3].split("|"), dtype=float)
        compton_ener_pol = np.array(lines[4].split("|"), dtype=float)
        compton_ener_unpol = np.array(lines[5].split("|"), dtype=float)
        # Extraction of compton position for pol and unpol events
        compton_firstpos_pol = np.array([val.split("_") for val in lines[6].split("|")], dtype=float)
        compton_firstpos_unpol = np.array([val.split("_") for val in lines[7].split("|")], dtype=float)
        compton_secpos_pol = np.array([val.split("_") for val in lines[8].split("|")], dtype=float)
        compton_secpos_unpol = np.array([val.split("_") for val in lines[9].split("|")], dtype=float)
        # Extraction of energy for single events
        single_ener_pol = np.array(lines[10].split("|"), dtype=float)

        if ergcut is not None:
          compton_pol_index = np.where(compton_ener_pol >= ergcut[0], np.where(compton_ener_pol <= ergcut[1], True, False), False)
          compton_unpol_index = np.where(compton_ener_unpol >= ergcut[0], np.where(compton_ener_unpol <= ergcut[1], True, False), False)
          single_index = np.where(single_ener_pol >= ergcut[0], np.where(single_ener_pol <= ergcut[1], True, False), False)

          compton_second_pol = compton_second_pol[compton_pol_index]
          compton_second_unpol = compton_second_unpol[compton_unpol_index]
          compton_ener_pol = compton_ener_pol[compton_pol_index]
          compton_ener_unpol = compton_ener_unpol[compton_unpol_index]
          compton_firstpos_pol = compton_firstpos_pol[compton_pol_index]
          compton_firstpos_unpol = compton_firstpos_unpol[compton_unpol_index]
          compton_secpos_pol = compton_secpos_pol[compton_pol_index]
          compton_secpos_unpol = compton_secpos_unpol[compton_unpol_index]
          single_ener_pol = single_ener_pol[single_index]

        pol, polar_from_position_pol = angle(compton_secpos_pol - compton_firstpos_pol, dec, ra, f"{dec}_{ra}_pol", 0, 0)
        unpol, polar_from_position_unpol = angle(compton_secpos_unpol - compton_firstpos_unpol, dec, ra, f"{dec}_{ra}_unpol", 0, 0)

        if armcut is not None:
          polar_from_energy_pol = calculate_polar_angle(compton_second_pol, compton_ener_pol)
          polar_from_energy_unpol = calculate_polar_angle(compton_second_unpol, compton_ener_unpol)
          arm_pol = polar_from_position_pol - polar_from_energy_pol
          arm_unpol = polar_from_position_unpol - polar_from_energy_unpol
          accepted_arm_pol = np.where(np.abs(arm_pol) <= armcut, True, False)
          accepted_arm_unpol = np.where(np.abs(arm_unpol) <= armcut, True, False)
          pol = pol[accepted_arm_pol]
          unpol = unpol[accepted_arm_unpol]

        hist_pol = np.histogram(pol, self.bins)[0] / binw
        hist_unpol = np.histogram(unpol, self.bins)[0] / binw
        fit_mod = None
        # fit_const = None
        if 0. in hist_unpol:
          print(f"Unpolarized data do not allow a fit - {dec}_{ra} : a bin is empty")
        else:
          polarigram_error = err_calculation(np.histogram(pol, self.bins)[0], np.histogram(unpol, self.bins)[0], binw)
          if 0. in polarigram_error:
            print(f"Polarized data do not allow a fit - {dec}_{ra} : a bin is empty leading to uncorrect fit")
          else:
            histo = hist_pol / hist_unpol * np.mean(hist_unpol)
            fit_mod = Fit(modulation_func, var_x, histo, yerr=polarigram_error, comment="modulation")
            # fit_const = Fit(lambda x, a: a * x / x, var_x, histo, yerr=polarigram_error, comment="constant")
        # pa, mu100, fit_compton_cr = fit_mod.popt
        pa, mu100 = fit_mod.popt[:2]
        if mu100 < 0:
          pa = (pa + 90) % 180
          mu100 = - mu100
        else:
          pa = pa % 180
        pa_err = np.sqrt(fit_mod.pcov[0][0])
        mu100_err = np.sqrt(fit_mod.pcov[1][1])
        # fit_compton_cr_err = np.sqrt(fit_mod.pcov[2][2])
        fit_goodness = fit_mod.q2 / (len(fit_mod.x) - fit_mod.nparam)

        seff_compton = len(pol) / self.fluence
        seff_single = len(single_ener_pol) / self.fluence

        f.write("NewPos\n")
        f.write(f"{dec}\n")
        f.write(f"{ra}\n")
        f.write(f"{mu100}\n")
        f.write(f"{mu100_err}\n")
        f.write(f"{pa}\n")
        f.write(f"{pa_err}\n")
        f.write(f"{fit_goodness}\n")
        f.write(f"{seff_compton}\n")
        f.write(f"{seff_single}\n")

  def read_data(self, file):
    """
    Function used to read the bkg txt file
    """
    with open(file, "r") as f:
      files_saved = f.read().split("NewPos\n")
    return [Mu100Data(file_saved) for file_saved in files_saved]


class Mu100Data:
  """
  Class containing the data for 1 GRB, for 1 sim, and 1 satellite
  """

  def __init__(self, data):
    """
    -data_list : list of 2 files (pol or pol+unpol) from which extract the data
    """
    ##############################################################
    # Attributes filled with file reading
    lines = data.split("\n")
    self.dec = lines[0]
    self.ra = lines[1]
    self.mu100 = lines[2]
    self.mu100_err = lines[3]
    self.pa = lines[4]
    self.pa_err = lines[5]
    # Set with the fit or for the fit
    # =0 : fit perfectly
    # ~1 : fit reasonably
    # >1 : not a good fit
    # >>1 : very poor fit
    self.fit_goodness = lines[6]
    self.s_eff_compton = lines[7]
    self.s_eff_single = lines[8]

  @staticmethod
  def get_keys():
    print("======================================================================")
    print("    Attributes")
    print(" mu100 for the satellite/constellation                                    .mu100")
    print(" Polarization angle obtained from the polarigram                          .pa")
    print(" Polarization angle error from the fit                                    .pa_err")
    print(" mu100 error from the fit                                                 .mu100_err")
    print(" Fit goodness                                                             .fit_goodness")
