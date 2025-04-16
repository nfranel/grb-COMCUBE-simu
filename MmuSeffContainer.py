# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules
import os.path

# Package imports
# import matplotlib.pyplot as plt
import matplotlib as mpl
# Developped modules imports
from funcmod import *
from launch_mu100_sim import make_ra_list
from MFit import Fit
import matplotlib.pyplot as plt

# Ploting adjustments
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')

# plt.rcParams.update({'font.size': 20})


class MuSeffContainer(list):
  """
  Class containing the information for mu100 files
  """

  def __init__(self, mu100parfile, ergcut=(100, 460), armcut=180):
    """
    :param mu100parfile: mu100 parameter file
    :param ergcut: energy cut to apply
    :param armcut: ARM cut to apply
    """
    geom, revanf, mimrecf, source_base, spectra, bandparam, poltime, unpoltime, decs, ras = read_mupar(mu100parfile)
    self.geometry = geom       # TODO To compare with data/mu100 and make sure everything works with the same softs
    self.revanfile = revanf    # To compare with data/mu100 and make sure everything works with the same softs
    self.mimrecfile = mimrecf  # To compare with data/mu100 and make sure everything works with the same softs
    self.bandparam = bandparam
    self.poltime = poltime
    self.unpoltime = unpoltime
    self.decs = decs
    self.ras = ras
    self.bins = set_bins("fixed")
    self.ergcut = ergcut
    self.fluence = use_scipyquad(band, self.ergcut[0], self.ergcut[1], func_args=tuple(bandparam), x_logscale=True)[0] * self.poltime
    geom_name = geom.split(".geo.setup")[0].split("/")[-1]

    saving = f"mu-seff-saved_{geom_name}_{self.decs[0]:.0f}-{self.decs[1]:.0f}-{self.decs[2]:.0f}_{self.ras[0]:.0f}-{self.ras[1]:.0f}-{self.ras[2]:.0f}.txt"
    ergname = f"ergcut-{ergcut[0]}-{ergcut[1]}"
    armname = f"armcut-{armcut}"
    cond_saving = f"condensed-mu-seff-saved_{geom_name}_{self.decs[0]:.0f}-{self.decs[1]:.0f}-{self.decs[2]:.0f}_{self.ras[0]:.0f}-{self.ras[1]:.0f}-{self.ras[2]:.0f}_{ergname}_{armname}.txt"
    if cond_saving not in os.listdir(f"./mu100/sim_{geom_name}"):
      if saving not in os.listdir(f"./mu100/sim_{geom_name}"):
        init_time = time()
        print("###########################################################################")
        print(" mu/Seff data not saved : Saving ")
        print("###########################################################################")
        self.save_fulldata(f"./mu100/sim_{geom_name}/{saving}")
        self.save_condenseddata(f"./mu100/sim_{geom_name}/{saving}", f"./mu100/sim_{geom_name}/{cond_saving}", ergcut, armcut)
        print("=======================================")
        print(" Saving of mu/Seff data finished in : ", time() - init_time, "seconds")
        print("=======================================")
      else:
        init_time = time()
        print("###########################################################################")
        print(" mu/Seff condensed data not saved : Saving ")
        print("###########################################################################")
        self.save_condenseddata(f"./mu100/sim_{geom_name}/{saving}", f"./mu100/sim_{geom_name}/{cond_saving}", ergcut, armcut)
        print("=======================================")
        print(" Saving of mu/Seff data finished in : ", time() - init_time, "seconds")
        print("=======================================")
    init_time = time()
    print("###########################################################################")
    print(" Extraction of mu/Seff data ")
    print("###########################################################################")
    list.__init__(self, self.read_data(f"./mu100/sim_{geom_name}/{cond_saving}"))
    print("=======================================")
    print(" Extraction of mu/Seff data finished in : ", time() - init_time, "seconds")
    print("=======================================")

  def save_fulldata(self, file):
    """
    Function used to save the mu100/seff data into a txt file
    :param file: path of the file to save full data
    """
    with open(file, "w") as f:
      f.write("# File containing raw mu100 data for : \n")
      f.write(f"# Geometry : {self.geometry}\n")
      f.write(f"# Revan file : {self.revanfile}\n")
      f.write(f"# Mimrec file : {self.mimrecfile}\n")
      f.write(f"# Pol simulation time : {self.poltime}\n")
      f.write(f"# Unpol simulation time : {self.unpoltime}\n")
      f.write(f"# dec min-max-number of value : {self.decs[0]}-{self.decs[1]}-{self.decs[2]}\n")
      f.write(f"# ra min-max-number of value (at equator) : {self.ras[0]}-{self.ras[1]}-{self.ras[2]}\n")
      # Keys if the usefull data are saved
      f.write("# Keys : dec | ra | compton_ener_pol | compton_ener_unpol | compton_second_pol | compton_second_unpol | single_ener_pol\n")
      for dec in np.linspace(self.decs[0], self.decs[1], self.decs[2]):
        for ra in make_ra_list(self.ras, dec):
          #  The commented parts are the ones that may not be useful
          geom_name = self.geometry.split(".geo.setup")[0].split("/")[-1]
          polsname = f"./mu100/sim_{geom_name}/sim/mu100_{dec:.1f}_{ra:.1f}pol.inc1.id1.extracted.tra"
          unpolsname = f"./mu100/sim_{geom_name}/sim/mu100_{dec:.1f}_{ra:.1f}unpol.inc1.id1.extracted.tra"
          # print("====== files loaded ======")
          datapol = readfile(polsname)
          dataunpol = readfile(unpolsname)
          compton_second_pol = []
          compton_second_unpol = []
          compton_ener_pol = []
          compton_ener_unpol = []
          compton_firstpos_pol = []
          compton_firstpos_unpol = []
          compton_secpos_pol = []
          compton_secpos_unpol = []
          single_ener_pol = []

          compton_second_pol_err = []
          compton_ener_pol_err = []
          compton_firstpos_pol_err = []
          compton_secpos_pol_err = []
          compton_second_unpol_err = []
          compton_ener_unpol_err = []
          compton_firstpos_unpol_err = []
          compton_secpos_unpol_err = []
          for event_pol in datapol:
            reading_pol = readevt(event_pol, None)
            if len(reading_pol) == 9:
              compton_second_pol.append(reading_pol[0])
              compton_ener_pol.append(reading_pol[1])
              compton_firstpos_pol.append(reading_pol[3])
              compton_secpos_pol.append(reading_pol[4])
              compton_second_pol_err.append(reading_pol[5])
              compton_ener_pol_err.append(reading_pol[6])
              compton_firstpos_pol_err.append(reading_pol[7])
              compton_secpos_pol_err.append(reading_pol[8])
            elif len(reading_pol) == 3:
              single_ener_pol.append(reading_pol[0])
          for event_unpol in dataunpol:
            reading_unpol = readevt(event_unpol, None)
            if len(reading_unpol) == 9:
              compton_second_unpol.append(reading_unpol[0])
              compton_ener_unpol.append(reading_unpol[1])
              compton_firstpos_unpol.append(reading_unpol[3])
              compton_secpos_unpol.append(reading_unpol[4])
              compton_second_unpol_err.append(reading_unpol[5])
              compton_ener_unpol_err.append(reading_unpol[6])
              compton_firstpos_unpol_err.append(reading_unpol[7])
              compton_secpos_unpol_err.append(reading_unpol[8])

          f.write("NewPos\n")
          f.write(f"{dec}\n")
          f.write(f"{ra}\n")
          for ite in range(len(compton_second_pol) - 1):
            f.write(f"{compton_second_pol[ite]}|")
          f.write(f"{compton_second_pol[-1]}\n")
          for ite in range(len(compton_second_unpol) - 1):
            f.write(f"{compton_second_unpol[ite]}|")
          f.write(f"{compton_second_unpol[-1]}\n")

          for ite in range(len(compton_second_pol_err) - 1):
            f.write(f"{compton_second_pol_err[ite]}|")
          f.write(f"{compton_second_pol_err[-1]}\n")
          for ite in range(len(compton_second_unpol_err) - 1):
            f.write(f"{compton_second_unpol_err[ite]}|")
          f.write(f"{compton_second_unpol_err[-1]}\n")

          for ite in range(len(compton_ener_pol) - 1):
            f.write(f"{compton_ener_pol[ite]}|")
          f.write(f"{compton_ener_pol[-1]}\n")
          for ite in range(len(compton_ener_unpol) - 1):
            f.write(f"{compton_ener_unpol[ite]}|")
          f.write(f"{compton_ener_unpol[-1]}\n")

          for ite in range(len(compton_ener_pol_err) - 1):
            f.write(f"{compton_ener_pol_err[ite]}|")
          f.write(f"{compton_ener_pol_err[-1]}\n")
          for ite in range(len(compton_ener_unpol_err) - 1):
            f.write(f"{compton_ener_unpol_err[ite]}|")
          f.write(f"{compton_ener_unpol_err[-1]}\n")

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

          for ite in range(len(compton_firstpos_pol_err) - 1):
            string = f"{compton_firstpos_pol_err[ite][0]}_{compton_firstpos_pol_err[ite][1]}_{compton_firstpos_pol_err[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_firstpos_pol_err[-1][0]}_{compton_firstpos_pol_err[-1][1]}_{compton_firstpos_pol_err[-1][2]}"
          f.write(f"{string}\n")
          for ite in range(len(compton_firstpos_unpol_err) - 1):
            string = f"{compton_firstpos_unpol_err[ite][0]}_{compton_firstpos_unpol_err[ite][1]}_{compton_firstpos_unpol_err[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_firstpos_unpol_err[-1][0]}_{compton_firstpos_unpol_err[-1][1]}_{compton_firstpos_unpol_err[-1][2]}"
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

          for ite in range(len(compton_secpos_pol_err) - 1):
            string = f"{compton_secpos_pol_err[ite][0]}_{compton_secpos_pol_err[ite][1]}_{compton_secpos_pol_err[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_secpos_pol_err[-1][0]}_{compton_secpos_pol_err[-1][1]}_{compton_secpos_pol_err[-1][2]}"
          f.write(f"{string}\n")
          for ite in range(len(compton_secpos_unpol_err) - 1):
            string = f"{compton_secpos_unpol_err[ite][0]}_{compton_secpos_unpol_err[ite][1]}_{compton_secpos_unpol_err[ite][2]}"
            f.write(f"{string}|")
          string = f"{compton_secpos_unpol_err[-1][0]}_{compton_secpos_unpol_err[-1][1]}_{compton_secpos_unpol_err[-1][2]}"
          f.write(f"{string}\n")

          for ite in range(len(single_ener_pol) - 1):
            f.write(f"{single_ener_pol[ite]}|")
          f.write(f"{single_ener_pol[-1]}\n")

  def save_condenseddata(self, fullfile, condfile, ergcut, armcut):
    """
    Function used to save the mu100/seff data into a txt file
    :param fullfile: path of the file where full data is saved
    :param condfile: path of the file to save condensed data
    :param ergcut: energy cut to apply
    :param armcut: ARM cut to apply
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
      f.write(f"# dec min-max-number of value : {self.decs[0]}-{self.decs[1]}-{self.decs[2]}\n")
      f.write(f"# ra min-max-number of value (at equator) : {self.ras[0]}-{self.ras[1]}-{self.ras[2]}\n")
      f.write("# Keys : dec | ra | mu100 | mu100_err | pa | pa_err | fit_goodness | seff_compton | seff_single\n")
      for filedata in fulldata[1:]:
        lines = filedata.split("\n")
        # Extraction of position
        dec = float(lines[0])
        ra = float(lines[1])
        # Extraction of compton energies for pol and unpol events
        compton_second_pol = np.array(lines[2].split("|"), dtype=float)
        compton_second_unpol = np.array(lines[3].split("|"), dtype=float)
        compton_second_pol_err = np.array(lines[4].split("|"), dtype=float)
        compton_second_unpol_err = np.array(lines[5].split("|"), dtype=float)
        compton_ener_pol = np.array(lines[6].split("|"), dtype=float)
        compton_ener_unpol = np.array(lines[7].split("|"), dtype=float)
        compton_ener_pol_err = np.array(lines[8].split("|"), dtype=float)
        compton_ener_unpol_err = np.array(lines[9].split("|"), dtype=float)
        # Extraction of compton position for pol and unpol events
        compton_firstpos_pol = np.array([val.split("_") for val in lines[10].split("|")], dtype=float)
        compton_firstpos_unpol = np.array([val.split("_") for val in lines[11].split("|")], dtype=float)
        compton_firstpos_pol_err = np.array([val.split("_") for val in lines[12].split("|")], dtype=float)
        compton_firstpos_unpol_err = np.array([val.split("_") for val in lines[13].split("|")], dtype=float)
        compton_secpos_pol = np.array([val.split("_") for val in lines[14].split("|")], dtype=float)
        compton_secpos_unpol = np.array([val.split("_") for val in lines[15].split("|")], dtype=float)
        compton_secpos_pol_err = np.array([val.split("_") for val in lines[16].split("|")], dtype=float)
        compton_secpos_unpol_err = np.array([val.split("_") for val in lines[17].split("|")], dtype=float)
        # Extraction of energy for single events
        single_ener_pol = np.array(lines[18].split("|"), dtype=float)

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
          compton_second_pol_err = compton_second_pol_err[compton_pol_index]
          compton_second_unpol_err = compton_second_unpol_err[compton_unpol_index]
          compton_ener_pol_err = compton_ener_pol_err[compton_pol_index]
          compton_ener_unpol_err = compton_ener_unpol_err[compton_unpol_index]
          compton_firstpos_pol_err = compton_firstpos_pol_err[compton_pol_index]
          compton_firstpos_unpol_err = compton_firstpos_unpol_err[compton_unpol_index]
          compton_secpos_pol_err = compton_secpos_pol_err[compton_pol_index]
          compton_secpos_unpol_err = compton_secpos_unpol_err[compton_unpol_index]
        scat_vec_pol_err = np.sqrt(compton_secpos_pol_err ** 2 + compton_firstpos_pol_err ** 2)
        scat_vec_unpol_err = np.sqrt(compton_secpos_unpol_err ** 2 + compton_firstpos_unpol_err ** 2)

        dec_err, ra_err = 1.12, 1.01  #  !! peut etre à re evaluer avec des valeurs de grb_wf err
        pol, polar_from_position_pol, pol_err = angle(compton_secpos_pol - compton_firstpos_pol, dec, ra, f"{dec}_{ra}_pol", 0, 0, scatter_vector_err=scat_vec_pol_err, grb_dec_sf_err=dec_err, grb_ra_sf_err=ra_err)
        unpol, polar_from_position_unpol, unpol_err = angle(compton_secpos_unpol - compton_firstpos_unpol, dec, ra, f"{dec}_{ra}_unpol", 0, 0, scatter_vector_err=scat_vec_unpol_err, grb_dec_sf_err=dec_err, grb_ra_sf_err=ra_err)

        polar_from_energy_pol, polar_from_energy_pol_err = calculate_polar_angle(compton_second_pol, compton_ener_pol, ener_sec_err=compton_second_pol_err, ener_tot_err=compton_ener_pol_err)
        polar_from_energy_unpol, polar_from_energy_unpol_err = calculate_polar_angle(compton_second_unpol, compton_ener_unpol, ener_sec_err=compton_second_unpol_err, ener_tot_err=compton_ener_unpol_err)
        arm_pol = polar_from_position_pol - polar_from_energy_pol
        arm_unpol = polar_from_position_unpol - polar_from_energy_unpol
        accepted_arm_pol = np.where(np.abs(arm_pol) <= armcut, True, False)
        accepted_arm_unpol = np.where(np.abs(arm_unpol) <= armcut, True, False)
        pol = pol[accepted_arm_pol]
        unpol = unpol[accepted_arm_unpol]

        hist_pol = np.histogram(pol, self.bins)[0]
        hist_unpol = np.histogram(unpol, self.bins)[0]
        hist_pol_err, hist_unpol_err = pol_unpol_hist_err(pol, unpol, pol_err, unpol_err, self.bins)

        # The polarigrams are normalized with the bin width !
        hist_pol_norm = hist_pol / binw
        hist_unpol_norm = hist_unpol / binw
        fit_mod = None
        if 0. in hist_unpol_norm:
          print(f"Unpolarized data do not allow a fit - {dec}_{ra} : a bin is empty")
        else:
          polarigram_error = err_calculation(hist_pol, hist_unpol, binw, hist_pol_err, hist_unpol_err)
          if 0. in polarigram_error:
            print(f"Polarized data do not allow a fit - {dec}_{ra} : a bin is empty leading to uncorrect fit")
          else:
            histo = hist_pol_norm / hist_unpol_norm * np.mean(hist_unpol_norm)
            fit_mod = Fit(modulation_func, var_x, histo, yerr=polarigram_error, comment="modulation")
        pa, mu100 = fit_mod.popt[:2]
        if mu100 < 0:
          pa = (pa + 90) % 180
          mu100 = - mu100
        else:
          pa = pa % 180
        pa_err = np.sqrt(fit_mod.pcov[0][0])
        mu100_err = np.sqrt(fit_mod.pcov[1][1])
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
    :param file: path of the file where condensed data is saved
    """
    with open(file, "r") as f:
      files_saved = f.read().split("NewPos\n")
    return [Mu100Data(file_saved) for file_saved in files_saved[1:]]

  def show_fit(self, dec_plot, ra_plot, armcut=180):

    geom_name = self.geometry.split(".geo.setup")[0].split("/")[-1]
    polsname = f"./mu100/sim_{geom_name}/sim/mu100_{dec_plot:.1f}_{ra_plot:.1f}pol.inc1.id1.extracted.tra"
    unpolsname = f"./mu100/sim_{geom_name}/sim/mu100_{dec_plot:.1f}_{ra_plot:.1f}unpol.inc1.id1.extracted.tra"
    if not(os.path.exists(polsname) and os.path.exists(unpolsname)):
      raise FileNotFoundError("Polarized or unpolarized file is not found")
    # print("====== files loaded ======")
    datapol = readfile(polsname)
    dataunpol = readfile(unpolsname)
    compton_second_pol = []
    compton_second_unpol = []
    compton_ener_pol = []
    compton_ener_unpol = []
    compton_firstpos_pol = []
    compton_firstpos_unpol = []
    compton_secpos_pol = []
    compton_secpos_unpol = []
    single_ener_pol = []

    compton_second_pol_err = []
    compton_ener_pol_err = []
    compton_firstpos_pol_err = []
    compton_secpos_pol_err = []
    compton_second_unpol_err = []
    compton_ener_unpol_err = []
    compton_firstpos_unpol_err = []
    compton_secpos_unpol_err = []
    for event_pol in datapol:
      reading_pol = readevt(event_pol, None)
      if len(reading_pol) == 9:
        compton_second_pol.append(reading_pol[0])
        compton_ener_pol.append(reading_pol[1])
        compton_firstpos_pol.append(reading_pol[3])
        compton_secpos_pol.append(reading_pol[4])
        compton_second_pol_err.append(reading_pol[5])
        compton_ener_pol_err.append(reading_pol[6])
        compton_firstpos_pol_err.append(reading_pol[7])
        compton_secpos_pol_err.append(reading_pol[8])
      elif len(reading_pol) == 3:
        single_ener_pol.append(reading_pol[0])
    for event_unpol in dataunpol:
      reading_unpol = readevt(event_unpol, None)
      if len(reading_unpol) == 9:
        compton_second_unpol.append(reading_unpol[0])
        compton_ener_unpol.append(reading_unpol[1])
        compton_firstpos_unpol.append(reading_unpol[3])
        compton_secpos_unpol.append(reading_unpol[4])
        compton_second_unpol_err.append(reading_unpol[5])
        compton_ener_unpol_err.append(reading_unpol[6])
        compton_firstpos_unpol_err.append(reading_unpol[7])
        compton_secpos_unpol_err.append(reading_unpol[8])

    compton_second_pol = np.array(compton_second_pol)
    compton_second_unpol = np.array(compton_second_unpol)
    compton_ener_pol = np.array(compton_ener_pol)
    compton_ener_unpol = np.array(compton_ener_unpol)
    compton_firstpos_pol = np.array(compton_firstpos_pol)
    compton_firstpos_unpol = np.array(compton_firstpos_unpol)
    compton_secpos_pol = np.array(compton_secpos_pol)
    compton_secpos_unpol = np.array(compton_secpos_unpol)
    single_ener_pol = np.array(single_ener_pol)

    compton_second_pol_err = np.array(compton_second_pol_err)
    compton_ener_pol_err = np.array(compton_ener_pol_err)
    compton_firstpos_pol_err = np.array(compton_firstpos_pol_err)
    compton_secpos_pol_err = np.array(compton_secpos_pol_err)
    compton_second_unpol_err = np.array(compton_second_unpol_err)
    compton_ener_unpol_err = np.array(compton_ener_unpol_err)
    compton_firstpos_unpol_err = np.array(compton_firstpos_unpol_err)
    compton_secpos_unpol_err = np.array(compton_secpos_unpol_err)

    var_x = .5 * (self.bins[1:] + self.bins[:-1])
    binw = self.bins[1:] - self.bins[:-1]

    if self.ergcut is not None:
      compton_pol_index = np.where(compton_ener_pol >= self.ergcut[0], np.where(compton_ener_pol <= self.ergcut[1], True, False), False)
      compton_unpol_index = np.where(compton_ener_unpol >= self.ergcut[0], np.where(compton_ener_unpol <= self.ergcut[1], True, False), False)
      single_index = np.where(single_ener_pol >= self.ergcut[0], np.where(single_ener_pol <= self.ergcut[1], True, False), False)

      compton_second_pol = compton_second_pol[compton_pol_index]
      compton_second_unpol = compton_second_unpol[compton_unpol_index]
      compton_ener_pol = compton_ener_pol[compton_pol_index]
      compton_ener_unpol = compton_ener_unpol[compton_unpol_index]
      compton_firstpos_pol = compton_firstpos_pol[compton_pol_index]
      compton_firstpos_unpol = compton_firstpos_unpol[compton_unpol_index]
      compton_secpos_pol = compton_secpos_pol[compton_pol_index]
      compton_secpos_unpol = compton_secpos_unpol[compton_unpol_index]
      single_ener_pol = single_ener_pol[single_index]
      compton_second_pol_err = compton_second_pol_err[compton_pol_index]
      compton_second_unpol_err = compton_second_unpol_err[compton_unpol_index]
      compton_ener_pol_err = compton_ener_pol_err[compton_pol_index]
      compton_ener_unpol_err = compton_ener_unpol_err[compton_unpol_index]
      compton_firstpos_pol_err = compton_firstpos_pol_err[compton_pol_index]
      compton_firstpos_unpol_err = compton_firstpos_unpol_err[compton_unpol_index]
      compton_secpos_pol_err = compton_secpos_pol_err[compton_pol_index]
      compton_secpos_unpol_err = compton_secpos_unpol_err[compton_unpol_index]
    scat_vec_pol_err = np.sqrt(compton_secpos_pol_err ** 2 + compton_firstpos_pol_err ** 2)
    scat_vec_unpol_err = np.sqrt(compton_secpos_unpol_err ** 2 + compton_firstpos_unpol_err ** 2)

    dec_err, ra_err = 1.12, 1.01  #  !! peut etre à re evaluer avec des valeurs de grb_wf err
    pol, polar_from_position_pol, pol_err = angle(compton_secpos_pol - compton_firstpos_pol, dec_plot, ra_plot, f"{dec_plot}_{ra_plot}_pol", 0, 0, scatter_vector_err=scat_vec_pol_err, grb_dec_sf_err=dec_err, grb_ra_sf_err=ra_err)
    unpol, polar_from_position_unpol, unpol_err = angle(compton_secpos_unpol - compton_firstpos_unpol, dec_plot, ra_plot, f"{dec_plot}_{ra_plot}_unpol", 0, 0, scatter_vector_err=scat_vec_unpol_err, grb_dec_sf_err=dec_err, grb_ra_sf_err=ra_err)

    polar_from_energy_pol, polar_from_energy_pol_err = calculate_polar_angle(compton_second_pol, compton_ener_pol, ener_sec_err=compton_second_pol_err, ener_tot_err=compton_ener_pol_err)
    polar_from_energy_unpol, polar_from_energy_unpol_err = calculate_polar_angle(compton_second_unpol, compton_ener_unpol, ener_sec_err=compton_second_unpol_err, ener_tot_err=compton_ener_unpol_err)
    arm_pol = polar_from_position_pol - polar_from_energy_pol
    arm_unpol = polar_from_position_unpol - polar_from_energy_unpol
    accepted_arm_pol = np.where(np.abs(arm_pol) <= armcut, True, False)
    accepted_arm_unpol = np.where(np.abs(arm_unpol) <= armcut, True, False)
    pol = pol[accepted_arm_pol]
    unpol = unpol[accepted_arm_unpol]

    hist_pol = np.histogram(pol, self.bins)[0]
    hist_unpol = np.histogram(unpol, self.bins)[0]
    hist_pol_err, hist_unpol_err = pol_unpol_hist_err(pol, unpol, pol_err, unpol_err, self.bins)

    # The polarigrams are normalized with the bin width !
    hist_pol_norm = hist_pol / binw
    hist_unpol_norm = hist_unpol / binw
    fit_mod = None
    if 0. in hist_unpol_norm:
      print(f"Unpolarized data do not allow a fit - {dec_plot}_{ra_plot} : a bin is empty")
    else:
      polarigram_error = err_calculation(hist_pol, hist_unpol, binw, hist_pol_err, hist_unpol_err)
      if 0. in polarigram_error:
        print(f"Polarized data do not allow a fit - {dec_plot}_{ra_plot} : a bin is empty leading to uncorrect fit")
      else:
        histo = hist_pol_norm / hist_unpol_norm * np.mean(hist_unpol_norm)
        fit_mod = Fit(modulation_func, var_x, histo, yerr=polarigram_error, comment="modulation")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
    ax1.stairs(hist_pol_norm, self.bins, label="Polarized histogram")
    ax1.set(xlabel=r"\eta'")
    ax1.legend()
    ax2.stairs(hist_unpol_norm, self.bins, label="Unpolarized histogram")
    ax2.set(xlabel=r"\eta'")
    ax2.legend()
    ax3.stairs(histo, self.bins, label="Unpolarized histogram")
    ax3.plot(np.linspace(-180, 180, 100, dtype=np.float32), modulation_func(np.linspace(-180, 180, 100, dtype=np.float32), *fit_mod.popt), label="Fitted modulation")
    ax3.set(xlabel=r"\eta'")
    ax3.legend()
    plt.show()


class Mu100Data:
  """
  Class containing the data for 1 GRB, for 1 sim, and 1 satellite
  """
  def __init__(self, data):
    """
    :param data: lines of data in a list extracted from condensed data file
    """
    ##############################################################
    # Attributes filled with file reading
    lines = data.split("\n")
    self.dec = float(lines[0])
    self.ra = float(lines[1])
    self.mu100 = float(lines[2])
    self.mu100_err = float(lines[3])
    self.pa = float(lines[4])
    self.pa_err = float(lines[5])
    # Set with the fit or for the fit
    # =0 : fit perfectly
    # ~1 : fit reasonably
    # >1 : not a good fit
    # >>1 : very poor fit
    self.fit_goodness = float(lines[6])
    self.s_eff_compton = float(lines[7])
    self.s_eff_single = float(lines[8])

  @staticmethod
  def get_keys():
    print("======================================================================")
    print("    Attributes")
    print(" mu100 for the satellite/constellation                                    .mu100")
    print(" Polarization angle obtained from the polarigram                          .pa")
    print(" Polarization angle error from the fit                                    .pa_err")
    print(" mu100 error from the fit                                                 .mu100_err")
    print(" Fit goodness                                                             .fit_goodness")
