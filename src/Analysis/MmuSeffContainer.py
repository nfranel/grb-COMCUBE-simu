# Autor Nathan Franel
# Date 01/12/2023
# Version 2 :
# Separating the code in different modules

# Package imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
# from scipy.stats import chi2
from time import time
import os.path

# Developped modules imports
from src.General.funcmod import read_mupar, use_scipyquad, set_bins, band, pol_unpol_hist_err, err_calculation, modulation_func, get_pol_unpol_event_data, compile_finder
from src.Launchers.launch_mu100_sim import make_ra_list
from src.Analysis.MFit import Fit

# Ploting adjustments
# mpl.use('Qt5Agg')
# mpl.use('TkAgg')

# plt.rcParams.update({'font.size': 20})


class MuSeffContainer:
  """
  Class containing the information for mu100 files
  """

  def __init__(self, mu100parfile, ergcut=(100, 460), armcut=180):
    """
    :param mu100parfile: mu100 parameter file
    :param ergcut: energy cut to apply
    :param armcut: ARM cut to apply
    """
    self.array_dtype = np.float32
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
    self.armcut = armcut
    self.fluence = use_scipyquad(band, self.ergcut[0], self.ergcut[1], func_args=tuple(bandparam), x_logscale=True)[0] * self.poltime
    geom_name = geom.split(".geo.setup")[0].split("/")[-1]

    saving = f"mu-seff-saved_{geom_name}_{self.decs[0]:.0f}-{self.decs[1]:.0f}-{self.decs[2]:.0f}_{self.ras[0]:.0f}-{self.ras[1]:.0f}-{self.ras[2]:.0f}.h5"
    cond_saving = f"cond_mu-seff-saved_{geom_name}_{self.decs[0]:.0f}-{self.decs[1]:.0f}-{self.decs[2]:.0f}_{self.ras[0]:.0f}-{self.ras[1]:.0f}-{self.ras[2]:.0f}_ergcut-{ergcut[0]}-{ergcut[1]}_armcut-{armcut}.h5"
    if cond_saving not in os.listdir(f"../Data/mu100/sim_{geom_name}"):
      if saving not in os.listdir(f"../Data/mu100/sim_{geom_name}"):
        init_time = time()
        print("###########################################################################")
        print(" mu/Seff data not saved : Saving ")
        print("###########################################################################")
        compile_finder()
        self.save_fulldata(f"../Data/mu100/sim_{geom_name}/{saving}", f"../Data/mu100/sim_{geom_name}/{cond_saving}")
        print("=======================================")
        print(" Saving of mu/Seff data finished in : ", time() - init_time, "seconds")
        print("=======================================")
      else:
        init_time = time()
        print("###########################################################################")
        print(" mu/Seff condensed data not saved : Saving ")
        print("###########################################################################")
        self.save_condensed_data(f"../Data/mu100/sim_{geom_name}/{saving}", f"../Data/mu100/sim_{geom_name}/{cond_saving}")
        print("=======================================")
        print(" Saving of mu/Seff data finished in : ", time() - init_time, "seconds")
        print("=======================================")
    init_time = time()
    print("###########################################################################")
    print(" Extraction of mu/Seff data ")
    print("###########################################################################")
    # list.__init__(self, self.read_data(f"../Data/mu100/sim_{geom_name}/{cond_saving}"))
    self.mudf = self.read_data(f"../Data/mu100/sim_{geom_name}/{cond_saving}")
    print(self.mudf.index)
    self.mudf.sort_values(by=["dec", "ra"], ascending=[True, True], inplace=True)
    print(self.mudf.index)

    print("=======================================")
    print(" Extraction of mu/Seff data finished in : ", time() - init_time, "seconds")
    print("=======================================")

  def save_fulldata(self, file, condensed_file):
    """
    Function used to save the mu100/seff data into a txt file
    :param file: path of the file to save full data
    :param condensed_file: path of the file to save condensed data
    """
    with pd.HDFStore(file, mode="w") as f:
      data_tab = []
      for ite_dec, dec in enumerate(np.linspace(self.decs[0], self.decs[1], self.decs[2])):
        for ite_ra, ra in enumerate(make_ra_list(self.ras, dec)):
          #  The commented parts are the ones that may not be useful
          geom_name = self.geometry.split(".geo.setup")[0].split("/")[-1]
          polname = f"../Data/mu100/sim_{geom_name}/sim/mu100_{dec:.1f}_{ra:.1f}pol.inc1.id1.extracted.tra"
          unpolname = f"../Data/mu100/sim_{geom_name}/sim/mu100_{dec:.1f}_{ra:.1f}unpol.inc1.id1.extracted.tra"
          if not (os.path.exists(polname) and os.path.exists(unpolname)):
            raise FileNotFoundError("Polarized or unpolarized file is not found")

          dec_err, ra_err = 1.12, 1.01
          compton_ener_pol, pol, pol_err, arm_pol, single_ener_pol, compton_ener_unpol, unpol, unpol_err, arm_unpol = get_pol_unpol_event_data(polname, unpolname, dec, ra, dec_err, ra_err, self.geometry, self.array_dtype)

          df_compton_pol = pd.DataFrame({"compton_ener_pol": compton_ener_pol, "pol": pol, "pol_err": pol_err, "arm_pol": arm_pol})
          df_compton_unpol = pd.DataFrame({"compton_ener_unpol": compton_ener_unpol, "unpol": unpol, "unpol_err": unpol_err, "arm_unpol": arm_unpol})

          key = f"itedec{ite_dec}_itera{ite_ra}"
          # Saving Compton event related quantities
          f.put(f"{key}/compton_pol", df_compton_pol)
          f.put(f"{key}/compton_unpol", df_compton_unpol)
          # Saving single event related quantities
          f.put(f"{key}/single_ener", pd.Series(single_ener_pol))
          # Saving scalar values
          # Specific to satellite
          f.get_storer(f"{key}/compton_pol").attrs.dec = dec
          f.get_storer(f"{key}/compton_pol").attrs.ra = ra

          if self.ergcut is not None:
            df_compton_pol = df_compton_pol[(df_compton_pol.compton_ener_pol >= self.ergcut[0]) & (df_compton_pol.compton_ener_pol <= self.ergcut[1])]
            df_compton_unpol = df_compton_unpol[(df_compton_unpol.compton_ener_unpol >= self.ergcut[0]) & (df_compton_unpol.compton_ener_unpol <= self.ergcut[1])]
            single_ener_pol = single_ener_pol[(single_ener_pol >= self.ergcut[0]) & (single_ener_pol <= self.ergcut[1])]

          if self.armcut is not None:
            df_compton_pol = df_compton_pol[df_compton_pol.arm_pol <= self.armcut]
            df_compton_unpol = df_compton_unpol[df_compton_unpol.arm_unpol <= self.armcut]

          hist_pol = np.histogram(df_compton_pol.pol, self.bins)[0]
          hist_unpol = np.histogram(df_compton_unpol.unpol, self.bins)[0]
          hist_pol_err, hist_unpol_err = pol_unpol_hist_err(df_compton_pol.pol, df_compton_unpol.unpol, df_compton_pol.pol_err, df_compton_unpol.unpol_err, self.bins)

          var_x = .5 * (self.bins[1:] + self.bins[:-1])
          binw = self.bins[1:] - self.bins[:-1]

          # The polarigrams are normalized with the bin width !
          hist_pol_norm = hist_pol / binw
          hist_unpol_norm = hist_unpol / binw
          fit_mod = None
          fit_lin = None
          if 0. in hist_unpol_norm:
            print(f"Unpolarized data do not allow a fit - {dec}_{ra} : a bin is empty")
          else:
            polarigram_error = err_calculation(hist_pol, hist_unpol, binw, hist_pol_err, hist_unpol_err)
            if 0. in polarigram_error:
              print(f"Polarized data do not allow a fit - {dec}_{ra} : a bin is empty leading to uncorrect fit")
            else:
              histo = hist_pol_norm / hist_unpol_norm * np.mean(hist_unpol_norm)
              fit_mod = Fit(modulation_func, var_x, histo, yerr=polarigram_error, comment="modulation")
              # fit_lin = Fit(lambda x, a:a*x/x, var_x, histo, yerr=polarigram_error, comment="constant")
          pa, mu100 = fit_mod.popt[:2]
          if mu100 < 0:
            pa = (pa + 90) % 180
            mu100 = - mu100
          else:
            pa = pa % 180
          pa_err = np.sqrt(fit_mod.pcov[0][0])
          mu100_err = np.sqrt(fit_mod.pcov[1][1])
          fit_p_value = fit_mod.p_value

          seff_compton = len(df_compton_pol) / self.fluence
          seff_single = len(single_ener_pol) / self.fluence

          data_tab.append([dec, ra, mu100, mu100_err, pa, pa_err, fit_p_value, seff_compton, seff_single])
      f.get_storer(f"{key}/compton_pol").attrs.description = f"# File containing mu100 and Seff data for : \n# Geometry : {self.geometry}\n# Revan file : {self.revanfile}\n# Mimrec file : {self.mimrecfile}\n# Polarized simulation time : {self.poltime}\n# Unpolarized simulation time : {self.unpoltime}\n# dec min-max-number of value : {self.decs[0]}-{self.decs[1]}-{self.decs[2]}\n# ra min-max-number of value (at equator) : {self.ras[0]}-{self.ras[1]}-{self.ras[2]}"
      f.get_storer(f"{key}/compton_pol").attrs.structure = "Keys : dec-ra/compton_pol or compton_unpol DataFrames or single_ener Serie"
    columns = ["dec", "ra", "mu100", "mu100_err", "pa", "pa_err", "fit_p_value", "seff_compton", "seff_single"]
    cond_df = pd.DataFrame(data=data_tab, columns=columns)

    with pd.HDFStore(condensed_file, mode="w") as fcond:
      fcond.put(f"museff_df", cond_df)
      fcond.get_storer("museff_df").attrs.description = f"# File containing mu100 and Seff data for : \n# Geometry : {self.geometry}\n# Revan file : {self.revanfile}\n# Mimrec file : {self.mimrecfile}\n# Polarized simulation time : {self.poltime}\n# Unpolarized simulation time : {self.unpoltime}\n# dec min-max-number of value : {self.decs[0]}-{self.decs[1]}-{self.decs[2]}\n# ra min-max-number of value (at equator) : {self.ras[0]}-{self.ras[1]}-{self.ras[2]}"
      fcond.get_storer("museff_df").attrs.structure = "Keys : mu100 and Seff DataFrame"
      fcond.get_storer("museff_df").attrs.ergcut = f"energy cut : {self.ergcut[0]}-{self.ergcut[1]}"
      fcond.get_storer("museff_df").attrs.armcut = f"ARM cut : {self.armcut}"

  def save_condensed_data(self, fullfile, condensed_file):
    """
    Function used to save the mu100/seff data into a txt file
    :param fullfile: path of the file where full data is saved
    :param condensed_file: path of the file to save condensed data
    """
    var_x = .5 * (self.bins[1:] + self.bins[:-1])
    binw = self.bins[1:] - self.bins[:-1]

    data_tab = []
    with pd.HDFStore(fullfile, mode="r") as f:
      for key in set(k.split("/")[1] for k in f.keys()):
        dec = f.get_storer(f"{key}/compton_pol").attrs.dec
        ra = f.get_storer(f"{key}/compton_pol").attrs.ra
        df_compton_pol = f[f"{key}/compton_pol"]
        df_compton_unpol = f[f"{key}/compton_unpol"]
        single_ener_pol = f[f"{key}/single_ener"]

        if self.ergcut is not None:
          df_compton_pol = df_compton_pol[(df_compton_pol.compton_ener_pol >= self.ergcut[0]) & (df_compton_pol.compton_ener_pol <= self.ergcut[1])]
          df_compton_unpol = df_compton_unpol[(df_compton_unpol.compton_ener_unpol >= self.ergcut[0]) & (df_compton_unpol.compton_ener_unpol <= self.ergcut[1])]
          single_ener_pol = single_ener_pol[(single_ener_pol >= self.ergcut[0]) & (single_ener_pol <= self.ergcut[1])]

        if self.armcut is not None:
          df_compton_pol = df_compton_pol[df_compton_pol.arm_pol <= self.armcut]
          df_compton_unpol = df_compton_unpol[df_compton_unpol.arm_unpol <= self.armcut]

        hist_pol = np.histogram(df_compton_pol.pol, self.bins)[0]
        hist_unpol = np.histogram(df_compton_unpol.unpol, self.bins)[0]
        hist_pol_err, hist_unpol_err = pol_unpol_hist_err(df_compton_pol.pol, df_compton_unpol.unpol, df_compton_pol.pol_err, df_compton_unpol.unpol_err, self.bins)

        # The polarigrams are normalized with the bin width !
        hist_pol_norm = hist_pol / binw
        hist_unpol_norm = hist_unpol / binw
        fit_mod = None
        fit_lin = None
        if 0. in hist_unpol_norm:
          print(f"Unpolarized data do not allow a fit - {dec}_{ra} : a bin is empty")
        else:
          polarigram_error = err_calculation(hist_pol, hist_unpol, binw, hist_pol_err, hist_unpol_err)
          if 0. in polarigram_error:
            print(f"Polarized data do not allow a fit - {dec}_{ra} : a bin is empty leading to uncorrect fit")
          else:
            histo = hist_pol_norm / hist_unpol_norm * np.mean(hist_unpol_norm)
            fit_mod = Fit(modulation_func, var_x, histo, yerr=polarigram_error, comment="modulation")
            # fit_lin = Fit(lambda x, a:a*x/x, var_x, histo, yerr=polarigram_error, comment="constant")
        pa, mu100 = fit_mod.popt[:2]
        if mu100 < 0:
          pa = (pa + 90) % 180
          mu100 = - mu100
        else:
          pa = pa % 180
        pa_err = np.sqrt(fit_mod.pcov[0][0])
        mu100_err = np.sqrt(fit_mod.pcov[1][1])
        fit_p_value = fit_mod.p_value

        seff_compton = len(df_compton_pol) / self.fluence
        seff_single = len(single_ener_pol) / self.fluence
        # Writing the condensed file
        data_tab.append([dec, ra, mu100, mu100_err, pa, pa_err, fit_p_value, seff_compton, seff_single])

    columns = ["dec", "ra", "mu100", "mu100_err", "pa", "pa_err", "fit_p_value", "seff_compton", "seff_single"]
    cond_df = pd.DataFrame(data=data_tab, columns=columns)

    with pd.HDFStore(condensed_file, mode="w") as fcond:
      fcond.put(f"museff_df", cond_df)
      fcond.get_storer("museff_df").attrs.description = f"# File containing mu100 and Seff data for : \n# Geometry : {self.geometry}\n# Revan file : {self.revanfile}\n# Mimrec file : {self.mimrecfile}\n# Polarized simulation time : {self.poltime}\n# Unpolarized simulation time : {self.unpoltime}\n# dec min-max-number of value : {self.decs[0]}-{self.decs[1]}-{self.decs[2]}\n# ra min-max-number of value (at equator) : {self.ras[0]}-{self.ras[1]}-{self.ras[2]}"
      fcond.get_storer("museff_df").attrs.structure = "Keys : mu100 and Seff DataFrame"
      fcond.get_storer("museff_df").attrs.ergcut = f"energy cut : {self.ergcut[0]}-{self.ergcut[1]}"
      fcond.get_storer("museff_df").attrs.armcut = f"ARM cut : {self.armcut}"

  def read_data(self, condensed_file):
    """
    Function used to read the bkg txt file
    :param file: path of the file where condensed data is saved
    """
    with pd.HDFStore(condensed_file, mode="r") as fcond:
      return fcond["museff_df"]

  def show_fit(self, dec_plot, ra_plot):

    geom_name = self.geometry.split(".geo.setup")[0].split("/")[-1]
    polname = f"../Data/mu100/sim_{geom_name}/sim/mu100_{dec_plot:.1f}_{ra_plot:.1f}pol.inc1.id1.extracted.tra"
    unpolname = f"../Data/mu100/sim_{geom_name}/sim/mu100_{dec_plot:.1f}_{ra_plot:.1f}unpol.inc1.id1.extracted.tra"
    if not (os.path.exists(polname) and os.path.exists(unpolname)):
      raise FileNotFoundError("Polarized or unpolarized file is not found")

    dec_err, ra_err = 1.12, 1.01
    compton_ener_pol, pol, pol_err, arm_pol, single_ener_pol, compton_ener_unpol, unpol, unpol_err, arm_unpol = get_pol_unpol_event_data(polname, unpolname, dec_plot, ra_plot, dec_err, ra_err, self.geometry, self.array_dtype)

    df_compton_pol = pd.DataFrame({"compton_ener_pol": compton_ener_pol, "pol": pol, "pol_err": pol_err, "arm_pol": arm_pol})
    df_compton_unpol = pd.DataFrame({"compton_ener_unpol": compton_ener_unpol, "unpol": unpol, "unpol_err": unpol_err, "arm_punol": arm_unpol})

    if self.ergcut is not None:
      df_compton_pol = df_compton_pol[(df_compton_pol.compton_ener_pol >= self.ergcut[0]) & (df_compton_pol.compton_ener_pol <= self.ergcut[1])]
      df_compton_unpol = df_compton_unpol[(df_compton_unpol.compton_ener_unpol >= self.ergcut[0]) & (df_compton_unpol.compton_ener_unpol <= self.ergcut[1])]
      single_ener_pol = single_ener_pol[(single_ener_pol >= self.ergcut[0]) & (single_ener_pol <= self.ergcut[1])]

    if self.armcut is not None:
      df_compton_pol = df_compton_pol[df_compton_pol.arm_pol <= self.armcut]
      df_compton_unpol = df_compton_unpol[df_compton_unpol.arm_unpol <= self.armcut]

    hist_pol = np.histogram(df_compton_pol.pol, self.bins)[0]
    hist_unpol = np.histogram(df_compton_unpol.unpol, self.bins)[0]
    hist_pol_err, hist_unpol_err = pol_unpol_hist_err(df_compton_pol.pol, df_compton_unpol.unpol, df_compton_pol.pol_err, df_compton_unpol.unpol_err, self.bins)

    var_x = .5 * (self.bins[1:] + self.bins[:-1])
    binw = self.bins[1:] - self.bins[:-1]

    # The polarigrams are normalized with the bin width !
    hist_pol_norm = hist_pol / binw
    hist_unpol_norm = hist_unpol / binw
    fit_mod = None
    fit_lin = None
    if 0. in hist_unpol_norm:
      print(f"Unpolarized data do not allow a fit - {dec_plot}_{ra_plot} : a bin is empty")
    else:
      polarigram_error = err_calculation(hist_pol, hist_unpol, binw, hist_pol_err, hist_unpol_err)
      if 0. in polarigram_error:
        print(f"Polarized data do not allow a fit - {dec_plot}_{ra_plot} : a bin is empty leading to uncorrect fit")
      else:
        histo = hist_pol_norm / hist_unpol_norm * np.mean(hist_unpol_norm)
        fit_mod = Fit(modulation_func, var_x, histo, yerr=polarigram_error, comment="modulation")
        # fit_lin = Fit(lambda x, a:a*x/x, var_x, histo, yerr=polarigram_error, comment="constant")

    mpl.use('Qt5Agg')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    ax1.stairs(hist_pol_norm, self.bins, label="Initial histogram")
    ax1.set(xlabel=r"$\eta$'")
    ax1.legend()
    ax2.stairs(hist_unpol_norm, self.bins, label="Unpolarized histogram")
    ax2.set(xlabel=r"$\eta$'")
    ax2.legend()
    ax3.stairs(histo, self.bins, label="Corrected histogram")
    ax3.plot(np.linspace(-180, 180, 100, dtype=np.float32), modulation_func(np.linspace(-180, 180, 100, dtype=np.float32), *fit_mod.popt), label="Fitted modulation")
    ax3.set(xlabel=r"$\eta$'")
    ax3.legend()
    plt.tight_layout()
    plt.show()
