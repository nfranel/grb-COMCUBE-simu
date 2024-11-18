from MAllSourceData import AllSourceData
import seaborn as sns
import pandas as pd
from time import time
from funcmod import *
from visualisation import *
from MCGRB import GRBSample
from catalog import SampleCatalog
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs


init_time = time()
# grb_sim_param = "/pdisk/ESA/Sim-article/COMCUBEv134-550km-C120/polGBM.par"
# grb_sim_param = "/pdisk/ESA/Sim-article/COMCUBEv134-550km-C30/polGBM.par"
# grb_sim_param = "/pdisk/ESA/Sim-article/COMCUBEv134-550km-C60/polGBM.par"

# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat--lc-sampledv4/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--45-45-45--27sat--lc-sampledv3/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--97.5-97.5-97.5--27sat--lc-sampledv3/polGBM.par"

# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat--lc-all/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--45-45-45--27sat--lc-all/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--97.5-97.5-97.5--27sat--lc-all/polGBM.par"

# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat--long/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-45-97.5--27sat--long/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--5-5-45--27sat--long/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-45-97.5--36sat--long/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat--short/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-45-97.5--27sat--short/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--5-5-45--27sat--short/polGBM.par"

grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat--lc-sampled/polGBM.par"

bkg_param = "./bkg/bkg-v15.par"
mu_param = "./mu100/mu100-v15.par"
# bkg_param = "./bkg/bkg-v134.par"
# mu_param = "./mu100/mu100-v134.par"
erg = (10, 1000)
# erg = (30, 1000)
arm = 180
test = AllSourceData(grb_sim_param, bkg_param, mu_param, erg, arm, parallel="all")
test.make_const()
test.analyze()
print("=======================================")
print("processing time : ", time()-init_time, "seconds")
print("=======================================")


from memory_profiler import memory_usage


print("==========================================================================")
print("=                        Now printing some results                        ")
print("==========================================================================")
print("=                          analyze informations                         =")
print(f"     param file : {grb_sim_param}")
print(f"     erg cut : {erg}")
print(f"     arm cut : {arm}")
print("=                          duty cycle                         =")
print(f"     constellation : {test.com_duty}")
inclinations = []
for info in test.sat_info:
  if info[0] not in inclinations:
    inclinations.append(info[0])
for inc in inclinations:
  print(f"     inclination {inc} : {calc_duty(inc, 0, 0, 500)}")
mdp30list = []
for ite_const, num_down in enumerate(test.number_of_down_per_const):
  print(f"================== Number of satellite down : {ite_const} ==================")

  print("=                        triggers count                        =")
  test.count_triggers(const_index=ite_const, graphs=False)
  # print("=                        MDP histogram                        =")
  # test.mdp_histogram()
  number_detected = 0
  mdp_list = []
  pflux_inf_10_l = []
  pflux_inf_30_l = []
  pflux_inf_50_l = []
  mflux_inf_10_l = []
  mflux_inf_30_l = []
  mflux_inf_50_l = []
  flnc_inf_10_l = []
  flnc_inf_30_l = []
  flnc_inf_50_l = []
  pfluxes_l = []
  mfluxes_l = []
  flnces_l = []

  pflux_inf_10_s = []
  pflux_inf_30_s = []
  pflux_inf_50_s = []
  mflux_inf_10_s = []
  mflux_inf_30_s = []
  mflux_inf_50_s = []
  flnc_inf_10_s = []
  flnc_inf_30_s = []
  flnc_inf_50_s = []
  pfluxes_s = []
  mfluxes_s = []
  flnces_s = []
  for source in test.alldata:
    if source is not None:
      if source.source_duration > 2:
        if source.best_fit_p_flux is not None:
          pfluxes_l.append(source.best_fit_p_flux)
          mfluxes_l.append(source.best_fit_mean_flux)
          flnces_l.append(source.source_fluence)
      else:
        if source.best_fit_p_flux is not None:
          pfluxes_s.append(source.best_fit_p_flux)
          mfluxes_s.append(source.best_fit_mean_flux)
          flnces_s.append(source.source_fluence)
      for sim in source:
        if sim is not None:
          if sim.const_data is not None:
            number_detected += 1
            if sim.const_data[ite_const] is not None:
              if sim.const_data[ite_const].mdp is not None:
                if sim.const_data[ite_const].mdp <= 1:
                  mdp_list.append(sim.const_data[ite_const].mdp * 100)
                  if source.source_duration > 2:
                    if sim.const_data[ite_const].mdp <= 0.1:
                      pflux_inf_10_l.append(source.best_fit_p_flux)
                      mflux_inf_10_l.append(source.best_fit_mean_flux)
                      flnc_inf_10_l.append(source.source_fluence)
                    elif sim.const_data[ite_const].mdp <= 0.3:
                      pflux_inf_30_l.append(source.best_fit_p_flux)
                      mflux_inf_30_l.append(source.best_fit_mean_flux)
                      flnc_inf_30_l.append(source.source_fluence)
                    elif sim.const_data[ite_const].mdp <= 0.5:
                      pflux_inf_50_l.append(source.best_fit_p_flux)
                      mflux_inf_50_l.append(source.best_fit_mean_flux)
                      flnc_inf_50_l.append(source.source_fluence)
                  else:
                    if sim.const_data[ite_const].mdp <= 0.1:
                      pflux_inf_10_s.append(source.best_fit_p_flux)
                      mflux_inf_10_s.append(source.best_fit_mean_flux)
                      flnc_inf_10_s.append(source.source_fluence)
                    elif sim.const_data[ite_const].mdp <= 0.3:
                      pflux_inf_30_s.append(source.best_fit_p_flux)
                      mflux_inf_30_s.append(source.best_fit_mean_flux)
                      flnc_inf_30_s.append(source.source_fluence)
                    elif sim.const_data[ite_const].mdp <= 0.5:
                      pflux_inf_50_s.append(source.best_fit_p_flux)
                      mflux_inf_50_s.append(source.best_fit_mean_flux)
                      flnc_inf_50_s.append(source.source_fluence)

  mdp_list = np.array(mdp_list)
  mdp30list.append(np.sum(np.where(mdp_list <= 30, 1, 0)) * test.weights)
  print(f" ========               MDP THRESHOLD USED : {2.6}   ========")
  print("=                        MDP detection rates                        =")
  print(f"   MDP<=80% : {np.sum(np.where(mdp_list <= 80, 1, 0)) * test.weights}")
  print(f"   MDP<=50% : {np.sum(np.where(mdp_list <= 50, 1, 0)) * test.weights}")
  print(f"   MDP<=30% : {np.sum(np.where(mdp_list <= 30, 1, 0)) * test.weights}")
  print(f"   MDP<=10% : {np.sum(np.where(mdp_list <= 10, 1, 0)) * test.weights}")

# # Searching for a good binning for the MCMC condition for a good sample :
# binning = np.logspace(-1, 4, 50)
# pflux_inf_10_hist = np.histogram(pflux_inf_10, bins=binning)
# mflux_inf_10_hist = np.histogram(mflux_inf_10, bins=binning)
# pflux_inf_30_hist = np.histogram(pflux_inf_30, bins=binning)
# mflux_inf_30_hist = np.histogram(mflux_inf_30, bins=binning)
# pflux_inf_50_hist = np.histogram(pflux_inf_50, bins=binning)
# mflux_inf_50_hist = np.histogram(mflux_inf_50, bins=binning)
#
# print(f"==== MDP < 10 ==")
# print(f"== Peakflux for GRB with MDP < 10 ==")
# for ite in range(len(pflux_inf_10_hist[0])):
#   print(f"Bin from {pflux_inf_10_hist[1][ite]:8.4f} to {pflux_inf_10_hist[1][ite+1]:8.4f}  : {pflux_inf_10_hist[0][ite]}")
# print(f"== Meanflux for GRB with MDP < 10 ==")
# for ite in range(len(mflux_inf_10_hist[0])):
#   print(f"Bin from {mflux_inf_10_hist[1][ite]:8.4f} to {mflux_inf_10_hist[1][ite+1]:8.4f}  : {mflux_inf_10_hist[0][ite]}")
#
# print(f"==== MDP < 30 ==")
# print(f"== Peakflux for GRB with MDP < 30 ==")
# for ite in range(len(pflux_inf_30_hist[0])):
#   print(f"Bin from {pflux_inf_30_hist[1][ite]:8.4f} to {pflux_inf_30_hist[1][ite+1]:8.4f}  : {pflux_inf_30_hist[0][ite]}")
# print(f"== Meanflux for GRB with MDP < 30 ==")
# for ite in range(len(mflux_inf_30_hist[0])):
#   print(f"Bin from {mflux_inf_30_hist[1][ite]:8.4f} to {mflux_inf_30_hist[1][ite+1]:8.4f}  : {mflux_inf_30_hist[0][ite]}")
#
# print(f"==== MDP < 30 ==")
# print(f"== Peakflux for GRB with MDP < 30 ==")
# for ite in range(len(pflux_inf_50_hist[0])):
#   print(f"Bin from {pflux_inf_50_hist[1][ite]:8.4f} to {pflux_inf_50_hist[1][ite+1]:8.4f}  : {pflux_inf_50_hist[0][ite]}")
# print(f"== Meanflux for GRB with MDP < 30 ==")
# for ite in range(len(mflux_inf_50_hist[0])):
#   print(f"Bin from {mflux_inf_50_hist[1][ite]:8.4f} to {mflux_inf_50_hist[1][ite+1]:8.4f}  : {mflux_inf_50_hist[0][ite]}")

binning = np.logspace(-1, 4, 50)
print(binning)
yscale = "log"
fig1, ((ax1l, ax2l, ax3l), (ax1s, ax2s, ax3s)) = plt.subplots(nrows=2, ncols=3, figsize=(20, 6))
ax1l.hist(pfluxes_l, bins=binning, histtype="step", label="all_l")
# ax1l.hist(pflux_inf_10_l, bins=binning, histtype="step", label="10_l")
# ax1l.hist(pflux_inf_30_l, bins=binning, histtype="step", label="30_l")
# ax1l.hist(pflux_inf_50_l, bins=binning, histtype="step", label="50_l")
ax1l.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax1l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax1l.legend()

ax2l.hist(mfluxes_l, bins=binning, histtype="step", label="all_l")
# ax2l.hist(mflux_inf_10_l, bins=binning, histtype="step", label="10_l")
# ax2l.hist(mflux_inf_30_l, bins=binning, histtype="step", label="30_l")
# ax2l.hist(mflux_inf_50_l, bins=binning, histtype="step", label="50_l")
ax2l.set(xlabel="mflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax2l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2l.legend()

ax3l.hist(flnces_l, bins=binning, histtype="step", label="all_l")
# ax3l.hist(flnc_inf_10_l, bins=binning, histtype="step", label="10_l")
# ax3l.hist(flnc_inf_30_l, bins=binning, histtype="step", label="30_l")
# ax3l.hist(flnc_inf_50_l, bins=binning, histtype="step", label="50_l")
ax3l.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax3l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax3l.legend()

ax1s.hist(pfluxes_s, bins=binning, histtype="step", label="all_s")
# ax1s.hist(pflux_inf_10_s, bins=binning, histtype="step", label="10_s")
# ax1s.hist(pflux_inf_30_s, bins=binning, histtype="step", label="30_s")
# ax1s.hist(pflux_inf_50_s, bins=binning, histtype="step", label="50_s")
ax1s.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax1s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax1s.legend()

ax2s.hist(mfluxes_s, bins=binning, histtype="step", label="all_s")
# ax2s.hist(mflux_inf_10_s, bins=binning, histtype="step", label="10_s")
# ax2s.hist(mflux_inf_30_s, bins=binning, histtype="step", label="30_s")
# ax2s.hist(mflux_inf_50_s, bins=binning, histtype="step", label="50_s")
ax2s.set(xlabel="mflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax2s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2s.legend()

ax3s.hist(flnces_s, bins=binning, histtype="step", label="all_s")
# ax3s.hist(flnc_inf_10_s, bins=binning, histtype="step", label="10_s")
# ax3s.hist(flnc_inf_30_s, bins=binning, histtype="step", label="30_s")
# ax3s.hist(flnc_inf_50_s, bins=binning, histtype="step", label="50_s")
ax3s.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax3s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax3s.legend()
plt.show()

# import pandas as pd
# dflong = pd.DataFrame({"pflux":pfluxes_l, "mflux":mfluxes_l, "flnc":flnces_l})
# dfshort = pd.DataFrame({"pflux":pfluxes_s, "mflux":mfluxes_s, "flnc":flnces_s})
# dflong.to_csv('data_long.csv', index=False)
# dfshort.to_csv('data_short.csv', index=False)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dfl = pd.read_csv('data_long.csv')
dfs = pd.read_csv('data_short.csv')
pfluxes_l = dfl.pflux.values
mfluxes_l = dfl.mflux.values
flnces_l = dfl.flnc.values
pfluxes_s = dfs.pflux.values
mfluxes_s = dfs.mflux.values
flnces_s = dfs.flnc.values

flux_lim = [20, 100]
flnc_l_lim = [300, 1000]
flnc_s_lim = 10
nfluxbin_l = [30, 6, 5]
nfluxbin_s = [30, 4, 5]
nflncbin_l = [30, 5, 5]
nflncbin_s = [40, 9]
bin_flux_l = np.concatenate((np.logspace(-1, np.log10(flux_lim[0]), nfluxbin_l[0] + 1),
                           np.logspace(np.log10(flux_lim[0]), np.log10(flux_lim[1]), nfluxbin_l[1] + 1)[1:],
                           np.logspace(np.log10(flux_lim[1]), 4, nfluxbin_l[2] + 1)[1:]))
bin_flux_s = np.concatenate((np.logspace(-1, np.log10(flux_lim[0]), nfluxbin_s[0] + 1),
                           np.logspace(np.log10(flux_lim[0]), np.log10(flux_lim[1]), nfluxbin_s[1] + 1)[1:],
                           np.logspace(np.log10(flux_lim[1]), 4, nfluxbin_s[2] + 1)[1:]))
bin_flnc_l = np.concatenate((np.logspace(-1, np.log10(flnc_l_lim[0]), nflncbin_l[0] + 1),
                           np.logspace(np.log10(flnc_l_lim[0]), np.log10(flnc_l_lim[1]), nflncbin_l[1] + 1)[1:],
                           np.logspace(np.log10(flnc_l_lim[1]), 4, nflncbin_l[2] + 1)[1:]))
bin_flnc_s = np.concatenate((np.logspace(-1, np.log10(flnc_s_lim), nflncbin_s[0] + 1),
                           np.logspace(np.log10(flnc_s_lim), 4, nflncbin_s[1] + 1)[1:]))
binning = np.logspace(-1, 4, 50)

print(binning)
print(len(bin_flux_l))
print(len(bin_flux_s))
print(len(bin_flnc_l))
print(len(bin_flnc_s))

yscale = "log"
fig1, ((ax1l, ax2l, ax3l), (ax1s, ax2s, ax3s)) = plt.subplots(nrows=2, ncols=3, figsize=(20, 6))
tt = ax1l.hist(pfluxes_l, bins=bin_flux_l, histtype="step", label="all_l")
ax1l.axvline(flux_lim[0])
ax1l.axvline(flux_lim[1])
ax1l.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax1l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax1l.legend()

ax2l.hist(mfluxes_l, bins=binning, histtype="step", label="all_l")
ax2l.set(xlabel="mflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax2l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2l.legend()

ax3l.hist(flnces_l, bins=bin_flnc_l, histtype="step", label="all_l")
ax3l.axvline(flnc_l_lim[0])
ax3l.axvline(flnc_l_lim[1])
ax3l.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax3l.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax3l.legend()

ax1s.hist(pfluxes_s, bins=bin_flux_s, histtype="step", label="all_s")
ax1s.axvline(flux_lim[0])
ax1s.axvline(flux_lim[1])
ax1s.set(xlabel="pflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax1s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax1s.legend()

ax2s.hist(mfluxes_s, bins=binning, histtype="step", label="all_s")
ax2s.set(xlabel="mflux (ph/cm²/s)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax2s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2s.legend()

ax3s.hist(flnces_s, bins=bin_flnc_s, histtype="step", label="all_s")
ax3s.axvline(flnc_s_lim)
ax3s.set(xlabel="flnc (ph/cm²)", ylabel="Number of GRB", xscale="log", yscale=yscale)
ax3s.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax3s.legend()
plt.show()




fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
ax.plot(range(test.n_sat), mdp30list)
ax.set(title="Evolution of detection rate for GRB with MDP <= 30% with the number of down satellite", xlabel="Number of satellite not working", ylabel="Detection rate of GRB with MDP <= 30% (/yr)")
ax.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
plt.show()











for ite in range(len(cat)):
  if cat.flnc_band_redchisq[ite].strip() != "" and cat.flnc_comp_redchisq[ite].strip() != "" and cat.flnc_sbpl_redchisq[ite].strip() != "" and cat.flnc_plaw_redchisq[ite].strip() != "":
    print(f"Best fit : {cat.flnc_best_fitting_model[ite].strip()},   band : {float(cat.flnc_band_redchisq[ite].strip())},  comp :  {float(cat.flnc_comp_redchisq[ite].strip())}, sbpl : {float(cat.flnc_sbpl_redchisq[ite].strip())},  plaw : {float(cat.flnc_plaw_redchisq[ite].strip())}")


count, countsup, band, band_comp, band_sbpl, band_plaw = 0, 0, 0, 0, 0, 0
bf_band, bf_comp, bf_sbpl, bf_plaw = [], [], [], []
for ite in range(len(cat)):
  if cat.flnc_band_redchisq[ite].strip() != "" and cat.flnc_comp_redchisq[ite].strip() != "" and cat.flnc_sbpl_redchisq[ite].strip() != "" and cat.flnc_plaw_redchisq[ite].strip() != "":
    count += 1
    model = cat.flnc_best_fitting_model[ite].strip()
    ph_flux = float(getattr(cat, f"{model}_phtflux")[ite])
    if ph_flux > 1:
      countsup += 1
      if cat.flnc_best_fitting_model[ite].strip() == "flnc_band":
        band += 1
        bf_band.append(float(cat.flnc_band_redchisq[ite].strip()))
        # print(float(cat.flnc_band_redchisq[ite].strip()))
      elif cat.flnc_best_fitting_model[ite].strip() == "flnc_comp":
        diff = (float(cat.flnc_band_redchisq[ite].strip()) - float(cat.flnc_comp_redchisq[ite].strip())) / float(cat.flnc_comp_redchisq[ite].strip())
        bf_comp.append(float(cat.flnc_comp_redchisq[ite].strip()))
        if diff < 0.05:
          band_comp += 1
      elif cat.flnc_best_fitting_model[ite].strip() == "flnc_sbpl":
        diff = (float(cat.flnc_band_redchisq[ite].strip()) - float(cat.flnc_sbpl_redchisq[ite].strip())) / float(cat.flnc_sbpl_redchisq[ite].strip())
        bf_sbpl.append(float(cat.flnc_sbpl_redchisq[ite].strip()))
        if diff < 0.05:
          band_sbpl += 1
      elif cat.flnc_best_fitting_model[ite].strip() == "flnc_plaw":
        diff = (float(cat.flnc_band_redchisq[ite].strip()) - float(cat.flnc_plaw_redchisq[ite].strip())) / float(cat.flnc_plaw_redchisq[ite].strip())
        bf_plaw.append(float(cat.flnc_plaw_redchisq[ite].strip()))
        if diff < 0.05:
          band_plaw += 1

bins = np.linspace(0, 10, 100)
fig, ax = plt.subplots(1, 1,  figsize=(10, 6))
ax.hist(bf_band, bins=bins, histtype="step", label="band redchi")
ax.hist(bf_comp, bins=bins, histtype="step", label="comp redchi")
ax.hist(bf_sbpl, bins=bins, histtype="step", label="sbpl redchi")
ax.hist(bf_plaw, bins=bins, histtype="step", label="plaw redchi")
ax.legend()
plt.show()








#######################################################################################################################
# Looking for comparable bursts
#######################################################################################################################

from MAllSourceData import AllSourceData
from time import time
from funcmod import *
from visualisation import *
from MCGRB import GRBSample
from catalog import SampleCatalog
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

cat_gbm = Catalog("./GBM/allGBM.txt", [4, '\n', 5, '|', 10000])
gbm_ep = []
gbm_t90 = []
gbm_ph_flux = []
gbm_ph_fluence = []

gbm_ep_l = []
gbm_t90_l = []
gbm_ph_flux_l = []
gbm_ph_fluence_l = []
gbm_name_l = []
gbm_ites_l = []

gbm_ep_s = []
gbm_t90_s = []
gbm_ph_flux_s = []
gbm_ph_fluence_s = []
for ite_gbm, ep_gbm in enumerate(cat_gbm.flnc_band_epeak):
  if ep_gbm.strip() != "":
    ep_temp = float(ep_gbm)
    t90_temp = float(cat_gbm.t90[ite_gbm])
    flux_temp = calc_flux_gbm(cat_gbm, ite_gbm, (10, 1000))
    fluence_temp = flux_temp * t90_temp
    name_temp = cat_gbm.name[ite_gbm]
    gbm_ep.append(ep_temp)
    gbm_t90.append(t90_temp)
    gbm_ph_flux.append(flux_temp)
    gbm_ph_fluence.append(fluence_temp)
    if float(cat_gbm.t90[ite_gbm]) <= 2:
      gbm_ep_s.append(ep_temp)
      gbm_t90_s.append(t90_temp)
      gbm_ph_flux_s.append(flux_temp)
      gbm_ph_fluence_s.append(fluence_temp)
    else:
      gbm_ep_l.append(ep_temp)
      gbm_t90_l.append(t90_temp)
      gbm_ph_flux_l.append(flux_temp)
      gbm_ph_fluence_l.append(fluence_temp)
      gbm_name_l.append(name_temp)
      gbm_ites_l.append(ite_gbm)
  else:
    print("Information : Find null Epeak in catalog")


cat_samp = SampleCatalog("./Sampled/sampled_grb_cat_0.5years.txt", [4, '\n', 5, '|', 10000])
samp_ep = []
samp_t90 = []
samp_ph_flux = []
samp_ph_fluence = []

samp_ep_l = []
samp_t90_l = []
samp_ph_flux_l = []
samp_ph_fluence_l = []
samp_name_l = []
samp_ites_l = []

samp_ep_s = []
samp_t90_s = []
samp_ph_flux_s = []
samp_ph_fluence_s = []
for ite_samp, name_samp in enumerate(cat_samp.df.name):
  samp_ep.append(cat_samp.df.ep_rest[ite_samp])
  samp_t90.append(cat_samp.df.t90[ite_samp])
  samp_ph_flux.append(cat_samp.df.mean_flux[ite_samp])
  samp_ph_fluence.append(cat_samp.df.fluence[ite_samp])
  if name_samp.startswith("sGRB"):
    samp_ep_s.append(cat_samp.df.ep_rest[ite_samp])
    samp_t90_s.append(cat_samp.df.t90[ite_samp])
    samp_ph_flux_s.append(cat_samp.df.mean_flux[ite_samp])
    samp_ph_fluence_s.append(cat_samp.df.fluence[ite_samp])
  else:
    samp_ep_l.append(cat_samp.df.ep_rest[ite_samp])
    samp_t90_l.append(cat_samp.df.t90[ite_samp])
    samp_ph_flux_l.append(cat_samp.df.mean_flux[ite_samp])
    samp_ph_fluence_l.append(cat_samp.df.fluence[ite_samp])
    samp_name_l.append(cat_samp.df.name[ite_samp])
    samp_ites_l.append(ite_samp)


gbm_flu_vals = np.array(gbm_ph_flux_l)
samp_flu_vals = np.array(samp_ph_flux_l)
gbm_t90_vals = np.array(gbm_t90_l)
samp_t90_vals = np.array(samp_t90_l)
gbm_name = np.array(gbm_name_l)
samp_name = np.array(samp_name_l)
gbm_ite = np.array(gbm_ites_l)
samp_ite = np.array(samp_ites_l)
minlim = 10
maxlim = 30
nbins = 50

ns_t90 = samp_t90_vals[np.logical_and(minlim<samp_flu_vals, samp_flu_vals<maxlim)]
ns_flu = samp_flu_vals[np.logical_and(minlim<samp_flu_vals, samp_flu_vals<maxlim)]
ns_name = samp_name[np.logical_and(minlim<samp_flu_vals, samp_flu_vals<maxlim)]
ns_ite = samp_ite[np.logical_and(minlim<samp_flu_vals, samp_flu_vals<maxlim)]
ng_t90 = gbm_t90_vals[np.logical_and(minlim<gbm_flu_vals, gbm_flu_vals<maxlim)]
ng_flu = gbm_flu_vals[np.logical_and(minlim<gbm_flu_vals, gbm_flu_vals<maxlim)]
ng_name = gbm_name[np.logical_and(minlim<gbm_flu_vals, gbm_flu_vals<maxlim)]
ng_ite = gbm_ite[np.logical_and(minlim<gbm_flu_vals, gbm_flu_vals<maxlim)]

plt.rcParams.update({'font.size': 13})
bins=np.logspace(np.log10(np.min(ns_flu)), np.log10(np.max(ns_flu)), nbins)
# bins = np.logspace(-8, 3, 40)
fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
ax1.hist(ns_flu, bins=bins, histtype="step", weights=[1/0.5] * len(ns_flu), color="darkblue", label="Sample")
ax1.hist(ng_flu, bins=bins, histtype="step", weights=[1/10] * len(ng_flu), color="red", label="GBM")
ax1.set(title="Flux histograms", xlabel="Photon flux (ph/cm²/s)", ylabel="Number of bursts", xscale="log", yscale="log")
ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax1.legend(loc='upper left')

ax2.hist(ns_t90, bins=nbins, histtype="step", weights=[1/0.5] * len(ns_t90), color="darkblue", label="Sample")
ax2.hist(ng_t90, bins=nbins, histtype="step", weights=[1/10] * len(ng_t90), color="red", label="GBM")
ax2.set(title="T90 histograms", xlabel="T90 (s)", ylabel="Number of bursts", xscale="linear", yscale="log")
ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2.legend(loc='upper left')
plt.show()


#######################################################################################################################
# Sample comparison
#######################################################################################################################
cat_gbm = Catalog("./GBM/allGBM.txt", test.sttype, "GBM/rest_frame_properties.txt")
cat_smp = SampleCatalog("./Sampled/sampled_grb_cat_5years.txt", test.sttype)

gbm_ep = []
gbm_t90 = []
gbm_ph_flux = []
gbm_ph_fluence = []

gbm_ep_l = []
gbm_t90_l = []
gbm_ph_flux_l = []
gbm_ph_fluence_l = []

gbm_ep_s = []
gbm_t90_s = []
gbm_ph_flux_s = []
gbm_ph_fluence_s = []
for ite_gbm, ep_gbm in enumerate(cat_gbm.df.flnc_band_epeak):
  if ep_gbm != "":
    ep_temp = ep_gbm
    t90_temp = cat_gbm.df.t90[ite_gbm]
    flux_temp = calc_flux_gbm(cat_gbm, ite_gbm, (10, 1000))
    fluence_temp = flux_temp * t90_temp
    gbm_ep.append(ep_temp)
    gbm_t90.append(t90_temp)
    gbm_ph_flux.append(flux_temp)
    gbm_ph_fluence.append(fluence_temp)
    if t90_temp <= 2:
      gbm_ep_s.append(ep_temp)
      gbm_t90_s.append(t90_temp)
      gbm_ph_flux_s.append(flux_temp)
      gbm_ph_fluence_s.append(fluence_temp)
    else:
      gbm_ep_l.append(ep_temp)
      gbm_t90_l.append(t90_temp)
      gbm_ph_flux_l.append(flux_temp)
      gbm_ph_fluence_l.append(fluence_temp)
  else:
    print("Information : Find null Epeak in catalog")

smp_ep = []
smp_t90 = []
smp_ph_flux = []
smp_ph_fluence = []

smp_ep_l = []
smp_t90_l = []
smp_ph_flux_l = []
smp_ph_fluence_l = []

smp_ep_s = []
smp_t90_s = []
smp_ph_flux_s = []
smp_ph_fluence_s = []
for ite_smp, name_smp in enumerate(cat_smp.df.name):
  smp_ep.append(cat_smp.df.ep_rest[ite_smp])
  smp_t90.append(cat_smp.df.t90[ite_smp])
  smp_ph_flux.append(cat_smp.df.mean_flux[ite_smp])
  smp_ph_fluence.append(cat_smp.df.fluence[ite_smp])
  if name_smp.startswith("sGRB"):
    smp_ep_s.append(cat_smp.df.ep_rest[ite_smp])
    smp_t90_s.append(cat_smp.df.t90[ite_smp])
    smp_ph_flux_s.append(cat_smp.df.mean_flux[ite_smp])
    smp_ph_fluence_s.append(cat_smp.df.fluence[ite_smp])
  else:
    smp_ep_l.append(cat_smp.df.ep_rest[ite_smp])
    smp_t90_l.append(cat_smp.df.t90[ite_smp])
    smp_ph_flux_l.append(cat_smp.df.mean_flux[ite_smp])
    smp_ph_fluence_l.append(cat_smp.df.fluence[ite_smp])

number_off_sat = 0
print("================================================================================================")
print(f"== Triggers according to GBM method with   {number_off_sat}   down satellite")
print("================================================================================================")
total_in_view = 0
const_trigger_counter_4s = 0
const_trigger_counter_3s = 0
const_trigger_counter_2s = 0
const_trigger_counter_1s = 0
trig_name = []
trig_ep = []
trig_t90 = []
trig_ph_flux = []
trig_ph_fluence = []

trig_ep_l = []
trig_t90_l = []
trig_ph_flux_l = []
trig_ph_fluence_l = []

trig_ep_s = []
trig_t90_s = []
trig_ph_flux_s = []
trig_ph_fluence_s = []
for source in test.alldata:
  if source is not None:
    for ite_sim, sim in enumerate(source):
      if sim is not None:
        total_in_view += 1
        if sim.const_data[number_off_sat] is not None:
          if True in (sim.const_data[number_off_sat].const_beneficial_trigger_4s >= 4):
            const_trigger_counter_4s += 1
          if True in (sim.const_data[number_off_sat].const_beneficial_trigger_3s >= 3):
            const_trigger_counter_3s += 1
            for smp_ite, smp_name in enumerate(cat_smp.df.name):
              if smp_name == source.source_name:
                temp_ep = cat_smp.df.ep_rest[smp_ite]
                temp_t90 = cat_smp.df.t90[smp_ite]
                temp_flux = cat_smp.df.mean_flux[smp_ite]
                temp_fluence = cat_smp.df.fluence[smp_ite]
                trig_ep.append(temp_ep)
                trig_t90.append(temp_t90)
                trig_ph_flux.append(temp_flux)
                trig_ph_fluence.append(temp_fluence)
            if source.source_name.startswith("sGRB"):
              trig_ep_s.append(temp_ep)
              trig_t90_s.append(temp_t90)
              trig_ph_flux_s.append(temp_flux)
              trig_ph_fluence_s.append(temp_fluence)
            else:
              trig_ep_l.append(temp_ep)
              trig_t90_l.append(temp_t90)
              trig_ph_flux_l.append(temp_flux)
              trig_ph_fluence_l.append(temp_fluence)
          if True in (sim.const_data[number_off_sat].const_beneficial_trigger_2s >= 2):
            const_trigger_counter_2s += 1
          if True in (sim.const_data[number_off_sat].const_beneficial_trigger_1s >= 1):
            const_trigger_counter_1s += 1


print(f"   Trigger for at least 4 satellites :        {const_trigger_counter_4s:.2f} triggers")
print(f"   Trigger for at least 3 satellites :        {const_trigger_counter_3s:.2f} triggers")
print(f"   Trigger for at least 2 satellites :        {const_trigger_counter_2s:.2f} triggers")
print(f"   Trigger for 1 satellite :        {const_trigger_counter_1s:.2f} triggers")
print("=============================================")
print(f" Over the {total_in_view} GRBs simulated in the constellation field of view")

gbmcorrec = 1 / 0.587 / 10
min_ph_flux, max_ph_flux  = -6, 2

plt.rcParams.update({'font.size': 13})
bins_ph_flux = np.logspace(min_ph_flux, max_ph_flux, 50)
# bins = np.logspace(-8, 3, 40)
fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(27, 6))
ax1.hist(smp_ph_flux, bins=bins_ph_flux, histtype="step", weights=[test.weights] * len(smp_ph_flux), color="darkblue", label="Sample")
ax1.hist(trig_ph_flux, bins=bins_ph_flux, histtype="step", weights=[test.weights] * len(trig_ph_flux), linestyle='--', color="lime", label="Detected")
ax1.hist(gbm_ph_flux, bins=bins_ph_flux, histtype="step", weights=[gbmcorrec] * len(gbm_ph_flux), color="red", label="GBM")
ax1.set(title="Flux histograms", xlabel="Photon flux (ph/cm²/s)", ylabel="Number of GRB/year", xscale="log", yscale="log")
ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax1.legend(loc='upper left')

ax2.hist(smp_ph_flux_s, bins=bins_ph_flux, histtype="step", weights=[test.weights] * len(smp_ph_flux_s), color="darkblue", label="Sample")
trig = ax2.hist(trig_ph_flux_s, bins=bins_ph_flux, histtype="step", weights=[test.weights] * len(trig_ph_flux_s), linestyle='--', color="lime", label="Detected")
gbm = ax2.hist(gbm_ph_flux_s, bins=bins_ph_flux, histtype="step", weights=[gbmcorrec] * len(gbm_ph_flux_s), color="red", label="GBM")
ax2.set(title="sGRB flux histograms", xlabel="Photon flux (ph/cm²/s)", ylabel="Number of GRB/year", xscale="log", yscale="log")
ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2.legend(loc='upper left')

ax3.hist(smp_ph_flux_l, bins=bins_ph_flux, histtype="step", weights=[test.weights] * len(smp_ph_flux_l), color="darkblue", label="Sample")
ax3.hist(trig_ph_flux_l, bins=bins_ph_flux, histtype="step", weights=[test.weights] * len(trig_ph_flux_l), linestyle='--', color="lime", label="Detected")
ax3.hist(gbm_ph_flux_l, bins=bins_ph_flux, histtype="step", weights=[gbmcorrec] * len(gbm_ph_flux_l), color="red", label="GBM")
ax3.set(title="lGRB flux histograms", xlabel="Photon flux (ph/cm²/s)", ylabel="Number of GRB/year", xscale="log", yscale="log")
ax3.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax3.legend(loc='upper left')
plt.show()

bins_flnc = np.logspace(-6, 4, 40)
fig2, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(27, 6))
ax1.hist(smp_ph_fluence, bins=bins_flnc, histtype="step", weights=[test.weights] * len(smp_ph_fluence), color="blue", label="Sample")
ax1.hist(trig_ph_fluence, bins=bins_flnc, histtype="step", weights=[test.weights] * len(trig_ph_fluence), color="green", label="Detected")
ax1.hist(gbm_ph_fluence, bins=bins_flnc, histtype="step", weights=[gbmcorrec] * len(gbm_ph_fluence), color="red", label="GBM")
ax1.set(title="Fluence histograms", xlabel="Photon fluence (ph/cm²)", ylabel="Number of GRB/year", xscale="log", yscale="log")
ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax1.legend()

ax2.hist(smp_ph_fluence_s, bins=bins_flnc, histtype="step", weights=[test.weights] * len(smp_ph_fluence_s), color="blue", label="Sample")
ax2.hist(trig_ph_fluence_s, bins=bins_flnc, histtype="step", weights=[test.weights] * len(trig_ph_fluence_s), color="green", label="Detected")
ax2.hist(gbm_ph_fluence_s, bins=bins_flnc, histtype="step", weights=[gbmcorrec] * len(gbm_ph_fluence_s), color="red", label="GBM")
ax2.set(title="sGRB fluence histograms", xlabel="Photon fluence (ph/cm²)", ylabel="Number of GRB/year", xscale="log", yscale="log")
ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2.legend()

ax3.hist(smp_ph_fluence_l, bins=bins_flnc, histtype="step", weights=[test.weights] * len(smp_ph_fluence_l), color="blue", label="Sample")
ax3.hist(trig_ph_fluence_l, bins=bins_flnc, histtype="step", weights=[test.weights] * len(trig_ph_fluence_l), color="green", label="Detected")
ax3.hist(gbm_ph_fluence_l, bins=bins_flnc, histtype="step", weights=[gbmcorrec] * len(gbm_ph_fluence_l), color="red", label="GBM")
ax3.set(title="lGRB fluence histograms", xlabel="Photon fluence (ph/cm²)", ylabel="Number of GRB/year", xscale="log", yscale="log")
ax3.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax3.legend()
plt.show()

bins_t90 = np.logspace(-3, 3, 40)
fig3, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(27, 6))
ax1.hist(smp_t90, bins=bins_t90, histtype="step", weights=[test.weights] * len(smp_t90), color="blue", label="Sample")
ax1.hist(trig_t90, bins=bins_t90, histtype="step", weights=[test.weights] * len(trig_t90), color="green", label="Detected")
ax1.hist(gbm_t90, bins=bins_t90, histtype="step", weights=[gbmcorrec] * len(gbm_t90), color="red", label="GBM")
ax1.set(title="T90 histograms", xlabel="T90 (s)", ylabel="Number of GRB/year", xscale="log", yscale="log")
ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax1.legend()

ax2.hist(smp_t90_s, bins=bins_t90, histtype="step", weights=[test.weights] * len(smp_t90_s), color="blue", label="Sample")
ax2.hist(trig_t90_s, bins=bins_t90, histtype="step", weights=[test.weights] * len(trig_t90_s), color="green", label="Detected")
ax2.hist(gbm_t90_s, bins=bins_t90, histtype="step", weights=[gbmcorrec] * len(gbm_t90_s), color="red", label="GBM")
ax2.set(title="sGRB T90 histograms", xlabel="T90 (s)", ylabel="Number of GRB/year", xscale="log", yscale="log")
ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2.legend()

ax3.hist(smp_t90_l, bins=bins_t90, histtype="step", weights=[test.weights] * len(smp_t90_l), color="blue", label="Sample")
ax3.hist(trig_t90_l, bins=bins_t90, histtype="step", weights=[test.weights] * len(trig_t90_l), color="green", label="Detected")
ax3.hist(gbm_t90_l, bins=bins_t90, histtype="step", weights=[gbmcorrec] * len(gbm_t90_l), color="red", label="GBM")
ax3.set(title="lGRB T90 histograms", xlabel="T90 (s)", ylabel="Number of GRB/year", xscale="log", yscale="log")
ax3.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax3.legend()
plt.show()


Fast test
cat_gbm = Catalog("./GBM/allGBM.txt", [4, '\n', 5, '|', 10000])
gbm_ep = []
gbm_t90 = []
gbm_ph_flux = []
gbm_ph_fluence = []

gbm_ep_l = []
gbm_t90_l = []
gbm_ph_flux_l = []
gbm_ph_fluence_l = []

gbm_ep_s = []
gbm_t90_s = []
gbm_ph_flux_s = []
gbm_ph_fluence_s = []
for ite_gbm, ep_gbm in enumerate(cat_gbm.flnc_band_epeak):
  if ep_gbm.strip() != "":
    ep_temp = float(ep_gbm)
    t90_temp = float(cat_gbm.t90[ite_gbm])
    flux_temp = calc_flux_gbm(cat_gbm, ite_gbm, (10, 1000))
    fluence_temp = flux_temp * t90_temp
    gbm_ep.append(ep_temp)
    gbm_t90.append(t90_temp)
    gbm_ph_flux.append(flux_temp)
    gbm_ph_fluence.append(fluence_temp)
    if float(cat_gbm.t90[ite_gbm]) <= 2:
      gbm_ep_s.append(ep_temp)
      gbm_t90_s.append(t90_temp)
      gbm_ph_flux_s.append(flux_temp)
      gbm_ph_fluence_s.append(fluence_temp)
    else:
      gbm_ep_l.append(ep_temp)
      gbm_t90_l.append(t90_temp)
      gbm_ph_flux_l.append(flux_temp)
      gbm_ph_fluence_l.append(fluence_temp)
  else:
    print("Information : Find null Epeak in catalog")

mflu = []
flu = []
tim = []
for source in test.alldata:
  mflu.append(source.best_fit_mean_flux)
  flu.append(source.source_fluence)
  tim.append(source.source_duration)

bins = np.logspace(-7, 4, 50)
fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(27, 6))
ax1.hist(gbm_ph_fluence, bins=bins, histtype="step", weights=[1/10] * len(gbm_ph_fluence))
ax1.hist(flu, bins=bins, histtype="step")
ax1.set(title="Fluence histograms", xlabel="Photon fluence (ph/cm²)", ylabel="Number of bursts", xscale="log", yscale="log")
ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)

ax2.hist(gbm_ph_flux, bins=bins, histtype="step", weights=[1/10] * len(gbm_ph_flux))
ax2.hist(mflu, bins=bins, histtype="step")
ax2.set(title="sGRB Fluence histograms", xlabel="Photon fluence (ph/cm²)", ylabel="Number of short bursts", xscale="log", yscale="log")
ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)

bins = np.linspace(0, 400, 50)
ax3.hist(gbm_t90, bins=bins, histtype="step", weights=[1/10] * len(gbm_t90))
ax3.hist(tim, bins=bins, histtype="step")
ax3.set(title="lGRB Fluence histograms", xlabel="Photon fluence (ph/cm²)", ylabel="Number of long bursts", xscale="log", yscale="log")
ax3.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
plt.show()


# not_trig = ["GRB081130212", "GRB090418816", "GRB100101028", "GRB100411516", "GRB141102112", "GRB150828901", "GRB180602938"]

print("=                       Constellation's Seff histograms (for different GRBs)                       ")
const_seff_compton = []
const_seff_single = []
const_b_compton_rate = []
const_b_single_rate = []
for ite in range(len(test.alldata)):
  source = test.alldata[ite]
  if source is not None:
    sim_finder = 0
    while source[sim_finder] is None:
      sim_finder += 1
    sim = source[sim_finder]
    if sim.const_data is not None:
      const_seff_compton.append(sim.const_data.s_eff_compton)
      const_seff_single.append(sim.const_data.s_eff_single)
      const_b_compton_rate.append(sim.const_data.compton_b_rate)
      const_b_single_rate.append(sim.const_data.single_b_rate)

fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
nbins1 = 30
ax1.hist(const_seff_compton, bins=nbins1)
ax2.hist(const_seff_single, bins=nbins1)
ax1.set(title="constellation's compton seff histograms", xlabel="Seff (cm²)")
ax2.set(title="constellation's single seff histograms", xlabel="Seff (cm²)")
ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
plt.show()

print("=                       background rates (for different detecting constellations)                       ")
fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
nbins2 = 20
ax1.hist(const_b_compton_rate, bins=nbins2)
ax2.hist(const_b_single_rate, bins=nbins2)
ax1.set(title="constellation's compton background rate histograms", xlabel="count rate (count/s)")
ax2.set(title="constellation's single background rate histograms", xlabel="count rate (count/s)")
ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
plt.show()


print("==========================================================================")
print("=                           Doing some analysis                           ")
print("==========================================================================")
mdp_thresh_list_coarse = [np.inf, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400]
test.study_mdp_threshold(mdp_thresh_list_coarse, savefile=f"mdp_threshold_test_coarse_{erg[0]}-{erg[1]}")
mdp_thresh_list_fine = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                        3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
test.study_mdp_threshold(mdp_thresh_list_fine, savefile=f"mdp_threshold_test_fine_{erg[0]}-{erg[1]}")


print("==========================================================================")
print("=                       Time resolved mdp analysis                        ")
print("==========================================================================")
high_mdp_grb = ["GRB090902462", "GRB101014175", "GRB101123952", "GRB120711115", "GRB130306991", "GRB130427324",
                "GRB130609902", "GRB150330828", "GRB151231443", "GRB160509374", "GRB160625945", "GRB171010792",
                "GRB171227000"]
# time_separations = [[6, 12, 17.5, "end"],
#                     []]
time_compton_list = []
ite_const = 0
list_b_rate = []
list_mu = []
for source in test.alldata:
  if source is not None and source.source_name in high_mdp_grb:
    for sim in source:
      if sim is not None:
        if sim.const_data is not None:
          if sim.const_data[ite_const].mdp is not None:
            print(f"Source with MDP < 10% : {source.source_name}, duration : {source.source_duration}, dec : {sim.dec_world_frame}, fluence :{source.source_energy_fluence}")
            time_compton_list.append(sim.const_data[ite_const].compton_time)
            list_b_rate.append(sim.const_data[ite_const].compton_b_rate)
            list_mu.append(sim.const_data[ite_const].mu100_ref)

for lc_index in range(len(time_compton_list)):
  fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
  ax1.hist(time_compton_list[lc_index], bins=np.arange(0, np.max(time_compton_list[lc_index])+0.5, 0.5))
  ax1.set(title="Light curve", xlabel="Time (s)")
  ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
  plt.show()

time_intervals = [[(3.8, 9.25), (9.25, 14.75), (14.75, 19.6)],
                  [(0, 16), (16, 40), (100, 120), (155, 165), (196, 203), (203, 225), (420, 450)],
                  [(0, 17), (44, 57), (100, 105)],
                  [(0, 13.5), (21.5, 45)],
                  [(12.5, 30), (30, 63), (63, 75)],
                  [(0, 11), (11, 16)],
                  [(0, 10), (10, 25), (168, 191)],
                  [(0, 4), (120, 138), (138, 150)],
                  [(0, 10), (58, 65), (65, 72)],
                  [(0, 25), (353, 370)],
                  [(0, 34), (450, 454)],
                  [(0, 32.5), (32.5, 68)],
                  [(7, 17), (17, 23)]]

for grb_ite, grb in enumerate(high_mdp_grb):
  print("=======================================================================================================")
  print(f" Source {grb}, {len(time_intervals[grb_ite])} peaks studied")
  print("=======================================================================================================")
  temp_lcs = []
  for interval in time_intervals[grb_ite]:
    inter_count = []
    for val_time in time_compton_list[grb_ite]:
      if interval[0] < val_time < interval[1]:
        inter_count.append(val_time)
    temp_lcs.append(inter_count)
  fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
  ax1.hist(time_compton_list[grb_ite], bins=np.arange(0, np.max(time_compton_list[grb_ite])+0.5, 0.5))
  ax1.set(title=f"Light curve {high_mdp_grb[grb_ite]}", xlabel="Time (s)", ylabel="Number of hits (counts)")
  ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
  ax2 = ax1.twinx()
  ax2.set(ylabel="MDP (%)")

  for ite_interv, interval in enumerate(time_intervals[grb_ite]):
    interv_duration = interval[1] - interval[0]
    mdp_temp = calc_mdp(len(temp_lcs[ite_interv]), list_b_rate[grb_ite] * interv_duration, list_mu[grb_ite])
    print(f"   Peak {ite_interv + 1} : MDP = {mdp_temp * 100} %")
    ax2.plot([interval[0], interval[1]], [mdp_temp * 100, mdp_temp * 100], color="red")
  plt.show()

# plt.rcParams.update({'font.size': 15})
high_mdp_select = ["GRB120711115", "GRB130306991", "GRB130427324", "GRB130609902"]
time_intervals_select = [[(0, 13.5), (21.5, 45)],
                         [(12.5, 30), (30, 63), (63, 75)],
                         [(0, 11), (11, 16)],
                         [(0, 10), (10, 25), (168, 191)]]
time_compton_list = []
ite_const = 0
list_b_rate = []
list_mu = []
for source in test.alldata:
  if source is not None and source.source_name in high_mdp_select:
    for sim in source:
      if sim is not None:
        if sim.const_data is not None:
          if sim.const_data[ite_const].mdp is not None:
            print(f"Source with MDP < 10% : {source.source_name}, duration : {source.source_duration}, dec : {sim.dec_world_frame}, fluence :{source.source_energy_fluence}")
            time_compton_list.append(sim.const_data[ite_const].compton_time)
            list_b_rate.append(sim.const_data[ite_const].compton_b_rate)
            list_mu.append(sim.const_data[ite_const].mu100_ref)

for grb_ite, grb in enumerate(high_mdp_select):
  print("=======================================================================================================")
  print(f" Source {grb}, {len(time_intervals_select[grb_ite])} peaks studied")
  print("=======================================================================================================")
  temp_lcs = []
  for interval in time_intervals_select[grb_ite]:
    inter_count = []
    for val_time in time_compton_list[grb_ite]:
      if interval[0] < val_time < interval[1]:
        inter_count.append(val_time)
    temp_lcs.append(inter_count)
  fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
  #
  ax1.hist(np.concatenate((time_compton_list[grb_ite],time_compton_list[grb_ite])), bins=np.arange(0, np.max(time_compton_list[grb_ite])+0.5, 0.5))
  if grb_ite == 2:
    ax1.set(title=f"Light curve {high_mdp_select[grb_ite]}", xlabel="Time (s)", ylabel="Count rate (counts/s)", xlim=(0, 40))
  else:
    ax1.set(title=f"Light curve {high_mdp_select[grb_ite]}", xlabel="Time (s)", ylabel="Count rate (counts/s)")
  ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
  ax2 = ax1.twinx()
  ax2.set(ylabel="MDP (%)")
  ax2.spines['right'].set_color('red')  # Réglage de la couleur du bord droit
  ax2.yaxis.label.set_color('red')  # Réglage de la couleur du label de l'axe y
  ax2.yaxis.set_tick_params(colors='red')
  ax2.set_ylim(0, 50)

  for ite_interv, interval in enumerate(time_intervals_select[grb_ite]):
    interv_duration = interval[1] - interval[0]
    mdp_temp = calc_mdp(len(temp_lcs[ite_interv]), list_b_rate[grb_ite] * interv_duration, list_mu[grb_ite])
    print(f"   Peak {ite_interv + 1} : MDP = {mdp_temp * 100} %")
    ax2.plot([interval[0], interval[1]], [mdp_temp * 100, mdp_temp * 100], color="red")
  plt.show()



print("==========================================================================")
print("=                       Testing the working of sims                       ")
print("==========================================================================")
testlist = [8, 37, 169, 364, 987, 1245, 1543, 1830]
phi_sat = np.linspace(0, 360, 181)
theta_sat = np.linspace(0, 114, 115)

nrows = len(theta_sat)
ncols = len(phi_sat)
mu100list = np.zeros((nrows, ncols))
seff_com_list = np.zeros((nrows, ncols))
seff_sin_list = np.zeros((nrows, ncols))
for i, theta in enumerate(theta_sat):
  for j, phi in enumerate(phi_sat):
    file = closest_mufile(theta, phi, test.muSeffdata)
    mu100list[i, j] = file[0]
    seff_com_list[i, j] = file[2]
    seff_sin_list[i, j] = file[3]
# smoothing the values
smooth_mu100list = np.zeros((nrows, ncols))
v2smooth_mu100list = np.zeros((nrows, ncols))
smooth_seff_com_list = np.zeros((nrows, ncols))
smooth_seff_sin_list = np.zeros((nrows, ncols))
mu100_vs_dec = np.mean(mu100list, axis=1)
seff_com_vs_dec = np.mean(seff_com_list, axis=1)
seff_sin_vs_dec = np.mean(seff_sin_list, axis=1)
for i, theta in enumerate(theta_sat):
  for j, phi in enumerate(phi_sat):
    file = closest_mufile(theta, phi, test.muSeffdata)
    smooth_mu100list[i, j] = (mu100list[i, np.mod(j - 1, ncols)] + mu100list[i, j] + mu100list[i, np.mod(j + 1, ncols)]) / 3
    v2smooth_mu100list[i, j] = (mu100list[i, np.mod(j - 2, ncols)] + mu100list[i, np.mod(j - 1, ncols)] + mu100list[i, j] + mu100list[i, np.mod(j + 1, ncols)] + mu100list[i, np.mod(j + 2, ncols)]) / 5
    smooth_seff_com_list[i, j] = (seff_com_list[i, np.mod(j - 1, ncols)] + seff_com_list[i, j] + seff_com_list[i, np.mod(j + 1, ncols)]) / 3
    smooth_seff_sin_list[i, j] = (seff_sin_list[i, np.mod(j - 1, ncols)] + seff_sin_list[i, j] + seff_sin_list[i, np.mod(j + 1, ncols)]) / 3


x_mu, y_mu = np.meshgrid(phi_sat, 90 - theta_sat)

print("=                       Seff values                       ")
for ite in testlist:
  source = test.alldata[ite]
  if source is not None:
    sim_finder = 0
    while source[sim_finder] is None:
      sim_finder += 1
    sim = source[sim_finder]
    if sim.const_data is not None:
      print("=  Seff values for the const : ")
      print(f"    const    compton : {sim.const_data.s_eff_compton:.6f}     single : {sim.const_data.s_eff_single:.6f}")
      print("=  Seff values for the satellitess : ")
      for sat_ite, sat in enumerate(sim):
        if sat is not None:
          print(f"    sat {sat_ite:2d}   compton : {sat.s_eff_compton:.6f}     single : {sat.s_eff_single:.6f}")

# Normal maps
print("=                       Seff values from files                       ")
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
p3 = ax.pcolormesh(x_mu, y_mu, seff_com_list, cmap="Blues", transform=ccrs.PlateCarree())
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Compton Seff map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
cbar = fig.colorbar(p3, ax=ax)
cbar.set_label("Compton Seff (cm²)", rotation=270, labelpad=20)
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
p4 = ax.pcolormesh(x_mu, y_mu, seff_sin_list, cmap="Blues", transform=ccrs.PlateCarree())
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Single Seff map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
cbar = fig.colorbar(p4, ax=ax)
cbar.set_label("Single Seff (cm²)", rotation=270, labelpad=20)
plt.show()

# Smoothed maps
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
p3 = ax.pcolormesh(x_mu, y_mu, smooth_seff_com_list, cmap="Blues", transform=ccrs.PlateCarree())
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Smoothed Compton Seff map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
cbar = fig.colorbar(p3, ax=ax)
cbar.set_label("Compton Seff (cm²)", rotation=270, labelpad=20)
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
p4 = ax.pcolormesh(x_mu, y_mu, smooth_seff_sin_list, cmap="Blues", transform=ccrs.PlateCarree())
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Smoothed single Seff map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
cbar = fig.colorbar(p4, ax=ax)
cbar.set_label("Single Seff (cm²)", rotation=270, labelpad=20)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_sat, seff_com_vs_dec)
# ysep = (max(seff_com_vs_dec)-min(seff_com_vs_dec))/15
# ax.set(xticks=np.arange(min(theta_sat), max(theta_sat)+5, 5), yticks=np.arange(min(seff_com_vs_dec), max(seff_com_vs_dec)+ysep, ysep), title="Variation of duty cycle with inclination", xlabel=None, ylabel="Inclination (°)")
ax.set(title="Variation of Compton Seff with declination", xlabel="Declination (°)", ylabel="Compton effective area (cm²)")
ax.grid()
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_sat, seff_sin_vs_dec)
ax.set(title="Variation of single Seff with declination", xlabel="Declination (°)", ylabel="Single effective area (cm²)")
ax.grid()
ax.legend()
plt.show()


print("=                       mu100 values                       ")
for ite in testlist:
  source = test.alldata[ite]
  if source is not None:
    sim_finder = 0
    while source[sim_finder] is None:
      sim_finder += 1
    sim = source[sim_finder]
    if sim.const_data is not None:
      print("=  mu100 values for the const : ")
      print(f"    const    : {sim.const_data.mu100_ref:.6f}")
      print("=  mu100 values for the satellitess : ")
      for sat_ite, sat in enumerate(sim):
        if sat is not None:
          print(f"    sat {sat_ite:2d}   : {sat.mu100_ref:.6f},      dec and ra : {sat.grb_dec_sat_frame:.6f}, {sat.grb_ra_sat_frame:.6f}        len(pol) : {len(sat.pol)}")

print("=                       mu100 values from files                       ")
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
p2 = ax.pcolormesh(x_mu, y_mu, mu100list, cmap="Blues", transform=ccrs.PlateCarree())
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="mu100 map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
cbar = fig.colorbar(p2, ax=ax)
cbar.set_label("mu100 values", rotation=270, labelpad=20)
plt.show()

# Smoothed maps
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
p2 = ax.pcolormesh(x_mu, y_mu, smooth_mu100list, cmap="Blues", transform=ccrs.PlateCarree())
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Smoothed mu100 map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
cbar = fig.colorbar(p2, ax=ax)
cbar.set_label("mu100 values", rotation=270, labelpad=20)
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
p2 = ax.pcolormesh(x_mu, y_mu, smooth_mu100list, cmap="Blues", transform=ccrs.PlateCarree())
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Smoothed mu100 map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
cbar = fig.colorbar(p2, ax=ax)
cbar.set_label("mu100 values", rotation=270, labelpad=20)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_sat, mu100_vs_dec)
ax.set(title="Variation of mu100 with declination", xlabel="Declination (°)", ylabel="Mu100 value (cm²)")
ax.grid()
ax.legend()
plt.show()


print("=                       bkg values                       ")
cr_compton_list = []
cr_single_list = []
pos_lat = []
pos_lon = []
for ite in testlist:
  source = test.alldata[ite]
  if source is not None:
    sim_finder = 0
    while source[sim_finder] is None:
      sim_finder += 1
    sim = source[sim_finder]
    for sat_ite, sat in enumerate(sim):
      if sat is not None:
        cr_compton_list.append(sat.compton_b_rate)
        cr_single_list.append(sat.single_b_rate)
        pos_lat.append(90 - sat.sat_dec_wf)
        pos_lon.append(np.mod(sat.sat_ra_wf, 360))

dec_range = np.linspace(0, 180, 181)
ra_range = np.linspace(0, 360, 361)
x_long, y_lat = np.meshgrid(ra_range, 90 - dec_range)
field_list = np.zeros((len(dec_range), len(ra_range)))
apex15 = Apex(date=2025)
item_legend = "compton events count rate (counts/s)"
field_index = 0
for row, dec in enumerate(dec_range):
  for col, ra in enumerate(ra_range):
    lat = 90 - dec
    # Geodetic to apex, scalar input
    mag_lat, mag_lon = apex15.convert(lat, ra, 'geo', 'apex', height=500)
    # print(f"init : {lat:.12f}, {ra:.12f}              final : {mag_lat:.12f}, {mag_lon:.12f}")
    mag_dec, mag_ra = 90 - mag_lat, mag_lon
    bkg_values = closest_bkg_values(mag_dec, mag_ra, 500, test.bkgdata)
    field_list[row][col] = bkg_values[field_index]

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, figsize=(10, 6))
p1 = ax.pcolormesh(x_long, y_lat, field_list, cmap="Blues")
psat = ax.scatter(pos_lon, pos_lat, c=cr_compton_list, cmap="Blues", transform=ccrs.PlateCarree(), norm=p1.norm, edgecolors='black')
ax.coastlines()
ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title=f"Background map for {item_legend} at {500}km \n and the satellites background rate")
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
cbar = fig.colorbar(p1, ax=ax)
cbar.set_label(f"Background {item_legend}", rotation=270, labelpad=20)
plt.show()
# bkg_data_map("compton_cr", test.bkgdata, 500, dec_range=np.linspace(0, 180, 181), ra_range=np.linspace(0, 360, 361))

dec_range = np.linspace(0, 180, 181)
ra_range = np.linspace(0, 360, 361)
x_long, y_lat = np.meshgrid(ra_range, 90 - dec_range)
field_list = np.zeros((len(dec_range), len(ra_range)))
apex15 = Apex(date=2025)
item_legend = "single events count rate (counts/s)"
field_index = 1
for row, dec in enumerate(dec_range):
  for col, ra in enumerate(ra_range):
    lat = 90 - dec
    # Geodetic to apex, scalar input
    mag_lat, mag_lon = apex15.convert(lat, ra, 'geo', 'apex', height=500)
    # print(f"init : {lat:.12f}, {ra:.12f}              final : {mag_lat:.12f}, {mag_lon:.12f}")
    mag_dec, mag_ra = 90 - mag_lat, mag_lon
    bkg_values = closest_bkg_values(mag_dec, mag_ra, 500, test.bkgdata)
    field_list[row][col] = bkg_values[field_index]

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, figsize=(10, 6))
p1 = ax.pcolormesh(x_long, y_lat, field_list, cmap="Blues")
psat = ax.scatter(pos_lon, pos_lat, c=cr_single_list, cmap="Blues", transform=ccrs.PlateCarree(), norm=p1.norm, edgecolors='black')
ax.coastlines()
ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title=f"Background map for {item_legend} at {500}km \n and the satellites background rate")
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
cbar = fig.colorbar(p1, ax=ax)
cbar.set_label(f"Background {item_legend}", rotation=270, labelpad=20)
plt.show()
# bkg_data_map("single_cr", test.bkgdata, 500, dec_range=np.linspace(0, 180, 181), ra_range=np.linspace(0, 360, 361))




######################################################################################################################
# Make some GRB samples
######################################################################################################################
sample_test = GRBSample()
# sample_test.short_comparison()
sample_test.long_distri()
sample_test.short_distri()




# Search for an mdp limit :
for threshold_mdp in [np.inf, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50 , 100, 200, 300, 400]:
  test.set_beneficial(threshold_mdp)
  test.make_const()
  test.analyze()
  print(f" ========               MDP THRESHOLD USED : {threshold_mdp}   ========")
  print("=                        MDP histogram                        =")
  # test.mdp_histogram()
  number_detected = 0
  mdp_list = []
  for source in test.alldata:
    if source is not None:
      for sim in source:
        if sim is not None:
          if sim.const_data is not None:
            number_detected += 1
            if sim.const_data[0].mdp is not None:
              if sim.const_data[0].mdp <= 1:
                mdp_list.append(sim.const_data[0].mdp * 100)
  mdp_list = np.array(mdp_list)
  print("=                        MDP detection rates                        =")
  print(f"   MDP<=80% : {np.sum(np.where(mdp_list<=80, 1, 0)) * test.weights}")
  print(f"   MDP<=50% : {np.sum(np.where(mdp_list<=50, 1, 0)) * test.weights}")
  print(f"   MDP<=30% : {np.sum(np.where(mdp_list<=30, 1, 0)) * test.weights}")
  print(f"   MDP<=10% : {np.sum(np.where(mdp_list<=10, 1, 0)) * test.weights}")


from MAllSourceData import AllSourceData
from MBkgContainer import BkgContainer
from MmuSeffContainer import MuSeffContainer
from time import time
from funcmod import *
from visualisation import *
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

bkg_param = "./bkg/bkg-v15.par"
mu_param = "./mu100/mu100-v15.par"
# ergmin = [10, 30, 40, 50, 60, 70, 80, 90, 100]
# ergmax = [300, 460, 600, 800, 1000]
ergmin = [10, 30, 60, 100]
ergmax = [460, 1000]
for emin in ergmin:
  for emax in ergmax:
    ergcut = (emin, emax)
    armcut = 180
    bkgdata = BkgContainer(bkg_param, True, ergcut)
    muSeffdata = MuSeffContainer(mu_param, ergcut, armcut)
    del bkgdata
    del muSeffdata


def results_func(emin, emax):
  grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat--long/polGBM.par"
  # grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat--short/polGBM.par"
  bkg_param = "./bkg/bkg-v15.par"
  mu_param = "./mu100/mu100-v15.par"
  erg = (emin, emax)
  arm = 180
  test = AllSourceData(grb_sim_param, bkg_param, mu_param, erg, arm, parallel="all")
  test.make_const()
  test.verif_const()
  test.analyze()
  test.verif_const()
  number_detected = 0
  mdp_list = []
  srn_list = []
  for source in test.alldata:
    if source is not None:
      for sim in source:
        if sim is not None:
          if sim.const_data is not None:
            number_detected += 1
            if sim.const_data.mdp is not None:
              if sim.const_data.mdp <= 1:
                mdp_list.append(sim.const_data.mdp * 100)
              if sim.const_data.snr_single >= 0:
                srn_list.append(sim.const_data.snr_single)
  mdp_list = np.array(mdp_list)
  srn_list = np.array(srn_list)
  r1 = erg
  r2 = np.sum(np.where(mdp_list <= 10, 1, 0)) * test.weights
  r3 = np.sum(np.where(mdp_list <= 30, 1, 0)) * test.weights
  r4 = np.sum(np.where(mdp_list <= 50, 1, 0)) * test.weights
  r5 = np.sum(np.where(mdp_list <= 80, 1, 0)) * test.weights
  r6 = np.sum(np.where(srn_list >= 3, 1, 0)) * test.weights
  r7 = np.sum(np.where(srn_list >= 5, 1, 0)) * test.weights
  r8 = np.sum(np.where(srn_list >= 7, 1, 0)) * test.weights
  r9 = np.sum(np.where(srn_list >= 10, 1, 0)) * test.weights
  r10 = np.sum(np.where(srn_list >= 15, 1, 0)) * test.weights
  r11 = np.sum(np.where(srn_list >= 30, 1, 0)) * test.weights
  r12 = np.sum(np.where(srn_list >= 50, 1, 0)) * test.weights
  r13 = np.sum(np.where(srn_list >= 70, 1, 0)) * test.weights
  r14 = np.sum(np.where(srn_list >= 100, 1, 0)) * test.weights
  r15 = np.sum(np.where(srn_list >= 200, 1, 0)) * test.weights
  r16 = np.sum(np.where(srn_list >= 300, 1, 0)) * test.weights
  r17 = np.sum(np.where(srn_list >= 600, 1, 0)) * test.weights
  return [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17]

bkg_param = "./bkg/bkg-v15.par"
mu_param = "./mu100/mu100-v15.par"
# ergmin = [10, 30, 40, 50, 60, 70, 80, 90, 100]
# ergmax = [300, 460, 600, 800, 1000]
ergmin = [10, 30, 60, 100]
ergmax = [460, 1000]
erglist = []
mdp_10_list = []
mdp_30_list = []
mdp_50_list = []
mdp_80_list = []
snr_3_list = []
snr_5_list = []
snr_7_list = []
snr_10_list = []
snr_15_list = []
snr_30_list = []
snr_50_list = []
snr_70_list = []
snr_100_list = []
snr_200_list = []
snr_300_list = []
snr_600_list = []
for emin in ergmin:
  for emax in ergmax:
    # init_time = time()
    # grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat--long/polGBM.par"
    # # grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat--short/polGBM.par"
    # bkg_param = "./bkg/bkg-v15.par"
    # mu_param = "./mu100/mu100-v15.par"
    # erg = (emin, emax)
    # arm = 180
    # test = AllSourceData(grb_sim_param, bkg_param, mu_param, erg, arm, parallel="all")
    # test.make_const()
    # test.verif_const()
    # test.analyze()
    # test.verif_const()
    # print("=======================================")
    # print("processing time : ", time() - init_time, "seconds")
    # print("=======================================")
    #
    # number_detected = 0
    # mdp_list = []
    # srn_list = []
    # for source in test.alldata:
    #   if source is not None:
    #     for sim in source:
    #       if sim is not None:
    #         if sim.const_data is not None:
    #           number_detected += 1
    #           if sim.const_data.mdp is not None:
    #             if sim.const_data.mdp <= 1:
    #               mdp_list.append(sim.const_data.mdp * 100)
    #             if sim.const_data.snr_single >= 0:
    #               srn_list.append(sim.const_data.snr_single)
    # mdp_list = np.array(mdp_list)
    # srn_list = np.array(srn_list)
    # erglist.append(erg)

    ret = results_func(emin, emax)
    erglist.append(ret[0])
    mdp_10_list.append(ret[1])
    mdp_30_list.append(ret[2])
    mdp_50_list.append(ret[3])
    mdp_80_list.append(ret[4])
    snr_3_list.append(ret[5])
    snr_5_list.append(ret[6])
    snr_7_list.append(ret[7])
    snr_10_list.append(ret[8])
    snr_15_list.append(ret[9])
    snr_30_list.append(ret[10])
    snr_50_list.append(ret[11])
    snr_70_list.append(ret[12])
    snr_100_list.append(ret[13])
    snr_200_list.append(ret[14])
    snr_300_list.append(ret[15])
    snr_600_list.append(ret[16])



ergmin = [10, 30, 40, 50, 60, 70, 80, 90, 100]
ergmax = [300, 460, 600, 800, 1000]
min_list = []
max_list = []
for emin in ergmin:
  for emax in ergmax:
    min_list.append(emin)
    max_list.append(emax)
snr_100_list = [41.042221, 43.085818, 43.426417, 43.767017, 43.767017, 40.871922, 42.745218, 43.085817, 43.256117, 43.256117, 40.020423, 42.234319, 42.234319, 42.574918, 42.404619, 39.339225, 41.553120, 41.212521, 41.382821, 41.382821, 38.650826, 40.361023, 40.531322, 40.871922, 40.701622, 37.295628, 38.658026, 38.828326, 38.998625, 38.828326, 35.762931, 37.295628, 37.295628, 37.465928, 37.636228, 34.400534, 35.762931, 35.762931, 35.762931, 35.762931, 33.719335, 34.400534, 34.911433, 34.911433, 34.911433]
mdp_80_list = [6.4713, 9.7071, 11.0694, 11.2398, 11.2398, 6.4713, 9.7071, 11.0694, 11.2398, 11.2398, 6.4713, 9.7071, 11.0694, 11.2398, 11.2398, 6.4713, 9.7071, 11.0694, 11.2398, 11.2398, 6.4713, 9.7071, 11.0694, 11.2398, 11.2398, 6.4713, 9.7071, 11.0694, 11.2398, 11.2398, 6.4713, 9.7071, 11.0694, 11.2398, 11.2398, 6.4713, 9.7071, 11.0694, 11.2398, 11.4101, 6.4713, 9.7071, 11.0694, 11.2398, 11.2398]
x, y = np.meshgrid(ergmin, ergmax)
z_snr_100 = np.transpose(np.reshape(snr_100_list, (9, 5)))
z_mdp_80 = np.transpose(np.reshape(mdp_80_list, (9, 5)))

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, z_snr_100, cmap="Blues", linewidth=0, antialiased=False)
cbar = fig.colorbar(surf)
cbar.set_label("GRB rate with snr > 100", rotation=90, labelpad=20)
ax.set(xlabel="Energy window lower edge (keV)", ylabel="Energy window higher edge (keV)", zlabel="Detection rate with snr > 100")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, z_mdp_80, cmap="Blues", linewidth=0, antialiased=False)
cbar = fig.colorbar(surf)
cbar.set_label("GRB rate with mdp < 80", rotation=90, labelpad=20)
ax.set(xlabel="Energy window lower edge (keV)", ylabel="Energy window higher edge (keV)", zlabel="Detection rate with mdp < 80")
plt.show()




cat_short = Catalog("GBM/shortGBM.txt", [4, '\n', 5, '|', 2000])
cat_long = Catalog("GBM/longGBM.txt", [4, '\n', 5, '|', 2000])
cat_short.tofloat("t90")
cat_long.tofloat("t90")
longtimes = np.array(cat_long.t90)
shorttimes = np.array(cat_short.t90)
alltimes = np.concatenate((longtimes, shorttimes))
print(np.median(alltimes))
print(np.median(longtimes))
print(np.median(shorttimes))
longmedtime = np.median(longtimes)

longtimefrac = []
for tim in longtimes:
  if tim - longmedtime > 0:
    longtimefrac.append((tim - longmedtime)/tim)

fig, ax = plt.subplots()
ax.hist(longtimefrac, bins=30, cumulative=0)
# 'barstacked', 'step'
# ax.set(xticks=np.linspace(0, 1, 11), yticks=np.linspace(0, 1000, 11), title="Cumulative distribution of the fraction of time a GRB longer than the median time is seen", xlabel="Time fraction over the whole duration of the burst", ylabel="Number of long GRB over the GBM catalog")
ax.grid(linestyle='--', color='black', alpha=0.3)
plt.show()

limittime = 20
longtimefrac2 = []
for tim in longtimes:
  if tim - limittime > 0:
    longtimefrac2.append((tim - limittime)/tim)
fig, ax = plt.subplots()
ax.hist(longtimefrac2, bins=30, cumulative=1, label=f"Total number of GRBs : {len(longtimefrac2)}")
# 'barstacked', 'step'
ax.set(xticks=np.linspace(0, 1, 11), yticks=np.arange(0, len(longtimefrac2) + 100, 100), title=f"Cumulative distribution of the fraction of time a GRB longer than {limittime}s is seen", xlabel="Time fraction over the whole duration of the burst", ylabel="Number of long GRB over the GBM catalog")
ax.grid(linestyle='--', color='black', alpha=0.3)
ax.legend()
plt.show()



fig, ax = plt.subplots()
log_bins = np.logspace(np.log10(min(shorttimes)), np.log10(max(longtimes)), 50)
ax.hist(longtimes, bins=log_bins, histtype='step', label="lGRBs")
ax.hist(shorttimes, bins=log_bins, histtype='step', label="sGRBs")
# 'barstacked', 'step'
ax.set(xscale='log', xlabel="T90 : Duration of the burst", ylabel="Number of GRBs from GBM catalog")
ax.legend()
plt.show()

erg_cut = (10, 1000)
fluence_long = []
fluence_short = []
for ite in range(len(cat_long)):
  fluence_long.append(calc_flux_gbm(cat_long, ite, erg_cut) * longtimes[ite])
for ite in range(len(cat_short)):
  fluence_short.append(calc_flux_gbm(cat_short, ite, erg_cut) * shorttimes[ite])

print(fluence_long)
print(fluence_short)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
log_binslong = np.logspace(np.log10(min(fluence_long)), np.log10(max(fluence_long)), 50)
log_binsshort = np.logspace(np.log10(min(fluence_short)), np.log10(max(fluence_short)), 50)
ax1.hist(fluence_long, bins=log_binslong)
ax2.hist(fluence_short, bins=log_binsshort)
ax1.set(xscale='log', title="Fluence histograms for long GBRs")
ax2.set(xscale='log', title="Fluence histograms for short GBRs")
ax1.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax1.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)
ax2.grid(True, which='major', linestyle='--', color='black', alpha=0.3)
ax2.grid(True, which='minor', linestyle=':', color='black', alpha=0.2)
plt.show()


from MAllSourceData import AllSourceData
from MBkgContainer import BkgContainer
from MmuSeffContainer import MuSeffContainer
from visualisation import bkg_data_map
from time import time
import matplotlib.pyplot as plt
import numpy as np
from funcmod import closest_mufile
from funcmod import read_grbpar, eff_area_func, grb_decra_worldf2satf, horizon_angle, orbitalparam2decra, verif_rad_belts
import matplotlib as mpl
import cartopy.crs as ccrs


grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat/polGBM.par"
bkg_param = "./bkg/bkg-v15.par"
mu_param = "./mu100/mu100-v15.par"
ergcut = (30, 1000)
armcut = 180
bkgdata = BkgContainer(bkg_param, True, ergcut)
muSeffdata = MuSeffContainer(mu_param, ergcut, armcut)

bkg_data_map("cr", bkgdata, 500, dec_range=np.linspace(0, 180, 181), ra_range=np.linspace(0, 360, 361))

phi_sat = np.linspace(0, 360, 181)
theta_sat = np.linspace(0, 114, 115)

mu100list = np.zeros((len(theta_sat), len(phi_sat)))
seff_com_list = np.zeros((len(theta_sat), len(phi_sat)))
seff_sin_list = np.zeros((len(theta_sat), len(phi_sat)))
for i, theta in enumerate(theta_sat):
  for j, phi in enumerate(phi_sat):
    file = closest_mufile(theta, phi, muSeffdata)
    mu100list[i, j] = file[0]
    seff_com_list[i, j] = file[2]
    seff_sin_list[i, j] = file[3]

theta_sat = 90 - theta_sat
x_mu, y_mu = np.meshgrid(phi_sat, theta_sat)

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)})
p2 = ax.pcolormesh(x_mu, y_mu, mu100list, cmap="Blues", transform=ccrs.PlateCarree())
#plt.axis('scaled')
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="mu100 map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
# ax.set_xticks(np.arange(0, 360, 30), crs=ccrs.PlateCarree())
# ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
cbar = fig.colorbar(p2, ax=ax)
cbar.set_label("mu100 values", rotation=270, labelpad=20)
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)})
p3 = ax.pcolormesh(x_mu, y_mu, seff_com_list, cmap="Blues", transform=ccrs.PlateCarree())
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Compton Seff map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
cbar = fig.colorbar(p3, ax=ax)
cbar.set_label("Compton Seff (cm²)", rotation=270, labelpad=20)
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)})
p4 = ax.pcolormesh(x_mu, y_mu, seff_sin_list, cmap="Blues", transform=ccrs.PlateCarree())
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Single Seff map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
cbar = fig.colorbar(p4, ax=ax)
cbar.set_label("Single Seff (cm²)", rotation=270, labelpad=20)
plt.show()




grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat/polGBM.par"
bkg_param = "./bkg/bkg-v15.par"
mu_param = "./mu100/mu100-v15.par"
ergcut = (30, 1000)
armcut = 180
muSeffdata = MuSeffContainer(mu_param, ergcut, armcut)

sat_info = read_grbpar(grb_sim_param)[-1]

num_val = 100
nu_val = 0
n_sat = len(sat_info)
alt = sat_info[0][3]
excludefile = None
phi_world = np.linspace(0, 360, num_val)
theta_world = np.linspace(0, 180, num_val)

detection = np.zeros((n_sat, num_val, num_val))
detection_compton = np.zeros((n_sat, num_val, num_val))
detection_single = np.zeros((n_sat, num_val, num_val))
sat_dec_wf_list, sat_ra_wf_list = [], []
for ite in range(n_sat):
  sat_dec_wf, sat_ra_wf = orbitalparam2decra(sat_info[ite][0], sat_info[ite][1], sat_info[ite][2], nu=nu_val)
  sat_dec_wf_list.append(sat_dec_wf)
  sat_ra_wf_list.append(sat_ra_wf)
  if verif_rad_belts(sat_dec_wf, sat_ra_wf, alt):
    detection[ite] = np.array([[0 for phi in phi_world] for theta in theta_world])
    detection_compton[ite] = np.array([[0 for phi in phi_world] for theta in theta_world])
    detection_single[ite] = np.array([[0 for phi in phi_world] for theta in theta_world])
  else:
    for i, theta in enumerate(theta_world):
      for j, phi in enumerate(phi_world):
        detection_compton[ite][i][j], detection_single[ite][i][j], detection[ite][i][j] = eff_area_func(theta, phi, sat_dec_wf, sat_ra_wf, sat_info[ite][3], muSeffdata)
sat_lat_wf_list = 90 - np.array(sat_dec_wf_list)
sat_ra_wf_list = np.array(sat_ra_wf_list)
phi_plot, theta_plot = np.meshgrid(phi_world, 90 - theta_world)

detec_sum = np.sum(detection, axis=0)
detec_sum_compton = np.sum(detection_compton, axis=0)
detec_sum_single = np.sum(detection_single, axis=0)

detec_min = int(np.min(detec_sum))
detec_max = int(np.max(detec_sum))
cmap = mpl.cm.Blues_r
levels = np.arange(int(detec_min), int(detec_max) + 1, max(1, int((int(detec_max) + 1 - int(detec_min)) / 10)))

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
ax.set_global()
h1 = ax.pcolormesh(phi_plot, theta_plot, detec_sum, cmap=cmap)
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Constellation coverage map")
cbar = fig.colorbar(h1, ax=ax, ticks=levels)
cbar.set_label("Number of satellite in sight", rotation=270, labelpad=20)
colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
ax.scatter(sat_ra_wf_list, sat_lat_wf_list, color=colors[3])
ax.coastlines()
plt.show()


detec_min_compton = int(np.min(detec_sum_compton))
detec_max_compton = int(np.max(detec_sum_compton))
cmap_compton = mpl.cm.Blues_r
levels_compton = np.arange(int(detec_min_compton), int(detec_max_compton) + 1, max(1, int((int(detec_max_compton) + 1 - int(detec_min_compton)) / 10)))

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
ax.set_global()
h1 = ax.pcolormesh(phi_plot, theta_plot, detec_sum_compton, cmap=cmap_compton)
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Constellation effective area for compton events map")
cbar = fig.colorbar(h1, ax=ax, ticks=levels_compton)
cbar.set_label("Effective area for compton events (cm²)", rotation=270, labelpad=20)
colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
ax.scatter(sat_ra_wf_list, sat_lat_wf_list, color=colors[3])
ax.coastlines()
plt.show()


detec_min_single = int(np.min(detec_sum_single))
detec_max_single = int(np.max(detec_sum_single))
cmap_single = mpl.cm.Blues_r
levels_single = np.arange(int(detec_min_single), int(detec_max_single) + 1, max(1, int((int(detec_max_single) + 1 - int(detec_min_single)) / 10)))

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
ax.set_global()
h1 = ax.pcolormesh(phi_plot, theta_plot, detec_sum_single, cmap=cmap_single)
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Constellation effective area for single events map")
cbar = fig.colorbar(h1, ax=ax, ticks=levels_single)
cbar.set_label("Effective area for single events (cm²)", rotation=270, labelpad=20)
colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
ax.scatter(sat_ra_wf_list, sat_lat_wf_list, color=colors[3])
ax.coastlines()
plt.show()


# theta_line = []
# phi_line = []
# Creating the lists of sat coordinates
# for ite_orbit in range(len(inc)):
#   line_temp_theta, line_temp_phi = orbitalparam2decra(inc[ite_orbit], ohm[ite_orbit], np.linspace(0, 360, 100))
#   theta_line.append(line_temp_theta)
#   phi_line.append(line_temp_phi)
# latitude_line = 90 - np.array(theta_line)
# phi_line = np.array(phi_line)

# TODO not correct because of changes on verif_rad_belts
theta_verif = np.linspace(-90, 90, 181)
phi_verif = np.linspace(-180, 180, 360, endpoint=False)
if excludefile is not None:
  plottitle = f"All radiation belt {alt}km"
  cancel_theta = []
  cancel_phi = []
  for theta in theta_verif:
    for phi in phi_verif:
      if verif_rad_belts(theta, phi, alt):
        cancel_theta.append(theta)
        cancel_phi.append(phi)
  cancel_theta = np.array(cancel_theta)
  cancel_phi = np.array(cancel_phi)
  ax.scatter(cancel_phi, cancel_theta, s=1)
  ax.set(title=plottitle)
# Adding the coasts

print(f"Le nombre moyen de satellite voyant une position certaine du ciel est de {np.mean(np.mean(detec_sum, axis=1))} cm²")

selected_sat="const"
mdp_threshold=1
cumul=1
n_bins=35
x_scale='linear'
y_scale="log"
grb_type = "lGRB"
number_detected = 0
mdp_list = []
for source in test.alldata:
  if source is not None:
    for sim in source:
      if sim is not None:
        if type(selected_sat) is int:
          if sim[selected_sat] is not None:
            number_detected += 1
            if sim[selected_sat].mdp is not None:
              if sim[selected_sat].mdp <= mdp_threshold:
                mdp_list.append(sim[selected_sat].mdp * 100)
        elif selected_sat == "const":
          if sim.const_data is not None:
            number_detected += 1
            if sim.const_data.mdp is not None:
              if sim.const_data.mdp <= mdp_threshold:
                mdp_list.append(sim.const_data.mdp * 100)
fig, ax = plt.subplots(1, 1)
ax.hist(mdp_list, bins=n_bins, cumulative=cumul, histtype="step", weights=[test.weights] * len(mdp_list),
        label=f"Number of GRBs with MDP < {mdp_threshold * 100}% : {len(mdp_list)} over {number_detected} detections")
ax.axvline(30, color='black')
if cumul == 1:
  ax.set(xlabel="MPD (%)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
         title=f"Cumulative distribution of the MDP - {grb_type}")
elif cumul == 0:
  ax.set(xlabel="MPD (%)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
         title=f"Distribution of the MDP - {grb_type}")
elif cumul == -1:
  ax.set(xlabel="MPD (%)", ylabel="Number of detection per year", xscale=x_scale, yscale=y_scale,
         title=f"Inverse cumulative distribution of the MDP - {grb_type}")
ax.legend(loc='upper left')
ax.grid(axis='both')
plt.show()


mdp10 = []
fluence10 = []
ener_fluence10 = []
mdp30 = []
fluence30 = []
ener_fluence30 = []
mdp50 = []
fluence50 = []
ener_fluence50 = []
mdp80 = []
fluence80 = []
ener_fluence80 = []
number_detected = 0
all_cat = Catalog(test.cat_file, test.sttype)
for source in test.alldata:
  if source is not None:
    for sim in source:
      if sim is not None:
        if sim.const_data is not None:
          number_detected += 1
          ener_fluence = None
          for ite, name in enumerate(all_cat.name):
            if name == source.source_name:
              ener_fluence = float(all_cat.fluence[ite])
          if sim.const_data.mdp is not None:
            if sim.const_data.mdp <= 0.1:
              mdp10.append(sim.const_data.mdp * 100)
              fluence10.append(source.source_fluence)
              ener_fluence10.append(ener_fluence)
            if sim.const_data.mdp <= 0.3:
              mdp30.append(sim.const_data.mdp * 100)
              fluence30.append(source.source_fluence)
              ener_fluence30.append(ener_fluence)
            if sim.const_data.mdp <= 0.5:
              mdp50.append(sim.const_data.mdp * 100)
              fluence50.append(source.source_fluence)
              ener_fluence50.append(ener_fluence)
            if sim.const_data.mdp <= 0.8:
              mdp80.append(sim.const_data.mdp * 100)
              fluence80.append(source.source_fluence)
              ener_fluence80.append(ener_fluence)

mdp_limits = [0.1, 0.3, 0.5, 0.8]
chosen_fluence = [fluence10, fluence30, fluence50, fluence80]
chosen_ener_fluence = [ener_fluence10, ener_fluence30, ener_fluence50, ener_fluence80]
flu_bins = np.logspace(-1, 4, 100)
ener_flu_bins = np.logspace(-9, -1, 100)
for ite, mdp_limit in enumerate(mdp_limits):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
  ax1.hist(chosen_fluence[ite], bins=flu_bins, histtype="step", weights=[test.weights] * len(chosen_fluence[ite]), label=f"Number of GRB with MDP < {mdp_limit * 100}% : {len(chosen_fluence[ite])} over {number_detected} detections")
  ax1.set(xlabel="Fluence (photons/cm²)", ylabel="Number of detection per year", xscale="log", yscale="log", title=f"Distribution of fluence for GRB with MDP < {mdp_limit * 100}")
  ax1.legend(loc='upper left')
  ax1.grid(axis='both')
  ax2.hist(chosen_ener_fluence[ite], bins=ener_flu_bins, histtype="step", weights=[test.weights] * len(chosen_ener_fluence[ite]), label=f"Number of GRB with MDP < {mdp_limit * 100}% : {len(chosen_ener_fluence[ite])} over {number_detected} detections")
  ax2.set(xlabel="Fluence (egr/cm²)", ylabel="Number of detection per year", xscale="log", yscale="log", title=f"Distribution of fluence in energy for GRB with MDP < {mdp_limit * 100}")
  ax2.legend(loc='upper left')
  ax2.grid(axis='both')
  plt.show()








all_cat = Catalog(test.cat_file, test.sttype)
total_in_view = 0
# Setting 1s mean triggers counter
single_instant_trigger_by_const = 0
single_instant_trigger_by_sat = 0
single_instant_trigger_by_comparison = 0
# Setting 1s peak triggers counter
single_peak_trigger_by_const = 0
single_peak_trigger_by_sat = 0
single_peak_trigger_by_comparison = 0
# Setting T90 mean triggers counter
single_t90_trigger_by_const = 0
single_t90_trigger_by_sat = 0
single_t90_trigger_by_comparison = 0

for source in test.alldata:
  if source is not None:
    for sim in source:
      if sim is not None:
        total_in_view += 1
        #    Setting the trigger count to 0
        # Instantaneous trigger
        sat_instant_triggers = 0
        sat_reduced_instant_triggers = 0
        # Peak trigger
        sat_peak_triggers = 0
        sat_reduced_peak_triggers = 0
        # t90 trigger
        sat_t90_triggers = 0
        sat_reduced_t90_triggers = 0
        # Calculation for the individual sats
        for sat in sim:
          if sat is not None:
            if sat.snr_single >= test.snr_min:
              sat_instant_triggers += 1
            if sat.snr_single >= test.snr_min - 2:
              sat_reduced_instant_triggers += 1
            if source.best_fit_p_flux is None:
              sat_peak_snr = sat.snr_single
            else:
              sat_peak_snr = calc_snr(
                rescale_cr_to_GBM_pf(sat.single_cr, source.best_fit_mean_flux, source.best_fit_p_flux),
                sat.single_b_rate)[0]
            if sat_peak_snr >= test.snr_min:
              sat_peak_triggers += 1
            if sat_peak_snr >= test.snr_min - 2:
              sat_reduced_peak_triggers += 1
            if sat.snr_single_t90 >= test.snr_min:
              sat_t90_triggers += 1
            if sat.snr_single_t90 >= test.snr_min - 2:
              sat_reduced_t90_triggers += 1
        # Calculation for the whole constellation
        if source.best_fit_p_flux is None:
          const_peak_snr = sim.const_data.snr_single
        else:
          const_peak_snr = calc_snr(
            rescale_cr_to_GBM_pf(sim.const_data.single_cr, source.best_fit_mean_flux, source.best_fit_p_flux),
            sim.const_data.single_b_rate)[0]
        # 1s mean triggers
        if sim.const_data.snr_single >= test.snr_min:
          single_instant_trigger_by_const += 1
        if sat_instant_triggers >= 1:
          single_instant_trigger_by_sat += 1
        if sat_reduced_instant_triggers >= 3:
          single_instant_trigger_by_comparison += 1
        # 1s peak triggers
        if const_peak_snr >= test.snr_min:
          single_peak_trigger_by_const += 1
        if sat_peak_triggers >= 1:
          single_peak_trigger_by_sat += 1
        if sat_reduced_peak_triggers >= 3:
          single_peak_trigger_by_comparison += 1
        # T90 mean triggers
        if sim.const_data.snr_single_t90 >= test.snr_min:
          single_t90_trigger_by_const += 1
        if sat_t90_triggers >= 1:
          single_t90_trigger_by_sat += 1
        if sat_reduced_t90_triggers >= 3:
          single_t90_trigger_by_comparison += 1

print("The number of trigger for single events for the different technics are the following :")
print(" == Integration time for the trigger : 1s, mean flux == ")
print(
  f"   For a {test.snr_min} sigma trigger with the number of hits summed over the constellation :  {single_instant_trigger_by_const:.2f} triggers")
# print(f"   For a {test.snr_min} sigma trigger on at least one of the satellites :                      {single_instant_trigger_by_sat:.2f} triggers")
print(
  f"   For a {test.snr_min - 2} sigma trigger in at least 3 satellites of the constellation :        {single_instant_trigger_by_comparison:.2f} triggers")
# print(" == Integration time for the trigger : T90, mean flux == ")
# print(f"   For a {test.snr_min} sigma trigger with the number of hits summed over the constellation : {single_t90_trigger_by_const} triggers")
# print(f"   For a {test.snr_min} sigma trigger on at least one of the satellites : {single_t90_trigger_by_sat} triggers")
# print(f"   For a {test.snr_min-2} sigma trigger in at least 3 satellites of the constellation : {single_t90_trigger_by_comparison} triggers")
print("The number of trigger using GBM pflux for an energy range between 10keV and 1MeV are the following :")
print(" == Integration time for the trigger : 1s, peak flux == ")
print(
  f"   For a {test.snr_min} sigma trigger with the number of hits summed over the constellation :  {single_peak_trigger_by_const:.2f} triggers")
# print(f"   For a {test.snr_min} sigma trigger on at least one of the satellites :                      {single_peak_trigger_by_sat:.2f} triggers")
print(
  f"   For a {test.snr_min - 2} sigma trigger in at least 3 satellites of the constellation :        {single_peak_trigger_by_comparison:.2f} triggers")
print("=============================================")
print(f" Over the {total_in_view} GRBs simulated in the constellation field of view")

print("================================================================================================")
print("== Triggers according to GBM method")
print("================================================================================================")
# bin_widths = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 10]
total_in_view = 0
bin_widths = [0.064, 0.256, 1.024]
sat_trigger_counter = 0
const_trigger_counter = 0
for source in test.alldata:
  if source is not None:
    for ite_sim, sim in enumerate(source):
      if sim is not None:
        total_in_view += 1
        # Getting the snrs for all satellites and the constellation
        list_bins = [np.arange(0, source.source_duration + width, width) for width in bin_widths]
        centroid_bins = [(list_bin[1:] + list_bin[:-1]) / 2 for list_bin in list_bins]
        sat_snr_list = []
        sat_trigg = 0
        for sat_ite, sat in enumerate(sim):
          if sat is not None:
            if len(np.concatenate((sat.compton_time, sat.single_time))) == 0:
              sat_trigg += 0
            else:
              temp_hist = [np.histogram(np.concatenate((sat.compton_time, sat.single_time)), bins=list_bin)[0] for list_bin in list_bins]
              arg_max_bin = [np.argmax(val_hist) for val_hist in temp_hist]
              # for index1, arg_max1 in enumerate(arg_max_bin):
              #   for index2, arg_max2 in enumerate(arg_max_bin[index1 + 1:]):
              #     if not compatibility_test(centroid_bins[index1][arg_max1], bin_widths[index1], centroid_bins[index1 + 1 + index2][arg_max2], bin_widths[index1 + 1 + index2]):
                    # print(f"Incompatibility between bins {bin_widths[index1]} and {bin_widths[index2]} for {source.source_name}, sim {ite_sim} and sat {sat_ite}")
                    # print(f"     Centroids of the incompatible bins : {centroid_bins[index1][arg_max1]} and {centroid_bins[index1 + 1 + index2][arg_max2]}")
              snr_list = [calc_snr(temp_hist[index][arg_max_bin[index]], (sat.single_b_rate + sat.compton_b_rate) * bin_widths[index])[0] for index in range(len(arg_max_bin))]
              sat_snr_list.append(snr_list)
              if max(snr_list) > 3:
                sat_trigg += 1
        if sim.const_data is not None:
          if len(np.concatenate((sim.const_data.compton_time, sim.const_data.single_time))) == 0:
            const_snr = [0]
          else:
            temp_hist = [np.histogram(np.concatenate((sim.const_data.compton_time, sim.const_data.single_time)), bins=list_bin)[0] for list_bin in list_bins]
            arg_max_bin = [np.argmax(val_hist) for val_hist in temp_hist]
            # for index1, arg_max1 in enumerate(arg_max_bin):
            #   for index2, arg_max2 in enumerate(arg_max_bin[index1 + 1:]):
            #     if not compatibility_test(centroid_bins[index1][arg_max1], bin_widths[index1], centroid_bins[index1 + 1 + index2][arg_max2], bin_widths[index1 + 1 + index2]):
                  # print(f"Incompatibility between bins {bin_widths[index1]} and {bin_widths[index2]} for {source.source_name}, sim {ite_sim} and constellation")
                  # print(f"     Centroids of the incompatible bins : {centroid_bins[index1][arg_max1]} and {centroid_bins[index1 + 1 + index2][arg_max2]}")
            const_snr = [calc_snr(temp_hist[index][arg_max_bin[index]], (sim.const_data.single_b_rate + sim.const_data.compton_b_rate) * bin_widths[index])[0] for index in range(len(arg_max_bin))]
        else:
          const_snr = [0]
        if sat_trigg >= 4:
          sat_trigger_counter += 1
        if max(const_snr) >= 6:
          const_trigger_counter += 1
        else:
          for ite, name in enumerate(all_cat.name):
            if name == source.source_name:
              ener_fluence = float(all_cat.fluence[ite])
          print(max(const_snr), source.source_duration, sim.dec_world_frame, ener_fluence)
print(
  f"   For a 6 sigma trigger with the number of hits summed over the constellation :  {const_trigger_counter:.2f} triggers")
print(
  f"   For a 3 sigma trigger in at least 4 satellites of the constellation :        {sat_trigger_counter:.2f} triggers")
print("=============================================")
print(f" Over the {total_in_view} GRBs simulated in the constellation field of view")

with open("test_mdp", "w") as f:
  f.write("Result file for the MDP threshold study\n")
  f.write("Threshold | MDP100 | MDP90 | MDP80 | MDP70 | MDP60 | MDP50 | MDP40 | MDP30 | MDP20 | MDP10 | Number detected | Number mdp <= 100\n")
  for threshold_mdp in [1]:
    test.set_beneficial(threshold_mdp)
    test.make_const()
    test.analyze()
    print(f" ========               MDP THRESHOLD USED : {threshold_mdp}   ========")
    number_detected = 0
    mdp_list = []
    for source in test.alldata:
      if source is not None:
        for sim in source:
          if sim is not None:
            if sim.const_data is not None:
              number_detected += 1
              if sim.const_data[0].mdp is not None:
                if sim.const_data[0].mdp <= 1:
                  mdp_list.append(sim.const_data[0].mdp * 100)
    mdp_list = np.array(mdp_list)
    mdp100 = np.sum(np.where(mdp_list <= 100, 1, 0)) * test.weights
    mdp90 = np.sum(np.where(mdp_list <= 90, 1, 0)) * test.weights
    mdp80 = np.sum(np.where(mdp_list <= 80, 1, 0)) * test.weights
    mdp70 = np.sum(np.where(mdp_list <= 70, 1, 0)) * test.weights
    mdp60 = np.sum(np.where(mdp_list <= 60, 1, 0)) * test.weights
    mdp50 = np.sum(np.where(mdp_list <= 50, 1, 0)) * test.weights
    mdp40 = np.sum(np.where(mdp_list <= 40, 1, 0)) * test.weights
    mdp30 = np.sum(np.where(mdp_list <= 30, 1, 0)) * test.weights
    mdp20 = np.sum(np.where(mdp_list <= 20, 1, 0)) * test.weights
    mdp10 = np.sum(np.where(mdp_list <= 10, 1, 0)) * test.weights

    f.write(f"{threshold_mdp} | {mdp100} | {mdp90} | {mdp80} | {mdp70} | {mdp60} | {mdp50} | {mdp40} | {mdp30} | {mdp20} | {mdp10} | {number_detected} | {len(mdp_list)}\n")

    print("=                        MDP detection rates                        =")
    print(f"   MDP<=100% : {mdp100}")
    print(f"   MDP<=90%  : {mdp90}")
    print(f"   MDP<=80%  : {mdp80}")
    print(f"   MDP<=70%  : {mdp70}")
    print(f"   MDP<=60%  : {mdp60}")
    print(f"   MDP<=50%  : {mdp50}")
    print(f"   MDP<=40%  : {mdp40}")
    print(f"   MDP<=30%  : {mdp30}")
    print(f"   MDP<=20%  : {mdp20}")
    print(f"   MDP<=10%  : {mdp10}")
