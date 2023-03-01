import sys
sys.path.append('/disk/megalib/test-nathan/')

from catalog import Catalog
from gmafw import *
import trafile
import argparse
from trafile import plaw, band, comp, sbpl

from scipy.integrate import quad
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

if __name__ == "__main__":
  #===================================================================#
  # Definition of arguments
  #===================================================================#
  parser = argparse.ArgumentParser(description="Data analyze tool for extracted simulated data of GRB")
  parser.add_argument("-d", "--directory", help="Name of the directory in which the analyse is done")
  parser.add_argument("-r", "--repartition", help="Plot the map of GRB in the sky, possibility to add a colormap. 'no_cm' will plot only the GRB, one can use the colormap with the following keywords : t90", default=None)
  parser.add_argument("-g", "--graph", help="String or list of string to specify what graph is to be plotted", default="mdp_hist")
  args = parser.parse_args()

  #===================================================================#
  # Initialisation of simulation parameters from parfile
  #===================================================================#
  directory = args.directory
  par_file = f"./{directory}/polGBM.par"
  poltab_folder = f"./{directory}/Poltab/"
  sat_info = []
  with open(par_file) as f:
    lines = f.read().split("\n")
  for line in lines:
    if line.startswith("@prefix"):
      result_prefix = line.split(" ")[1].split("/")
      if result_prefix[3]=='long':
        GRB_type = "long GRB"
      else:
        GRB_type = "short GRB"
      result_prefix = f'./{directory}/{result_prefix[2]}/{result_prefix[3]}'
    elif line.startswith("@mode"):
      mode = line.split(" ")[1:]
    elif line.startswith("@sttype"):
      sttype = line.split(" ")[1:]
    elif line.startswith("@file"):
      GRB_file = f'../{line.split(" ")[1]}'
    elif line.startswith("@simulationsperevent"):
      n_simu = int(line.split(" ")[1])
    elif line.startswith("@satellite"):
      temp = [float(e) for e in line.split(" ")[1:]]
      if len(temp) == 3:#satellite pointing
        dat = [temp[0], temp[1], horizonAngle(temp[2])]
      else:#satellite orbital parameters
        #inclination, omega, nu = map(np.deg2rad, temp[:3]) #rad
        #thetasat = np.arccos(np.sin(inclination)*np.cos(nu)) #rad
        #phisat = np.arctan2( (np.cos(nu)*np.cos(inclination)*np.cos(omega)-np.sin(nu)*np.sin(omega)) , (-np.sin(nu)*np.cos(omega)-np.cos(nu)*np.cos(inclination)*np.sin(omega)) ) #rad
        # Extracting inclination, ohm, omega, respectively the inclination, the right ascention of the ascending node and the argument of periapsis
        inclination, ohm, omega = map(np.deg2rad, temp[:3])
        thetasat = np.arccos(np.sin(inclination)*np.sin(omega)) #rad
        phisat = np.arctan2( (np.cos(omega) * np.sin(ohm) + np.sin(omega) * np.cos(inclination) *np.cos(ohm)) , (np.cos(omega) * np.cos(ohm) - np.sin(omega) * np.cos(inclination) * np.sin(ohm)) ) #rad
        dat = [thetasat, phisat, trafile.horizonAngle(temp[3])]
        sat_info.append(dat)
  sat_info = np.array(sat_info, dtype = float)
  const                         = range(len(sat_info))
  # Extraction of grb info from grb file
  c = Catalog(GRB_file, sttype)
  c.tofloat("t90")
  c.tofloat('fluence')
  # Creation of the list of peak flux obtained with the best fit model
  grb_pflux = []
  if mode == "flnc":
    for ite, best_fit in enumerate(c.flnc_best_fitting_model):
      exec(f'grb_pflux.append(float(c.{best_fit.rstrip()}_phtflux[ite]))')
  else:
    for ite, best_fit in enumerate(c.pflx_best_fitting_model):
      exec(f'grb_pflux.append(float(c.{best_fit.rstrip()}_phtflux[ite]))')
  n_GRB                     = len(c.name)

  #===================================================================#
  # Initialisation of analyse parameters (those are not fixed)
  #===================================================================#
  # Analyse parameters
  sim_duration                  = 100
  y_scale                       = 'log'
  n_bins                        = 100
  SNR_min                       = 5
  erg_cut                       = (100, 460)
  cat_duration                  = 10 #years
  GBM_duty                      = 0.6 #GBM duty cycle
  COM_duty                      = 1
  GBM_fov                       = (1 - np.cos(np.deg2rad(trafile.horizonAngle(565)))) / 2
  COM_fov                       = 1 # A implémenter et vérifier !
  count_weights                 = 1/n_simu/cat_duration*COM_duty/GBM_duty*COM_fov/GBM_fov

  mpl.rcParams.update({'font.size': 20})

  if GRB_type == "long GRB":
    count_weights = count_weights * 1928 / len(c.name)
  else:
    count_weights = count_weights * 361 / len(c.name)

  # Extraction of background count rate
  bkgrnd_file                   = "./Background_550km-0deg-3600s_0.0_0.0.inc1.id1.extracted.tra"
  if bkgrnd_file==None:
    bkg_count = 0.
  else:
    tra_bkgrnd = trafile.Trafile(bkgrnd_file, items=["CE"], optanalysis=[trafile.treatCE])
    tra_bkgrnd.setCompton(ergCut=erg_cut)
    bkg_count_rate = tra_bkgrnd.compton/3600
    bkg_count = tra_bkgrnd.compton/3600*sim_duration
  
  #===================================================================#
  # Boolean variables
  #===================================================================#
  Activate_all                       = args.graph == "all"
  Activate_mdp_hist                  = args.graph == "mdp_hist" or "mdp_hist" in args.graph or Activate_all
  Activate_hits_vs_energy            = args.graph == "hits_vs_energy" or "hits_vs_energy" in args.graph or Activate_all
  Activate_const_FoV                 = args.graph == "const_FoV" or "const_FoV" in args.graph or Activate_all
  Activate_const_eff_area            = args.graph == "const_eff_area" or "const_eff_area" in args.graph or Activate_all
  Activate_detec_rate_vs_flux        = args.graph == "detec_rate_vs_flux" or "detec_rate_vs_flux" in args.graph or Activate_all
  Activate_detec_proba_vs_flux       = args.graph == "detec_proba_vs_flux" or "detec_proba_vs_flux" in args.graph or Activate_all
  Activate_compt_detec_proba_vs_flux = args.graph == "compt_detec_proba_vs_flux" or "compt_detec_proba_vs_flux" in args.graph or Activate_all
  Activate_detec_rate_vs_mu          = args.graph == "detec_rate_vs_mu" or "detec_rate_vs_mu" in args.graph or Activate_all
  Activate_detec_rate_vs_pa          = args.graph == "detec_rate_vs_pa" or "detec_rate_vs_pa" in args.graph or Activate_all
  Activate_detec_rate_vs_mdp         = args.graph == "detec_rate_vs_mdp" or "detec_rate_vs_mdp" in args.graph or Activate_all
  Activate_mdp_vs_fluence            = args.graph == "mdp_vs_fluence" or "mdp_vs_fluence" in args.graph or Activate_all
  Activate_viewing_angle             = args.graph == "viewing_angle" or "viewing_angle" in args.graph

  #===================================================================#
  # Analyse of the raw data to obtain the display data
  #===================================================================#
  if Activate_mdp_hist or Activate_detec_rate_vs_mu or Activate_detec_rate_vs_pa or Activate_detec_rate_vs_mdp or Activate_mdp_vs_fluence or Activate_all:
    #Creating the Poltab data object and extracting polarigram data and mdp
    # duration is made to be Tsim
    datap=Poltab(result_prefix,c.name,n_simu,trafile.parFile(par_file), ergCut=erg_cut,duration=sim_duration)
    datap.write(poltab_folder)
    data=datap.analyze(const)
    mdp=mdp99(data,c.t90,bkg_count_rate)

  if Activate_hits_vs_energy or Activate_detec_rate_vs_flux or Activate_detec_proba_vs_flux or Activate_compt_detec_proba_vs_flux or Activate_all:
    tratab = trafile.Trafiletab(result_prefix, c.name, nsim=n_simu, nsat=len(const))
    tratab.setCompton(ergCut=erg_cut)
        
    SNR_each_grb_simu = tratab.cSNR(const, "compton", bkg_count)
    det_prob_grb = tratab.cdetect("compton", bkg_count, SNR_min, nsim=n_simu)
    det_prob_grb_in_FoV = tratab.cdetect("compton", bkg_count, SNR_min)
    proba_comp_image = tratab.cCompImage(nsim=n_simu)
    proba_comp_image_in_FoV =tratab.cCompImage()

  #===================================================================#
  # Creation of the plots
  #===================================================================#

  #===================================================================#
  # Viewing angle study
  #===================================================================#
  if Activate_viewing_angle:
    ####    Importation of the data for 1 sat            ####
    infos_sat0 = trafile.parFile(par_file)
    infos_sat0['satellites'] = [infos_sat0['satellites'][0]]
    datap_va=Poltab(result_prefix, c.name, n_simu, infos_sat0, ergCut=erg_cut, duration=sim_duration)
    data_va=datap_va.analyze([0])
    mdp_va=mdp99(data_va,c.t90,bkg_count_rate)

    tratab_va = trafile.Trafiletab(result_prefix, c.name, nsim=n_simu, nsat=1)
    tratab_va.setCompton(ergCut=erg_cut)
    SNR_va = tratab_va.cSNR([0], "compton", bkg_count)

    c.tofloat('flnc_spectrum_start')
    c.tofloat('flnc_spectrum_stop')
    c.tofloat('pflx_plaw_ampl')
    c.tofloat('pflx_plaw_index')
    c.tofloat('pflx_plaw_pivot')
    c.tofloat('pflx_comp_ampl')
    c.tofloat('pflx_comp_index')
    c.tofloat('pflx_comp_epeak')
    c.tofloat('pflx_comp_pivot')
    c.tofloat('pflx_band_ampl')
    c.tofloat('pflx_band_alpha')
    c.tofloat('pflx_band_beta')
    c.tofloat('pflx_band_epeak')
    c.tofloat('pflx_sbpl_ampl')
    c.tofloat('pflx_sbpl_indx1')
    c.tofloat('pflx_sbpl_indx2')
    c.tofloat('pflx_sbpl_brken')
    c.tofloat('pflx_sbpl_brksc')
    c.tofloat('pflx_sbpl_pivot')
    c.tofloat('flnc_plaw_ampl')
    c.tofloat('flnc_plaw_index')
    c.tofloat('flnc_plaw_pivot')
    c.tofloat('flnc_comp_ampl')
    c.tofloat('flnc_comp_index')
    c.tofloat('flnc_comp_epeak')
    c.tofloat('flnc_comp_pivot')
    c.tofloat('flnc_band_ampl')
    c.tofloat('flnc_band_alpha')
    c.tofloat('flnc_band_beta')
    c.tofloat('flnc_band_epeak')
    c.tofloat('flnc_sbpl_ampl')
    c.tofloat('flnc_sbpl_indx1')
    c.tofloat('flnc_sbpl_indx2')
    c.tofloat('flnc_sbpl_brken')
    c.tofloat('flnc_sbpl_brksc')
    c.tofloat('flnc_sbpl_pivot')
    
    ####    Calculation of the fluence of GRB in ncounts/cm² and comparison to compton events to obtain Seff            ####
    def calc_fluence(index, ergCut):
      """
      Return the number of photons per cm² for a given energy range, averaged over the duration of the sim : ncount/cm²/s
      """
      model = c.flnc_best_fitting_model[ite].strip()
      if model == "pflx_plaw":
        func = lambda x: plaw(x, c.pflx_plaw_ampl[ite], c.pflx_plaw_index[ite], c.pflx_plaw_pivot[ite])
      elif model == "pflx_comp":
        func = lambda x: comp(x, c.pflx_comp_ampl[ite], c.pflx_comp_index[ite], c.pflx_comp_epeak[ite], c.pflx_comp_pivot[ite])
      elif model == "pflx_band":
        func = lambda x: band(x, c.pflx_band_ampl[ite], c.pflx_band_alpha[ite], c.pflx_band_beta[ite], c.pflx_band_epeak[ite])
      elif model == "pflx_sbpl":
        func = lambda x: sbpl(x, c.pflx_sbpl_ampl[ite], c.pflx_sbpl_indx1[ite], c.pflx_sbpl_indx2[ite], c.pflx_sbpl_brken[ite], c.pflx_sbpl_brksc[ite], c.pflx_sbpl_pivot[ite])
      elif model == "flnc_plaw":
        func = lambda x: plaw(x, c.flnc_plaw_ampl[ite], c.flnc_plaw_index[ite], c.flnc_plaw_pivot[ite])
      elif model == "flnc_comp":
        func = lambda x: comp(x, c.flnc_comp_ampl[ite], c.flnc_comp_index[ite], c.flnc_comp_epeak[ite], c.flnc_comp_pivot[ite])
      elif model == "flnc_band":
        func = lambda x: band(x, c.flnc_band_ampl[ite], c.flnc_band_alpha[ite], c.flnc_band_beta[ite], c.flnc_band_epeak[ite])
      elif model == "flnc_sbpl":
        func = lambda x: sbpl(x, c.flnc_sbpl_ampl[ite], c.flnc_sbpl_indx1[ite], c.flnc_sbpl_indx2[ite], c.flnc_sbpl_brken[ite], c.flnc_sbpl_brksc[ite], c.flnc_sbpl_pivot[ite])
      else:
        print("Could not find best fit model for {} (indicated {}). Aborting this GRB.".format(c.name[ite], model))
        return
      return quad(func, ergCut[0], ergCut[1])[0]
    
    fluence = {}
    for ite in range(len(c.name)):
      fluence[c.name[ite]] = calc_fluence(ite, erg_cut)*sim_duration
    
    s_eff = [np.array([tratab_va[0][grb][sim].compton/fluence[c.name[grb]] for sim in range(len(tratab_va[0][grb]))]) for grb in range(len(c.name))]
    
    def grb_zenith_angle_func(dir_sat, theta_grb, phi_grb):
      """
      The unit is in deg, so it must be changed into rad
      """
      theta_grb_rad, phi_grb_rad = np.deg2rad(theta_grb), np.deg2rad(phi_grb)
      dir_grb = [np.sin(theta_grb_rad)*np.cos(phi_grb_rad), np.sin(theta_grb_rad)*np.sin(phi_grb_rad), np.cos(theta_grb_rad)]
      return np.arccos(np.dot(dir_sat, dir_grb))
    
    dec_sat = infos_sat0["satellites"][0][0]
    ra_sat = infos_sat0["satellites"][0][1]
    dir_sat = [np.sin(dec_sat)*np.cos(ra_sat), np.sin(dec_sat)*np.sin(ra_sat), np.cos(dec_sat)]
    
    grb_zenith_angle = [np.array([grb_zenith_angle_func(dir_sat, tratab_va[0][grb][sim].theta, tratab_va[0][grb][sim].phi) for sim in range(len(tratab_va[0][grb]))]) for grb in range(len(c.name))]
    
    figure, ax = plt.subplots(2,2, figsize=(16,12))
    figure.suptitle("Effective area as a function of detection angle")
    for graph in range(4):
      for ite in range(graph * 10, graph * 10 + 10):
        ax[int(graph/2)][graph%2].scatter(grb_zenith_angle[ite], s_eff[ite], label=f"Fluence : {np.around(fluence[c.name[ite]], decimals=1)} ph/cm²")
      ax[int(graph/2)][graph%2].set(xlabel="GRB zenith angle (rad)", ylabel="Effective area (cm²)")#, yscale="linear")
      ax[int(graph/2)][graph%2].legend()
    plt.show()

    ####    Tests to estimate a function describing the evolution of S_eff as function of incidence angle    ####
    from scipy.optimize import curve_fit
    for ite in range(5):
      figure, ax = plt.subplots(1,1, figsize=(6,4))
      figure.suptitle("Effective area as a function of detection angle")
      ax.set(xlabel="GRB zenith angle (rad)", ylabel="Effective area (cm²)")#, yscale="linear")
      #ite = 0
      ax.scatter(grb_zenith_angle[ite], s_eff[ite], label=f"Fluence : {np.around(fluence[c.name[ite]], decimals=1)} ph/cm²")
      ax.legend()
      plt.show()
    
    figure, ax = plt.subplots(1,1, figsize=(6,4))
    figure.suptitle("Effective area as a function of detection angle")
    ax.set(xlabel="GRB zenith angle (rad)", ylabel="Effective area (cm²)")#, yscale="linear")
    ite = 15
    ax.scatter(grb_zenith_angle[ite], s_eff[ite], label=f"Fluence : {np.around(fluence[c.name[ite]], decimals=1)} ph/cm²")
    ax.plot(np.sort(grb_zenith_angle[ite]), np.absolute((8.6-1.5) * np.cos((np.sort(grb_zenith_angle[ite]) - np.pi/5) * 1.7)) + 2 , color='red')
    ax.legend()
    plt.show()
    
    
    def func_fit1(angle, ampl, ang_freq, phi0, y_off_set):
      return np.absolute((ampl) * np.cos(angle * 2*np.pi * ang_freq - phi0)) + y_off_set
    
    index_sort = np.argsort(grb_zenith_angle[ite])
    xdata = grb_zenith_angle[ite][index_sort]
    ydata = s_eff[ite][index_sort]
    popt, pcov = curve_fit(func_fit1, xdata, ydata, bounds=([0, 0.15, -np.inf, 0], [np.inf, 0.5, np.inf, 5]))
    plt.scatter(xdata, func_fit1(xdata, *popt))
    #plt.scatter(xdata, func_fit1(xdata, 8.6, 1.5, 0.27, 1.07, 2), color="red")
    plt.scatter(xdata, ydata, color="red")
    plt.show()
    
    ampl_dist = []
    ang_freq_dist = []
    phi0_dist = []
    y_off_set_dist = []
    
    for ite_val in range(40):
      index_sort = np.argsort(grb_zenith_angle[ite_val])
      xdata = grb_zenith_angle[ite_val][index_sort]
      ydata = s_eff[ite_val][index_sort]
      popt, pcov = curve_fit(func_fit1, xdata, ydata, bounds=([0, 0.15, -np.inf, 0], [np.inf, 0.5, np.inf, 5]))
      ampl_dist.append(popt[0])
      ang_freq_dist.append(popt[1])
      phi0_dist.append(popt[2])
      y_off_set_dist.append(popt[3])
      plt.scatter(xdata, func_fit1(xdata, *popt))
      plt.scatter(xdata, ydata, color="red")
      plt.scatter(xdata, func_fit1(xdata, 5.5, 0.222, 0.76, 2.5))
      plt.show()
    
    max_s_eff = []
    flu_list = []
    for ite in range(len(c.name)):
      #figure, ax = plt.subplots(1,1, figsize=(6,4))
      #figure.suptitle("Effective area as a function of detection angle")
      #ax.set(xlabel="GRB zenith angle (rad)", ylabel="Effective area (cm²)")#, yscale="linear")
      #ite = 0
      #ax.scatter(grb_zenith_angle[ite], s_eff[ite], label=f"Fluence : {np.around(fluence[c.name[ite]], decimals=1)} ph/cm²")
      #ax.legend()
      #plt.show()
      max_s_eff.append(np.max(s_eff[ite]))
      flu_list.append(fluence[c.name[ite]])
      print(np.max(s_eff[ite]), fluence[c.name[ite]])      




  #===================================================================#
  # FoV constellation (in projected coordinates and not)
  #===================================================================#
  ####    Printing the detection efficiency over the sky    ####
  if Activate_const_eff_area:
    def eff_area_pola_func(theta, angle_lim, func_type="cos"):
      if func_type=="cos":
        if theta<np.deg2rad(angle_lim):
          ampl = 5.5
          ang_freq = 0.222
          phi0 = 0.76
          y_off_set = 2.5
          return np.absolute((ampl) * np.cos(theta * 2*np.pi * ang_freq - phi0)) + y_off_set
        else:
          return 0
      elif func_type=="FoV":
        if theta<np.deg2rad(angle_lim):
          return 1
        else:
          return 0

    def eff_area_spectro_func(theta, angle_lim, func_type="data"):
      if func_type=="data":
        if theta<np.deg2rad(angle_lim):
          angles = np.deg2rad(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 89, 91, 100, 110]))
          eff_area = np.array([137.4, 148.5, 158.4, 161.9, 157.4, 150.4, 133.5, 112.8, 87.5, 63.6, 64.7, 71.8, 77.3])
          interpo_ite = 1
          if theta > angles[-1]:
            return eff_area[-2] + (eff_area[-1] - eff_area[-2]) / (angles[-1] - angles[-2]) * (theta - angles[-2])
          else:
            while theta > angles[interpo_ite]:
              interpo_ite += 1
            return eff_area[interpo_ite - 1] + (eff_area[interpo_ite] - eff_area[interpo_ite - 1]) / (angles[interpo_ite] - angles[interpo_ite - 1]) * (theta - angles[interpo_ite - 1])
        else:
          return 0
      elif func_type=="FoV":
        if theta<np.deg2rad(angle_lim):
          return 1
        else:
          return 0

    num_val = 500
    phi_world = np.linspace(0, 2*np.pi, num_val)
    # theta will be converted in sat coord with decra2tp, which takes dec in world coord with 0 being north pole and 180 the south pole !
    theta_world = np.linspace(0, np.pi, num_val)
    detection_pola = np.zeros((len(sat_info), num_val, num_val))
    detection_spectro = np.zeros((len(sat_info), num_val, num_val))

    for ite in range(len(sat_info)):
      #detection[ite] = np.array([[eff_area_func(trafile.decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="FoV") for phi in phi_world] for theta in theta_world])
      detection_pola[ite] = np.array([[eff_area_pola_func(trafile.decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="cos") for phi in phi_world] for theta in theta_world])
      detection_spectro[ite] = np.array([[eff_area_spectro_func(trafile.decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="data") for phi in phi_world] for theta in theta_world])

    detec_sum_pola = np.sum(detection_pola, axis=0)
    detec_sum_spectro= np.sum(detection_spectro, axis=0)
    
    phi_plot, theta_plot = np.meshgrid(phi_world, theta_world)
    detec_min_pola = int(np.min(detec_sum_pola))
    detec_max_pola = int(np.max(detec_sum_pola))
    detec_min_spectro = int(np.min(detec_sum_spectro))
    detec_max_spectro = int(np.max(detec_sum_spectro))
    cmap_pola = mpl.cm.Greens_r
    cmap_spectro = mpl.cm.Oranges_r

    # Eff_area plots for polarimetry
    #levels_pola = range(int(detec_min_pola / 2) * 2, detec_max_pola + 1)
    levels_pola = range(int(detec_min_pola), int(detec_max_pola) + 1, int((int(detec_max_pola) + 1 - int(detec_min_pola))/15))

    plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_pola, cmap=cmap_pola)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_pola)
    cbar.set_label("Effective area at for polarisation (cm²)", rotation=270, labelpad=20)
    plt.show()
 
    plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum_pola, cmap=cmap_pola)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_pola)
    cbar.set_label("Effective area for polarisation (cm²)", rotation=270, labelpad=20)
    plt.show()
  
    # Eff_area plots for spectroscopy
    #levels_spectro = range(int(detec_min_spectro / 2) * 2, detec_max_spectro + 1)
    levels_spectro = range(int(detec_min_spectro), int(detec_max_spectro) + 1, int((int(detec_max_spectro) + 1 - int(detec_min_spectro))/15))

    plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_spectro, cmap=cmap_spectro)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_spectro)
    cbar.set_label("Effective area for spectrometry (cm²)", rotation=270, labelpad=20)
    plt.show()

    plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum_spectro, cmap=cmap_spectro)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_spectro)
    cbar.set_label("Effective area for spectrometry (cm²)", rotation=270, labelpad=20)
    plt.show()

    print(f"La surface efficace moyenne pour la polarisation est de {np.mean(np.mean(detec_sum_pola, axis=1))} cm²")
    print(f"La surface efficace moyenne pour la spectrométrie est de {np.mean(np.mean(detec_sum_spectro, axis=1))} cm²")

  #===================================================================#
  # GRB map plot
  #===================================================================#
  if args.repartition is not None:
    plt.subplot(111, projection="aitoff")
    plt.xlabel("RA (°)")
    plt.ylabel("DEC (°)")
    plt.grid(True)
    plt.title("Map of GRB")
    #===================================================================#
    # Extracting dec and ra from catalog and transforms decimal degrees into degrees into the right frame
    #===================================================================#
    thetap = [np.sum(np.array(dec.split(" ")).astype(np.float)/[1, 60, 3600]) if len(dec.split("+"))==2 else np.sum(np.array(dec.split(" ")).astype(np.float)/[1, -60, -3600]) for dec in c.dec]
    print(thetap)
    #thetap = np.deg2rad(90-np.array(thetap))
    thetap = np.deg2rad(np.array(thetap))
    phip = [np.sum(np.array(ra.split(" ")).astype(np.float)/[1, 60, 3600]) if len(ra.split("+"))==2 else np.sum(np.array(ra.split(" ")).astype(np.float)/[1, -60, -3600]) for ra in c.ra]
    phip = np.mod(np.deg2rad(np.array(phip))+np.pi, 2*np.pi)-np.pi
    #phip = np.deg2rad(np.array(phip))
    if args.repartition=="no_cm":
      plt.scatter(phip, thetap, s=12 ,marker="*")
    elif args.repartition=="t90":
      sc = plt.scatter(phip, thetap, s=12 ,marker="*", c=c.t90, norm=colors.LogNorm())
      plt.colorbar(sc)
    plt.show()
  else:
    print("GRB map has not been required")

  #===================================================================#
  # mdp histogram
  #===================================================================#
  if Activate_mdp_hist:
    cumul = True
    # physical mdp is between 0 and 1, multiplying it by 100 makes it a percentage
    mdp_reduced = mdp[np.where(mdp<0.8, True, False)]*100
    plt.hist(mdp_reduced.flatten(), bins = n_bins, cumulative=cumul, histtype="step", weights=[count_weights]*len(mdp_reduced.flatten())) 
    if cumul:
      plt.title(f"Cumulative distribution of the MDP - {GRB_type}")
    else:
      plt.title(f"Distribution of the MDP - {GRB_type}")
    plt.xlabel("MPD (%)")
    plt.ylabel("Number of detection per year")
    plt.grid()
    plt.yscale(y_scale)
    plt.show()
  #===================================================================#
  # Hits vs energy (hist)
  #===================================================================#
  if Activate_hits_vs_energy:
    hits_energy = []
    for sats in range(len(tratab)):
      for sims in range(len(tratab[sats][0])):
        hits_energy += tratab[sats][0][sims].CE
    distrib, ax1 = plt.subplots(1,1, figsize=(8,6))
    distrib.suptitle("Energy distribution of photons for a GRB")
    ax1.hist(hits_energy, bins = n_bins, cumulative=0, histtype="step")
    ax1.set(xlabel="Energy (keV)", ylabel="Number of photon detected", yscale="linear")
    plt.show()

  #===================================================================#
  # FoV constellation (in projected coordinates and not)
  #===================================================================#
  if Activate_const_FoV:
    from matplotlib.colors import BoundaryNorm
    import matplotlib as mpl
    
    #mpl.rcParams.update({'font.size': 22})

    num_val = 500
    phi_world = np.linspace(0, 2*np.pi, num_val)
    # theta will be used with decra2tp, which takes dec in world coord with 0 being north pole and 180 the south pole !
    theta_world = np.linspace(0, np.pi, num_val)
    detection = np.zeros((len(sat_info), num_val, num_val))
    for ite in range(len(sat_info)):
      detection[ite] = np.array([[1 if trafile.decra2tp(theta, phi, sat_info[ite], unit="rad")[0]<np.deg2rad(sat_info[ite][2]) else 0 for phi in phi_world] for theta in theta_world])
    detec_sum = np.sum(detection, axis=0)

    phi_plot, theta_plot = np.meshgrid(phi_world, theta_world)
    detec_min = int(np.min(detec_sum))
    detec_max = int(np.max(detec_sum))
    cmap = mpl.cm.Blues_r
    
    #separated = True
    #if separated:
    # Usual plot
    levels1 = range(int(detec_min/2)*2, detec_max+1)
    
    plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi/2 - theta_plot, detec_sum, cmap=cmap)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels1)
    cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
    plt.show()
   
    plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot-np.pi, np.pi/2 - theta_plot, detec_sum, cmap=cmap)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels1)
    cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
    plt.show()
    
    print(f"Le nombre moyen de satellites couvrant le ciel est {np.mean(np.mean(detec_sum, axis=1))}")

    ## A activer si on veut des échelles de couleur différentes sur les graph
    if True:
      levels2 = range(int(detec_min/2)*2, detec_max+1, 2)
      norm2 = BoundaryNorm(levels2, ncolors=cmap.N)
    
      plt.subplot(projection=None)
      h2 = plt.pcolormesh(phi_plot, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm2)
      plt.axis('scaled')
      plt.xlabel("Right ascention (rad)")
      plt.ylabel("Declination (rad)")
      cbar = plt.colorbar(ticks=levels2)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
      plt.show()
    
      plt.subplot(projection="mollweide")
      h2 = plt.pcolormesh(phi_plot-np.pi, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm2)
      plt.grid(alpha=0.4)
      plt.xlabel("Right ascention (rad)")
      plt.ylabel("Declination (rad)")
      cbar = plt.colorbar(ticks=levels2)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
      plt.show()
    
      levels3 = range(int(detec_min/3)*3, detec_max+1, 3)
      norm3 = BoundaryNorm(levels3, ncolors=cmap.N)
      
      h3 = plt.pcolormesh(phi_plot, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm3)
      plt.axis('scaled')
      plt.xlabel("Right ascention (rad)")
      plt.ylabel("Declination (rad)")
      cbar = plt.colorbar(ticks=levels3)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
      plt.show()
    
      plt.subplot(projection="mollweide")
      h3 = plt.pcolormesh(phi_plot-np.pi, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm3)
      plt.grid(alpha=0.4)
      plt.xlabel("Right ascention (rad)")
      plt.ylabel("Declination (rad)")
      cbar = plt.colorbar(ticks=levels3)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
      plt.show()

      n_color = int(np.log2(detec_max - detec_min + 2) - 1)
      levels4 = [detec_min + 2**ite - 1 if ite <3 else detec_max - 2**(2 * n_color -1 - ite) + 1 for ite in range(2 * n_color)]
      norm4 = BoundaryNorm(levels4, ncolors=cmap.N)
    
      h4 = plt.pcolormesh(phi_plot, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm4)
      plt.axis('scaled')
      plt.xlabel("Right ascention (rad)")
      plt.ylabel("Declination (rad)")
      cbar = plt.colorbar(ticks=levels4)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
      plt.show()

      plt.subplot(projection="mollweide")
      h4 = plt.pcolormesh(phi_plot-np.pi, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm4)
      plt.grid(alpha=0.4)
      plt.xlabel("Right ascention (rad)")
      plt.ylabel("Declination (rad)")
      cbar = plt.colorbar(ticks=levels4)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
      plt.show()
    
    
      levels1 = range(int(detec_min/2)*2, detec_max+1)
      levels2 = range(int(detec_min/2)*2, detec_max+1, 2)
      norm2 = BoundaryNorm(levels2, ncolors=cmap.N)
      levels3 = range(int(detec_min/3)*3, detec_max+1, 3)
      norm3 = BoundaryNorm(levels3, ncolors=cmap.N)
      n_color = int(np.log2(detec_max - detec_min + 2) - 1)
      levels4 = [detec_min + 2**ite - 1 if ite <3 else detec_max - 2**(2 * n_color -1 - ite) + 1 for ite in range(2 * n_color)]
      norm4 = BoundaryNorm(levels4, ncolors=cmap.N)
    
      fig, axes = plt.subplots(2, 2, figsize=(12, 8))
      h1 = axes[0][0].pcolormesh(phi_plot, np.pi/2 - theta_plot, detec_sum, cmap=cmap)
      axes[0][0].set(xlabel="Right ascention (rad)", ylabel="Declination (rad)", aspect='equal')
      cbar = fig.colorbar(h1, ax = axes[0][0], ticks=levels1)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
    
      h2 = axes[0][1].pcolormesh(phi_plot, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm2)
      axes[0][1].set(xlabel="Right ascention (rad)", ylabel="Declination (rad)", aspect='equal')
      cbar = fig.colorbar(h2, ax = axes[0][1], ticks=levels2)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
    
      h3 = axes[1][0].pcolormesh(phi_plot, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm3)
      axes[1][0].set(xlabel="Right ascention (rad)", ylabel="Declination (rad)", aspect='equal')
      cbar = fig.colorbar(h3, ax = axes[1][0], ticks=levels3)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
    
      h4 = axes[1][1].pcolormesh(phi_plot, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm4)
      axes[1][1].set(xlabel="Right ascention (rad)", ylabel="Declination (rad)", aspect='equal')
      cbar = fig.colorbar(h4, ax = axes[1][1], ticks=levels4)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
      plt.show()

      fig= plt.figure()
      ax1 = fig.add_subplot(221, projection='mollweide')
      h1 = ax1.pcolormesh(phi_plot-np.pi, np.pi/2 - theta_plot, detec_sum, cmap=cmap) 
      ax1.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
      ax1.grid(alpha=0.4)
      cbar = fig.colorbar(h1, ax = ax1, ticks=levels1)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)

      ax2 = fig.add_subplot(222, projection='mollweide')
      h2 = ax2.pcolormesh(phi_plot-np.pi, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm2)
      ax2.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
      ax2.grid(alpha=0.4)
      cbar = fig.colorbar(h2, ax = ax2, ticks=levels2)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
  
      ax3 = fig.add_subplot(223, projection='mollweide')
      h3 = ax3.pcolormesh(phi_plot-np.pi, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm3)
      ax3.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
      ax3.grid(alpha=0.4)
      cbar = fig.colorbar(h3, ax = ax3, ticks=levels3)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)

      ax4 = fig.add_subplot(224, projection='mollweide')
      h4 = ax4.pcolormesh(phi_plot-np.pi, np.pi/2 - theta_plot, detec_sum, cmap=cmap, norm=norm4)
      ax4.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
      ax4.grid(alpha=0.4)
      cbar = fig.colorbar(h4, ax = ax4, ticks=levels4)
      cbar.set_label("Number of satellites covering the area", rotation=270, labelpad=20)
      plt.show()


  #===================================================================#
  # Detec rate GRB vs flux (hist)
  #===================================================================#
  if Activate_detec_rate_vs_flux:
    hist_pflux = []
    for grb in range(len(grb_pflux)):
      for sim_snr in SNR_each_grb_simu[grb]:
        if sim_snr > SNR_min:
          hist_pflux.append(grb_pflux[grb])
    distrib, ax1 = plt.subplots(1,1, figsize=(8,6))
    distrib.suptitle("Peak flux distribution of detected long GRB")
    ax1.hist(hist_pflux, bins = np.logspace(int(np.log10(min(hist_pflux)))-1, int(np.log10(max(hist_pflux))), n_bins), cumulative=False,histtype="step",weights=[count_weights]*len(hist_pflux))
    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Number of detection per year",xscale='log', yscale=y_scale)
    ax1.legend()
    plt.show()

  #===================================================================#
  # Detec proba vs flux (not hist)
  #===================================================================#
  if Activate_detec_proba_vs_flux:
    distrib, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
    distrib.suptitle("Detec proba vs peak flux of detected long GRB - GRB in the whole sky (left) and only in the FoV (right)")
    ax1.scatter(grb_pflux, det_prob_grb, s=2)
      
    ax2.scatter(grb_pflux, det_prob_grb_in_FoV, s=2)

    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Detection probability",xscale='log',)
    ax1.legend()
    ax2.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Detection probability",xscale='log',)
    ax2.legend()
    plt.show()

  #===================================================================#
  # Compton detec proba vs flux (not hist)
  #===================================================================#
  if Activate_compt_detec_proba_vs_flux:
    distrib, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
    distrib.suptitle("Compton Image proba vs peak flux of detected long GRB - GRB in the whole sky (left) and only in the FoV (right)")
    ax1.scatter(grb_pflux, proba_comp_image, s=2)
      
    ax2.scatter(grb_pflux, proba_comp_image_in_FoV, s=2)

    ax1.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Compton image probability",xscale='log',)
    ax1.legend()
    ax2.set(xlabel="Peak flux (photons/cm2/s)", ylabel="Compton image probability",xscale='log',)
    ax2.legend()
    plt.show()

  #===================================================================#
  # Detec rate vs mu100 (hist)
  #===================================================================#
  if Activate_detec_rate_vs_mu:
    mu_index = np.where(np.isnan(data[:,:,2].flatten()), False, True)
    mu_list = data[:,:,1].flatten()[mu_index]
    distrib, ax1 = plt.subplots(1,1, figsize=(8,6))
    distrib.suptitle("mu100 distribution of detected GRB")
    ax1.hist(mu_list, bins = n_bins, cumulative=0, histtype="step", weights=[count_weights]*len(mu_list))
    ax1.set(xlabel="mu100 (dimensionless)", ylabel="Number of detection per year", yscale=y_scale)
    plt.show()

  #===================================================================#
  # Detec rate vs PA (hist)
  #===================================================================#
  if Activate_detec_rate_vs_pa:
    distrib, ax1 = plt.subplots(1,1, figsize=(8,6))
    distrib.suptitle("Polarization angle distribution of detected GRB")
    ax1.hist(data[:,:,2].flatten(), bins = n_bins, cumulative=0, histtype="step", weights=[count_weights]*len(data[:,:,0].flatten()))
    ax1.set(xlabel="Polarization angle (°)", ylabel="Number of detection per year", yscale=y_scale)
    plt.show()

  #===================================================================#
  # Detec rate vs MDP99 (hist)
  #===================================================================#
  if Activate_detec_rate_vs_mdp:
    # Removes both nan and values >100
    mdp_cleaned = mdp.flatten()[np.where(mdp.flatten()<100, True, False)]
     
    distrib, ax1 = plt.subplots(1,1, figsize=(8,6))
    distrib.suptitle("mdp99 distribution of detected GRB")
    ax1.hist(mdp_cleaned.flatten(), bins = n_bins, cumulative=0, histtype="step", weights=[count_weights]*len(mdp_cleaned.flatten()), label=f'Detected GRB \nRatio of detectable polarization : {len(mdp_cleaned)/len(mdp.flatten())}')
    ax1.set(xlabel="mdp99 (%)", ylabel="Number of detection per year", yscale=y_scale)
    ax1.legend()
    plt.show()

  #===================================================================#
  # MDP vs fluence (no hist)
  #===================================================================#
  if Activate_mdp_vs_fluence:
    flc_list = np.dot(np.array(c.fluence).reshape(len(c.fluence),1), np.ones(n_simu).reshape(1,n_simu)).flatten()
    mdp_detec_index = np.where(np.isnan(mdp.flatten()), False, np.where(mdp.flatten()>100, False, True))
    no_detec_flc = flc_list[np.logical_not(mdp_detec_index)]
    detec_flc = flc_list[mdp_detec_index]
    detec_mdp = mdp.flatten()[mdp_detec_index]

    distrib, ax1 = plt.subplots(1,1, figsize=(8,6))
    distrib.suptitle("MDP99 as a functin of fluence of detected GRB")
    for val in np.unique(no_detec_flc):
      ax1.axvline(val, ymin=0., ymax=0.01, ms=1, c='black')
    ax1.scatter(detec_flc, detec_mdp, s=3, label=f'Detected GRB \nRatio of detectable polarization : {len(detec_flc)/len(flc_list)}')
    ax1.set(xlabel="fluence (erg.cm-2)", ylabel="MDP99 (%)", yscale='linear', xscale='log', xlim=(10**(int(np.log10(np.min(flc_list)))-1),10**(int(np.log10(np.max(flc_list))))))
    ax1.legend()
    plt.show()


