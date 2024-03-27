import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from funcmod import grb_decra_worldf2satf, horizon_angle, eff_area_compton_func, eff_area_single_func, duty_calc

def fov_const(parfile, num_val=500, dutymode=True, show=True, save=False):
    """
    Plots a map of the sensibility (s_eff) over the sky
      Polarization gives the sensibility to polarization
      Spectrometry gives the sensibility to spectrometry (capacity of detection)
    :param parfile:
    :param num_val:
    :param dutymode:
    :param show:
    :param save:
    """
    # TODO update this file with new functions
    sat_info = []
    inc_list = []
    with open(parfile) as f:
      lines = f.read().split("\n")
    for line in lines:
      if line.startswith("@satellite"):
        temp = [float(e) for e in line.split(" ")[1:]]
        if len(temp) == 3:  # satellite pointing
          dat = [temp[0], temp[1], horizon_angle(temp[2])]
        else:  # satellite orbital parameters
          inc_list.append(temp[0])
          inclination, ohm, omega = map(np.deg2rad, temp[:3])
          thetasat = np.arccos(np.sin(inclination) * np.sin(omega))  # rad
          phisat = np.arctan2(
            (np.cos(omega) * np.sin(ohm) + np.sin(omega) * np.cos(inclination) * np.cos(ohm)),
            (np.cos(omega) * np.cos(ohm) - np.sin(omega) * np.cos(inclination) * np.sin(ohm)))  # rad
          dat = [np.rad2deg(thetasat), np.rad2deg(phisat), horizon_angle(temp[3])]
        sat_info.append(dat)

    path = parfile.split("polGBM")[0]

    n_sat = len(sat_info)
    phi_world = np.linspace(0, 360, num_val)
    # theta will be converted in sat coord with decra2tp, which takes dec in world coord with 0 being north pole and 180 the south pole !
    theta_world = np.linspace(0, 180, num_val)
    detection = np.zeros((n_sat, num_val, num_val))
    detection_pola = np.zeros((n_sat, num_val, num_val))
    detection_spectro = np.zeros((n_sat, num_val, num_val))
    dutyval = 1
    if dutymode:
      for ite in range(n_sat):
        dutyval = duty_calc(inc_list[ite])
        detection[ite] = np.array([[eff_area_compton_func(grb_decra_worldf2satf(theta, phi, sat_info[ite][0],
                                                                                sat_info[ite][1])[0], sat_info[ite][2], func_type="FoV", duty=dutyval) for phi in phi_world] for theta in theta_world])
        detection_pola[ite] = np.array([[eff_area_compton_func(grb_decra_worldf2satf(theta, phi, sat_info[ite][0],
                                                                                     sat_info[ite][1])[0], sat_info[ite][2], func_type="cos", duty=dutyval) for phi in phi_world] for theta in theta_world])
        detection_spectro[ite] = np.array([[eff_area_single_func(grb_decra_worldf2satf(theta, phi, sat_info[ite][0],
                                                                                       sat_info[ite][1])[0], sat_info[ite][2], func_type="data", duty=dutyval) for phi in phi_world] for theta in theta_world])
        # for theta in theta_world:
        #   for phi in phi_world:
        #     print(decra2tp(theta, phi, sat_info[ite])[0], sat_info[ite][2], eff_area_compton_func(decra2tp(theta, phi, sat_info[ite])[0], sat_info[ite][2], func_type="FoV", duty=dutyval))
    else:
      for ite in range(n_sat):
        detection[ite] = np.array([[eff_area_compton_func(grb_decra_worldf2satf(theta, phi, sat_info[ite][0],
                                                                                sat_info[ite][1])[0], sat_info[ite][2], func_type="FoV", duty=dutyval) for phi in phi_world] for theta in theta_world])
        detection_pola[ite] = np.array([[eff_area_compton_func(grb_decra_worldf2satf(theta, phi, sat_info[ite][0],
                                                                                     sat_info[ite][1])[0], sat_info[ite][2], func_type="cos", duty=dutyval) for phi in phi_world] for theta in theta_world])
        detection_spectro[ite] = np.array([[eff_area_single_func(
          grb_decra_worldf2satf(theta, phi, sat_info[ite][0], sat_info[ite][1])[0], sat_info[ite][2], func_type="data", duty=dutyval) for phi in phi_world] for theta in theta_world])

    detec_sum = np.sum(detection, axis=0)
    detec_sum_pola = np.sum(detection_pola, axis=0)
    detec_sum_spectro = np.sum(detection_spectro, axis=0)

    phi_plot, theta_plot = np.meshgrid(np.deg2rad(phi_world), np.deg2rad(theta_world))
    detec_min = int(np.min(detec_sum))
    detec_max = int(np.max(detec_sum))
    detec_min_pola = int(np.min(detec_sum_pola))
    detec_max_pola = int(np.max(detec_sum_pola))
    detec_min_spectro = int(np.min(detec_sum_spectro))
    detec_max_spectro = int(np.max(detec_sum_spectro))
    cmap_fov = mpl.cm.Blues_r
    cmap_pola = mpl.cm.Greens_r
    cmap_spectro = mpl.cm.Oranges_r

    # fov plots 
    # levels_fov = range(int(detec_min / 2) * 2, detec_max + 1)
    levels_fov = range(int(detec_min), int(detec_max) + 1, max(int((int(detec_max) + 1 - int(detec_min)) / 15), 1))

    f1 = plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum, cmap=cmap_fov)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_fov)
    cbar.set_label("Number of satellites seeing this point of the sky", rotation=270, labelpad=20)
    if save:
      print(f"saving {path}fov_noproj")
      plt.savefig(f"{path}fov_noproj")
    if show:
      plt.show()
    else:
      plt.close()

    f1 = plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum, cmap=cmap_fov)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_fov)
    cbar.set_label("Number of satellites seeing this point of the sky", rotation=270, labelpad=20)
    if save:
      print(f"saving {path}fov_proj")
      plt.savefig(f"{path}fov_proj")
    if show:
      plt.show()
    else:
      plt.close()

    # Eff_area plots for polarimetry
    # levels_pola = range(int(detec_min_pola / 2) * 2, detec_max_pola + 1)
    levels_pola = range(int(detec_min_pola), int(detec_max_pola) + 1,
                        max(int((int(detec_max_pola) + 1 - int(detec_min_pola)) / 15), 1))

    f1 = plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_pola, cmap=cmap_pola)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_pola)
    cbar.set_label("Effective area for polarisation (cm²)", rotation=270, labelpad=20)
    #plt.savefig("figtest")
    if save:
      print(f"saving {path}eff_area_noproj_pola")
      plt.savefig(f"{path}eff_area_noproj_pola")
    if show:
      plt.show()
    else:
      plt.close()

    f1 = plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum_pola, cmap=cmap_pola)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_pola)
    cbar.set_label("Effective area for polarisation (cm²)", rotation=270, labelpad=20)
    if save:
      print(f"saving {path}eff_area_proj_pola")
      plt.savefig(f"{path}eff_area_proj_pola")
    if show:
      plt.show()
    else:
      plt.close()

    # Eff_area plots for spectroscopy
    # levels_spectro = range(int(detec_min_spectro / 2) * 2, detec_max_spectro + 1)
    levels_spectro = range(int(detec_min_spectro), int(detec_max_spectro) + 1,
                           max(int((int(detec_max_spectro) + 1 - int(detec_min_spectro)) / 15), 1))

    f1 = plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_spectro, cmap=cmap_spectro)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_spectro)
    cbar.set_label("Effective area for spectrometry (cm²)", rotation=270, labelpad=20)
    if save:
      print(f"saving {path}eff_area_noproj_spectro")
      plt.savefig(f"{path}eff_area_noproj_spectro")
    if show:
      plt.show()
    else:
      plt.close()

    f1 = plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum_spectro, cmap=cmap_spectro)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_spectro)
    cbar.set_label("Effective area for spectrometry (cm²)", rotation=270, labelpad=20)
    if save:
      print(f"saving {path}eff_area_proj_spectro")
      plt.savefig(f"{path}eff_area_proj_spectro")
    if show:
      plt.show()
    else:
      plt.close()

    print(f"Le nombre moyen de satellite voyant une portion quelconque du ciel est de {np.mean(np.mean(detec_sum, axis=1))} ")
    print(f"La surface efficace moyenne pour la polarisation est de {np.mean(np.mean(detec_sum_pola, axis=1))} cm²")
    print(f"La surface efficace moyenne pour la spectrométrie est de {np.mean(np.mean(detec_sum_spectro, axis=1))} cm²")
    with open(f"{path}Mean_values.txt", "w") as file:
      file.write(f"Le nombre moyen de satellite voyant une portion quelconque du ciel est de {np.mean(np.mean(detec_sum, axis=1))} \n")
      file.write(f"La surface efficace moyenne pour la polarisation est de {np.mean(np.mean(detec_sum_pola, axis=1))} cm²\n")
      file.write(f"La surface efficace moyenne pour la spectrométrie est de {np.mean(np.mean(detec_sum_spectro, axis=1))} cm²\n")

savevar = True
# if not savevar:
  # plt.rcParams.update({'font.size': 20})

# fov_const("./polGBM.par", num_val=200, dutymode=True, show=False, save=savevar)

fov_const("esa_prop/Pres_14nov/500km/equatorial/polGBM.par", num_val=500, dutymode=True, show=False, save=savevar)
# fov_const("esa_prop/Pres_14nov/400km/5-5-45/polGBM.par", num_val=500, dutymode=True, show=False, save=savevar)
# fov_const("esa_prop/Pres_14nov/400km/0-45-97/polGBM.par", num_val=500, dutymode=True, show=False, save=savevar)
# fov_const("esa_prop/Pres_14nov/500km/equatorial/polGBM.par", num_val=500, dutymode=True, show=False, save=savevar)
# fov_const("esa_prop/Pres_14nov/500km/5-5-45/polGBM.par", num_val=500, dutymode=True, show=False, save=savevar)
# fov_const("esa_prop/Pres_14nov/500km/0-45-97.5/polGBM.par", num_val=500, dutymode=True, show=False, save=savevar)
