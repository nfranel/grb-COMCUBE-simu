import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from funcmod import decra2tp, horizonAngle, eff_area_pola_func, eff_area_spectro_func, duty_calc

plt.rcParams.update({'font.size': 20})

def fov_const(parfile, num_val=500, dutymode=True, mode="polarization", show=True, save=False):
    """
    Plots a map of the sensibility (s_eff) over the sky
    Mode is the mode used to obtain the sensibility :
      Polarization gives the sensibility to polarization
      Spectrometry gives the sensibility to spectrometry (capacity of detection)
    """
    sat_info = []
    inc_list = []
    with open(parfile) as f:
      lines = f.read().split("\n")
    for line in lines:
      if line.startswith("@satellite"):
        temp = [float(e) for e in line.split(" ")[1:]]
        if len(temp) == 3:  # satellite pointing
          dat = [temp[0], temp[1], horizonAngle(temp[2])]
        else:  # satellite orbital parameters
          inc_list.append(temp[0])
          inclination, ohm, omega = map(np.deg2rad, temp[:3])
          thetasat = np.arccos(np.sin(inclination) * np.sin(omega))  # rad
          phisat = np.arctan2(
            (np.cos(omega) * np.sin(ohm) + np.sin(omega) * np.cos(inclination) * np.cos(ohm)),
            (np.cos(omega) * np.cos(ohm) - np.sin(omega) * np.cos(inclination) * np.sin(ohm)))  # rad
          dat = [thetasat, phisat, horizonAngle(temp[3])]
        sat_info.append(dat)
    n_sat = len(sat_info)
    phi_world = np.linspace(0, 2 * np.pi, num_val)
    # theta will be converted in sat coord with decra2tp, which takes dec in world coord with 0 being north pole and 180 the south pole !
    theta_world = np.linspace(0, np.pi, num_val)
    detection = np.zeros((n_sat, num_val, num_val))
    detection_pola = np.zeros((n_sat, num_val, num_val))
    detection_spectro = np.zeros((n_sat, num_val, num_val))
    dutyval = 1
    if dutymode:
      for ite in range(n_sat):
        dutyval = duty_calc(inc_list[ite])
        print(dutyval)
        detection[ite] = np.array([[eff_area_pola_func(decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="FoV", duty=dutyval) for phi in phi_world] for theta in theta_world])
        detection_pola[ite] = np.array([[eff_area_pola_func(decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="cos", duty=dutyval) for phi in phi_world] for theta in theta_world])
        detection_spectro[ite] = np.array([[eff_area_spectro_func(decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="data", duty=dutyval) for phi in phi_world] for theta in theta_world])
    else:
      for ite in range(n_sat):
        detection[ite] = np.array([[eff_area_pola_func(decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="FoV", duty=dutyval) for phi in phi_world] for theta in theta_world])
        detection_pola[ite] = np.array([[eff_area_pola_func(decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="cos", duty=dutyval) for phi in phi_world] for theta in theta_world])
        detection_spectro[ite] = np.array([[eff_area_spectro_func(decra2tp(theta, phi, sat_info[ite], unit="rad")[0], sat_info[ite][2], func_type="data", duty=dutyval) for phi in phi_world] for theta in theta_world])

    detec_sum = np.sum(detection, axis=0)
    detec_sum_pola = np.sum(detection_pola, axis=0)
    detec_sum_spectro = np.sum(detection_spectro, axis=0)

    phi_plot, theta_plot = np.meshgrid(phi_world, theta_world)
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

    plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum, cmap=cmap_fov)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_fov)
    cbar.set_label("Number of satellites seeing this point of the sky", rotation=270, labelpad=20)
    #plt.savefig("figtest")
    if save:
      plt.savefig("fov_noproj")
    if show:
      plt.show()

    plt.subplot(projection="mollweide")
    h1 = plt.pcolormesh(phi_plot - np.pi, np.pi / 2 - theta_plot, detec_sum, cmap=cmap_fov)
    plt.grid(alpha=0.4)
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_fov)
    cbar.set_label("Number of satellites seeing this point of the sky", rotation=270, labelpad=20)
    if save:
      plt.savefig("fov_proj")
    if show:
      plt.show()

    # Eff_area plots for polarimetry
    # levels_pola = range(int(detec_min_pola / 2) * 2, detec_max_pola + 1)
    levels_pola = range(int(detec_min_pola), int(detec_max_pola) + 1,
                        max(int((int(detec_max_pola) + 1 - int(detec_min_pola)) / 15), 1))

    plt.subplot(projection=None)
    h1 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_pola, cmap=cmap_pola)
    plt.axis('scaled')
    plt.xlabel("Right ascention (rad)")
    plt.ylabel("Declination (rad)")
    cbar = plt.colorbar(ticks=levels_pola)
    cbar.set_label("Effective area for polarisation (cm²)", rotation=270, labelpad=20)
    #plt.savefig("figtest")
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
                           max(int((int(detec_max_spectro) + 1 - int(detec_min_spectro)) / 15), 1))

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

    print(f"Le nombre moyen de satellite voyant une portion quelconque du ciel est de {np.mean(np.mean(detec_sum, axis=1))} cm²")
    print(f"La surface efficace moyenne pour la polarisation est de {np.mean(np.mean(detec_sum_pola, axis=1))} cm²")
    print(f"La surface efficace moyenne pour la spectrométrie est de {np.mean(np.mean(detec_sum_spectro, axis=1))} cm²")

fov_const("sim-const_test/polGBM.par", num_val=500, dutymode=True, mode="polarization", show=True, save=False)