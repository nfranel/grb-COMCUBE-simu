# ================================================================
# Author      : Nathan Franel
# Version     : 1.0
# Created     : 2023-12-01
# Description  :  visualisation.py
# Contains various functions to visualise and calculate quantities that do not depend on GRB simulations
# ================================================================

# Module to have functions to make so usual actions not using GRB simulated data

# Package imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from apexpy import Apex

# Developped modules imports
from src.General.funcmod import read_grbpar, closest_mufile, closest_bkg_info, orbital_period_calc, earth_rotation_offset, true_anomaly_calc, orbitalparam2decra, verif_rad_belts, eff_area_func


############################################################
# Usefull functions :
############################################################
def bkg_data_map(field, bkgdata, altitude, dec_range=np.linspace(0, 180, 181), ra_range=np.linspace(0, 360, 361), language="en", ergcut=(10, 1000), proj="carre", save=False):
  """
  Plots a map of a background quantity over the sky in geographic coordinates.
  The background count rates are obtained by finding the closest background file
  to each geographic position after converting to geomagnetic coordinates.
  :param field: str, field plotted on the map. Accepted values:
      "compton_cr"  - Compton events count rate
      "single_cr"   - single events count rate
      "calor"       - calorimeter count rate
      "dsssd"       - DSSD count rate
      "side"        - side scintillator count rate
      "total_hits"  - total hit count rate
  :param bkgdata: object, background data container obtained with MBkgContainer
  :param altitude: float, altitude for the background simulation [km]
  :param dec_range: np.ndarray, range of declinations for the map [deg], default=np.linspace(0, 180, 181)
  :param ra_range: np.ndarray, range of right ascensions for the map [deg], default=np.linspace(0, 360, 361)
  :param language: str, language for axis labels and legend - "en" (English) or "fr" (French), default="en"
  :param ergcut: tuple, energy window (Emin, Emax) [keV], default=(10, 1000)
  :param proj: str, map projection to use - "carre" (PlateCarree) or "mollweide", default="carre"
  :param save: bool, if True saves the figure to disk instead of displaying it, default=False
  :raises ValueError: if field, language, or proj is not one of the accepted values
  """
  mpl.use("TkAgg")
  x_long, y_lat = np.meshgrid(ra_range, 90 - dec_range)
  field_list = np.zeros((len(dec_range), len(ra_range)))
  apex15 = Apex(date=2025)
  if language == "en":
    if field == "compton_cr":
      item_legend = "Compton events count rate (counts/s)"
      field_index = 0
    elif field == "single_cr":
      item_legend = "single events count rate (counts/s)"
      field_index = 1
    elif field == "calor":
      item_legend = "D2A count rate (counts/s)"
      field_index = 2
    elif field == "dsssd":
      item_legend = "D1A and D1B count rate (counts/s)"
      field_index = 3
    elif field == "side":
      item_legend = "D2B count rate (counts/s)"
      field_index = 4
    elif field == "total_hits":
      item_legend = "hit count rate (counts/s)"
      field_index = 5
    else:
      raise ValueError("Wrong name given for the background field")
  elif language == "fr":
    if field == "compton_cr":
      item_legend = "Taux de comptage d'évènements Compton (coups/s)"
      field_index = 0
    elif field == "single_cr":
      item_legend = "Taux de comptage d'évènements simples (coups/s)"
      field_index = 1
    elif field == "calor":
      item_legend = "Taux de comptage dans le calorimètre (coups/s)"
      field_index = 2
    elif field == "dsssd":
      item_legend = "Taux de comptage dans les DSSD (coups/s)"
      field_index = 3
    elif field == "side":
      item_legend = "Taux de comptage dans le scintillateur de côté (coups/s)"
      field_index = 4
    elif field == "total_hits":
      item_legend = "Taux de comptage total dans les détecteurs (coups/s)"
      field_index = 5
    else:
      raise ValueError("Wrong name given for the background field")
  else:
    raise ValueError("Wrong value given for the language : only en (english) and fr (french) set")

  for row, dec in enumerate(dec_range):
    for col, ra in enumerate(ra_range):
      print(f"bkg_map construction for {field} at {altitude} km : {round((len(ra_range) * row + col) * 100 / (len(dec_range) * len(ra_range)), 0)} %", end="\r")
      lat = 90 - dec
      # Geodetic to apex, scalar input
      mag_lat, mag_lon = apex15.convert(lat, ra, 'geo', 'apex', height=altitude)
      # print(f"init : {lat:.12f}, {ra:.12f}              final : {mag_lat:.12f}, {mag_lon:.12f}")
      mag_dec, mag_ra = 90 - mag_lat, mag_lon
      compton_cr, single_cr, bkg_id = closest_bkg_info(mag_dec, altitude, bkgdata)
      det_count = bkgdata.bkgdf.com_det_stats.values[bkg_id] + bkgdata.bkgdf.sin_det_stats.values[bkg_id]
      side_count = det_count[0] + det_count[1] + det_count[5] + det_count[6] + det_count[10] + det_count[11] + det_count[15] + det_count[16]
      dsssd_count = det_count[2] + det_count[3] + det_count[7] + det_count[8] + det_count[12] + det_count[13] + det_count[17] + det_count[18]
      calor_count = det_count[4] + det_count[9] + det_count[14] + det_count[19]
      total_hits = np.sum(det_count)
      bkg_values = [compton_cr, single_cr, calor_count / bkgdata.sim_time, dsssd_count / bkgdata.sim_time, side_count / bkgdata.sim_time, total_hits / bkgdata.sim_time]
      field_list[row][col] = bkg_values[field_index]
  fontsize = 15
  plt.rcParams.update({'font.size': fontsize})
  if proj == "carre":
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
  elif proj == "mollweide":
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Mollweide()}, figsize=(10, 6))
  else:
    raise ValueError("Use a correct value for proj 'carre' or 'mollweide'")
  p1 = ax.pcolormesh(x_long, y_lat, field_list, cmap="Blues", transform=ccrs.PlateCarree())
  ax.grid(True)
  ax.set_global()
  ax.coastlines()
  cbar = fig.colorbar(p1)
  if language == "en":
    ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)")
    cbar.set_label(f"Background {item_legend}", rotation=270, labelpad=20, fontsize=fontsize)
  elif language == "fr":
    ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)")
    cbar.set_label(f"{item_legend}", rotation=270, labelpad=20, fontsize=fontsize)
  else:
    raise ValueError("Wrong value given for the language : only en (english) and fr (french) set")
  plt.tight_layout()
  if save:
    plt.savefig(f"bkg_map-{field}-alt_{altitude}-{ergcut[0]}_{ergcut[1]}-{proj}")
    plt.close(fig)
  else:
    plt.show()
  print(f"bkg_map construction done for {field} at {altitude} km, results save : {save}")


def mu100_data_map(mu100data, theta_sat=np.linspace(0, 114, 115), phi_sat=np.linspace(0, 360, 181)):
  """
  Plots multiple maps and graphs of the mu100 modulation factor and effective areas
  as a function of the source direction in the satellite frame.
  Displays smoothed and unsmoothed maps of mu100, Compton effective area, and single
  event effective area, as well as their variation with declination.
  :param mu100data: object, mu100 data container with a mudf DataFrame attribute
  :param theta_sat: np.ndarray, polar angle (declination) grid in satellite frame [deg],
      default=np.linspace(0, 114, 115)
  :param phi_sat: np.ndarray, azimuthal angle (right ascension) grid in satellite frame [deg],
      default=np.linspace(0, 360, 181)
  """
  mpl.use("Qt5Agg")

  fontsize = 15
  plt.rcParams.update({'font.size': fontsize})

  nrows = len(theta_sat)
  ncols = len(phi_sat)
  mu100list = np.zeros((nrows, ncols))
  seff_com_list = np.zeros((nrows, ncols))
  seff_sin_list = np.zeros((nrows, ncols))
  for i, theta in enumerate(theta_sat):
    for j, phi in enumerate(phi_sat):
      mu_ret = closest_mufile(theta, phi, mu100data)
      mu100list[i, j] = mu_ret[1]
      seff_com_list[i, j] = mu_ret[3]
      seff_sin_list[i, j] = mu_ret[4]
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
      smooth_mu100list[i, j] = (mu100list[i, np.mod(j - 1, ncols)] + mu100list[i, j] + mu100list[i, np.mod(j + 1, ncols)]) / 3
      v2smooth_mu100list[i, j] = (mu100list[i, np.mod(j - 2, ncols)] + mu100list[i, np.mod(j - 1, ncols)] + mu100list[i, j] + mu100list[i, np.mod(j + 1, ncols)] + mu100list[i, np.mod(j + 2, ncols)]) / 5
      smooth_seff_com_list[i, j] = (seff_com_list[i, np.mod(j - 1, ncols)] + seff_com_list[i, j] + seff_com_list[i, np.mod(j + 1, ncols)]) / 3
      smooth_seff_sin_list[i, j] = (seff_sin_list[i, np.mod(j - 1, ncols)] + seff_sin_list[i, j] + seff_sin_list[i, np.mod(j + 1, ncols)]) / 3

  x_mu, y_mu = np.meshgrid(phi_sat, 90 - theta_sat)

  xgrid, ygrid = np.arange(-135, 136, 45), np.arange(-30, 90, 30)
  # # Seff maps
  # fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
  # plt.suptitle("Compton event effective area map")
  # p1 = ax.pcolormesh(x_mu, y_mu, seff_com_list, cmap="Blues", transform=ccrs.PlateCarree())
  # cbar = fig.colorbar(p1, ax=ax)
  # cbar.set_label("Compton event effective area (cm²)", rotation=270, labelpad=20, fontsize=12)
  # ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)")
  # ax.gridlines(draw_labels=False, xlocs=xgrid, ylocs=ygrid)
  # ax.text(0, 90, ' 90°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(180, 60, ' 60°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(180, 30, ' 30°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(180, 0, '0°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(0, -35, '0°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(45, -35, '45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(90, -35, '90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(135, -35, '135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(-45, -35, '-45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(-90, -35, '-90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(-135, -35, '-135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # plt.show()
  #
  # fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
  # plt.suptitle("Single event effective area map")
  # p2 = ax.pcolormesh(x_mu, y_mu, seff_sin_list, cmap="Blues", transform=ccrs.PlateCarree())
  # cbar = fig.colorbar(p2, ax=ax)
  # cbar.set_label("Single event effective area (cm²)", rotation=270, labelpad=20, fontsize=12)
  # ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)")
  # ax.gridlines(draw_labels=False, xlocs=xgrid, ylocs=ygrid)
  # ax.text(0, 90, ' 90°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(180, 60, ' 60°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(180, 30, ' 30°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(180, 0, '0°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(0, -35, '0°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(45, -35, '45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(90, -35, '90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(135, -35, '135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(-45, -35, '-45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(-90, -35, '-90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(-135, -35, '-135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # plt.show()

  # Smoothed Seff maps
  fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
  # plt.suptitle("Smoothed Compton event effective area map")
  p3 = ax.pcolormesh(x_mu, y_mu, smooth_seff_com_list, cmap="Blues", transform=ccrs.PlateCarree())
  cbar = fig.colorbar(p3, ax=ax)
  cbar.set_label("Compton event effective area (cm²)", rotation=270, labelpad=20)
  ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)")
  ax.gridlines(draw_labels=False, xlocs=xgrid, ylocs=ygrid)
  ax.text(0, 90, ' 90°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 60, ' 60°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 30, ' 30°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 0, '0°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(0, -35, '0°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(45, -35, '45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(90, -35, '90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(135, -35, '135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-45, -35, '-45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-90, -35, '-90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-135, -35, '-135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  plt.tight_layout()
  plt.show()

  fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
  # plt.suptitle("Smoothed single event effective area map")
  p4 = ax.pcolormesh(x_mu, y_mu, smooth_seff_sin_list, cmap="Blues", transform=ccrs.PlateCarree())
  cbar = fig.colorbar(p4, ax=ax)
  cbar.set_label("Single event effective area (cm²)", rotation=270, labelpad=20)
  ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)")
  ax.gridlines(draw_labels=False, xlocs=xgrid, ylocs=ygrid)
  ax.text(0, 90, ' 90°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 60, ' 60°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 30, ' 30°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 0, '0°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(0, -35, '0°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(45, -35, '45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(90, -35, '90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(135, -35, '135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-45, -35, '-45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-90, -35, '-90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-135, -35, '-135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  plt.tight_layout()
  plt.show()

  # Seff vs declination
  fig, ax = plt.subplots(figsize=(10, 6))
  # plt.suptitle("Variation of Compton event effective area with declination")
  ax.plot(theta_sat, seff_com_vs_dec)
  ax.set(xlabel="Declination (°)", ylabel="Compton event effective area (cm²)")
  ax.grid()
  plt.tight_layout()
  plt.show()

  fig, ax = plt.subplots(figsize=(10, 6))
  # plt.suptitle("Variation of single event effective area with declination")
  ax.plot(theta_sat, seff_sin_vs_dec)
  ax.set(xlabel="Declination (°)", ylabel="Single event effective area (cm²)")
  ax.grid()
  plt.tight_layout()
  plt.show()

  fig, ax1 = plt.subplots(figsize=(10, 6))
  # plt.suptitle("Variation of single and Compton event effective area with declination")
  c1 = ax1.plot(theta_sat, seff_sin_vs_dec, label="Single events", color="blue")[0]
  ax1.set_xlabel("Declination (°)")
  ax1.set_ylabel("Single event effective area (cm²)", color="blue")
  ax1.tick_params(axis='y', labelcolor="blue")
  ax1.grid()
  ax2 = ax1.twinx()
  c2 = ax2.plot(theta_sat, seff_com_vs_dec, label="Compton events", color="red")[0]
  ax2.set_ylabel("Compton event effective area (cm²)", color="red")
  ax2.tick_params(axis='y', labelcolor="red")
  # ax2.grid()
  ax1.legend([c1, c2], [c1.get_label(), c2.get_label()])
  fig.tight_layout()
  plt.show()

  # mu100 maps
  fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
  # plt.suptitle(r"$\mu_{100}$ map")
  p5 = ax.pcolormesh(x_mu, y_mu, mu100list, cmap="Blues", transform=ccrs.PlateCarree())
  cbar = fig.colorbar(p5, ax=ax)
  cbar.set_label(r"$\mu_{100}$ values", rotation=270, labelpad=20)
  ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)")
  ax.gridlines(draw_labels=False, xlocs=xgrid, ylocs=ygrid)
  ax.text(0, 90, ' 90°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 60, ' 60°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 30, ' 30°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 0, '0°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(0, -35, '0°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(45, -35, '45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(90, -35, '90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(135, -35, '135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-45, -35, '-45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-90, -35, '-90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-135, -35, '-135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  plt.tight_layout()
  plt.show()

  # Smoothed mu100 maps
  fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
  # plt.suptitle(r"Smoothed $\mu_{100}$ map")
  p6 = ax.pcolormesh(x_mu, y_mu, smooth_mu100list, cmap="Blues", transform=ccrs.PlateCarree())
  cbar = fig.colorbar(p6, ax=ax, pad=0.1)
  cbar.set_label(r"$\mu_{100}$ values", rotation=270, labelpad=20)
  ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)")
  ax.gridlines(draw_labels=False, xlocs=xgrid, ylocs=ygrid)
  ax.text(0, 90, ' 90°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 60, ' 60°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 30, ' 30°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(180, 0, '0°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  ax.text(0, -35, '0°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(45, -35, '45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(90, -35, '90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(135, -35, '135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-45, -35, '-45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-90, -35, '-90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  ax.text(-135, -35, '-135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  plt.tight_layout()
  plt.show()

  # fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)}, figsize=(10, 6))
  # plt.suptitle(r"Smoothed $\mu_{100}$ map")
  # p7 = ax.pcolormesh(x_mu, y_mu, v2smooth_mu100list, cmap="Blues", transform=ccrs.PlateCarree())
  # cbar = fig.colorbar(p7, ax=ax)
  # cbar.set_label(r"$\mu_{100}$ values", rotation=270, labelpad=20, fontsize=12)
  # ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)")
  # ax.gridlines(draw_labels=False, xlocs=xgrid, ylocs=ygrid)
  # ax.text(0, 90, ' 90°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(180, 60, ' 60°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(180, 30, ' 30°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(180, 0, '0°\n\n', transform=ccrs.PlateCarree(), ha="center", va='center')
  # ax.text(0, -35, '0°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(45, -35, '45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(90, -35, '90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(135, -35, '135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(-45, -35, '-45°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(-90, -35, '-90°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # ax.text(-135, -35, '-135°', transform=ccrs.PlateCarree(), ha="center", va='baseline')
  # plt.show()

  # mu100 vs declination
  fig, ax = plt.subplots(figsize=(10, 6))
  # plt.suptitle(r"Variation of $\mu_{100}$ with declination")
  ax.plot(theta_sat, mu100_vs_dec)
  ax.set(xlabel="Declination (°)", ylabel=r"$\mu_{100}$ value")
  ax.grid()
  plt.tight_layout()
  plt.show()


def magnetic_latitude_convert(altitude, lat_range=np.linspace(90, -90, 361), lon_range=np.linspace(0, 360, 361)):
  """
  Plots maps of geomagnetic latitudes derived from geographic coordinates at a given altitude,
  using both a scatter plot and a contour plot representation.
  :param altitude: float, altitude for the geomagnetic conversion [km]
  :param lat_range: np.ndarray, range of geographic latitudes [deg], default=np.linspace(90, -90, 361)
  :param lon_range: np.ndarray, range of geographic longitudes [deg], default=np.linspace(0, 360, 361)
  """
  apex15 = Apex(date=2025)
  # WITH SCATTER
  mag_lat_list = []
  lat_scat = []
  lon_scat = []
  fig1, ax1 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
  for lon in lon_range:
    for lat in lat_range:
      mag_lat_list.append(apex15.convert(lat, lon, 'geo', 'apex', height=altitude)[0])
      lat_scat.append(lat)
      lon_scat.append(lon)
  sc = ax1.scatter(lon_scat, lat_scat, c=mag_lat_list, cmap="viridis", s=5)
  cbar1 = fig1.colorbar(sc, ax=ax1, orientation='vertical', shrink=0.7, pad=0.05)
  cbar1.set_label("Valeur de z")
  cbar1.set_label(f"Geomagnetic latitudes (deg)", rotation=270, labelpad=20)
  ax1.coastlines()
  ax1.set_global()
  ax1.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title=f"Lines of constant geomagnetic latitudes")
  plt.show()

  # WITH CONTOUR
  x_lat, y_lat = np.meshgrid(lon_range, lat_range)
  mag_lat = np.zeros((len(lat_range), len(lon_range)))
  for ite_lat, lat in enumerate(lat_range):
    for ite_lon, lon in enumerate(lon_range):
      mag_lat[ite_lat, ite_lon] = apex15.convert(lat, lon, 'geo', 'apex', height=altitude)[0]
  fig2, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
  levels = np.linspace(-90, 90, 19)
  p1 = ax2.contour(x_lat, y_lat, mag_lat, levels=levels)
  cbar2 = fig2.colorbar(p1)
  cbar2.set_label(f"Geomagnetic latitudes (deg)", rotation=270, labelpad=20)
  cbar2.set_ticks(levels)
  ax2.coastlines()
  ax2.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title=f"Lines of constant geomagnetic latitudes")
  plt.show()


def orbitalparam2cartesian(i, ohm, omega):
  """
  Calculates the cartesian coordinates of a point defined by orbital parameters (at a unit distance from the center)
  :param i: float, inclination of the orbit [deg]
  :param ohm: float, longitude/ra of the ascending node of the orbit [deg]
  :param omega: float or np.ndarray, argument of periapsis of the orbit [deg]
  :returns: float or np.ndarray, float or np.ndarray, float or np.ndarray, x coordinate, y coordinate, z coordinate
  """
  x = np.cos(omega) * np.cos(ohm) - np.sin(omega) * np.cos(i) * np.sin(ohm)
  y = np.cos(omega) * np.sin(ohm) + np.sin(omega) * np.cos(i) * np.cos(ohm)
  z = np.sin(omega) * np.sin(i)
  return x, y, z


def orbit(inc, ohm, omega, nsat, alt):
  """
  Plots a 3D representation of a constellation of satellites on their respective orbits,
  together with a wireframe Earth for spatial reference.
  :param inc: list, inclination of the orbits [deg]
  :param ohm: list, longitude/ra of the ascending node of the orbits [deg]
  :param omega: float, argument of periapsis common to all orbits [deg]
  :param nsat: list, number of satellites per orbit
  :param alt: float, altitude of the orbits [km]
  """
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.plot([0, 1], [0, 0], [0, 0], c="black")
  ax.plot([0, 0], [0, 1], [0, 0], c="black")
  ax.plot([0, 0], [0, 0], [0, 1], c="black")
  ax.axis('off')
  ax.grid(False)

  r = (6371 - alt) / 6371
  x = []
  y = []
  z = []
  n_ite_eq = 100
  for theta in np.linspace(-np.pi/2, np.pi/2, int(n_ite_eq/2)):
    if theta == -np.pi/2 or theta == np.pi/2:
      phi_list = [0]
    else:
      phi_list = np.linspace(0, 2 * np.pi, int(np.cos(theta) * n_ite_eq))
    for phi in phi_list:
      x.append(r * np.cos(phi) * np.cos(theta))
      y.append(r * np.sin(phi) * np.cos(theta))
      z.append(r * np.sin(theta))
  ax.scatter(x, y, z, s=1, c='slategrey', alpha=0.1)

  x1, y1, z1 = orbitalparam2cartesian(np.deg2rad(inc[0]), np.deg2rad(ohm[0]), omega + np.linspace(0, 2 * np.pi, nsat[0], endpoint=False))
  x2, y2, z2 = orbitalparam2cartesian(np.deg2rad(inc[1]), np.deg2rad(ohm[1]), omega + np.linspace(0 + 2 * np.pi / (3 * nsat[1]), 2 * np.pi + 2 * np.pi / (3 * nsat[1]), nsat[1], endpoint=False))
  x3, y3, z3 = orbitalparam2cartesian(np.deg2rad(inc[2]), np.deg2rad(ohm[2]), omega + np.linspace(0 + (2 * np.pi / (3 * nsat[1])) * 2, 2 * np.pi + (2 * np.pi / (3 * nsat[1])) * 2, nsat[2], endpoint=False))

  xl1, yl1, zl1 = orbitalparam2cartesian(np.deg2rad(inc[0]), np.deg2rad(ohm[0]), np.linspace(0, 2 * np.pi, 100))
  xl2, yl2, zl2 = orbitalparam2cartesian(np.deg2rad(inc[1]), np.deg2rad(ohm[1]), np.linspace(0, 2 * np.pi, 100))
  xl3, yl3, zl3 = orbitalparam2cartesian(np.deg2rad(inc[2]), np.deg2rad(ohm[2]), np.linspace(0, 2 * np.pi, 100))

  ax.scatter3D(x1, y1, z1, s=40, c="blue")
  ax.scatter3D(x2, y2, z2, s=40, c='cornflowerblue')
  ax.scatter3D(x3, y3, z3, s=40, c='darkblue')
  ax.plot(xl1, yl1, zl1, c="blue", label=f'Inclination : {inc[0]}° \nRA of the ascending node : {ohm[0]}°')
  ax.plot(xl2, yl2, zl2, c=f'cornflowerblue', label=f'Inclination : {inc[1]}° \nRA of the ascending node : {ohm[1]}°')
  ax.plot(xl3, yl3, zl3, c=f'darkblue', label=f'Inclination : {inc[2]}° \nRA of the ascending node : {ohm[2]}°')
  ax.legend(bbox_to_anchor = (0.1, 1.15), loc='upper left')
  ax.set(xlabel="x", ylabel="y", zlabel="z")
  ax.set_box_aspect([1.0, 1.0, 1.0])
  plt.show()


def trajectory(inc, ohm, nsat, alt, zonetype="all", omega=0, projection="carre"):
  """
  Plots the trajectory of one or more satellite orbits on a 2D sky map,
  along with their satellite positions and the exclusion zones due to radiation belts.
  :param inc: list, inclination of the orbits [deg]
  :param ohm: list, longitude/ra of the ascending node of the orbits [deg]
  :param nsat: list, number of satellites per orbit
  :param alt: float, altitude of the orbits [km]
  :param zonetype: str, name of the exclusion zone considered - "all", "electron" or "proton", default="all"
  :param omega: float, argument of periapsis of the orbits [deg], default=0
  :param projection: str, type of map projection to use - "carre" (PlateCarree) or "mollweide", default="carre"
  :raises ValueError: if projection is not one of the accepted values
  """
  # Create the plot with the correct projection
  if projection == "mollweide":
    proj = ccrs.Mollweide(central_longitude=0)
  elif projection == "carre":
    proj = ccrs.PlateCarree(central_longitude=0)
  else:
    print("Please use either carre or mollweide for projection")
    return None

  fig, ax = plt.subplots(subplot_kw={'projection': proj})
  ax.set_global()

  theta_verif = np.linspace(0, 180, 181)
  phi_verif = np.linspace(0, 360, 360, endpoint=False)
  plottitle = f"Map of {zonetype} radiation belt at {alt} km"
  ax.set(title=plottitle)

  cancel_theta = []
  cancel_phi = []
  for theta in theta_verif:
    for phi in phi_verif:
      if verif_rad_belts(theta, phi, alt, zonetype=zonetype):
        cancel_theta.append(90 - theta)
        cancel_phi.append(phi if phi <= 180 else phi % 180 - 180)
  cancel_theta = np.array(cancel_theta)
  cancel_phi = np.array(cancel_phi)
  ax.scatter(cancel_phi, cancel_theta, s=1, transform=ccrs.PlateCarree())

  latitude_list = []
  phi_list = []
  latitude_line = []
  phi_line = []
  colors = ["royalblue", "blue", "navy", "lightsteelblue", "cornflowerblue"]
  # Creating the lists of sat coordinates
  for ite_orbit in range(len(inc)):
    temp_theta = []
    temp_phi = []
    starting = 360/nsat[ite_orbit] / len(inc) * ite_orbit  # offset in the repartition of sat of different orbits so that there is a constant repartition in omega and that the first sat of each orbit are not at the same place
    for ite_sat in range(nsat[ite_orbit]):
      separation = 360/nsat[ite_orbit] * ite_sat  # separation between sats of a same orbit
      theta, phi = orbitalparam2decra(inc[ite_orbit], ohm[ite_orbit], omega + separation + starting)
      temp_theta.append(90 - theta)
      temp_phi.append(np.mod(phi, 360))
    line_temp_theta, line_temp_phi = orbitalparam2decra(inc[ite_orbit], ohm[ite_orbit], np.linspace(0, 360, 100))
    latitude_line.append(90 - line_temp_theta)
    phi_line.append(np.mod(line_temp_phi, 360))
    latitude_list.append(temp_theta)
    phi_list.append(temp_phi)
  latitude_list = np.array(latitude_list)
  phi_list = np.array(phi_list)
  latitude_line = np.array(latitude_line)
  phi_line = np.array(phi_line)
  for ite_orbit in range(len(inc)):
    ax.scatter(phi_list[ite_orbit], latitude_list[ite_orbit], color=colors[ite_orbit], transform=ccrs.PlateCarree())
    ax.scatter(phi_line[ite_orbit], latitude_line[ite_orbit], s=1, color=colors[ite_orbit], transform=ccrs.PlateCarree())

  # Adding the coasts
  ax.coastlines()
  plt.show()


def calc_duty(inc, ohm, omega, alt, show=False, show_sat=False):
  """
  Calculates the duty cycle of a satellite orbit, defined as the fraction of time
  spent outside the radiation belt exclusion zones. Optionally plots the trajectory
  and exclusion zones on a geographic map.
  :param inc: float, inclination of the orbit [deg]
  :param ohm: float, longitude/ra of the ascending node of the orbit [deg]
  :param omega: float, argument of periapsis of the orbit [deg]
  :param alt: float, altitude of the orbit [km]
  :param show: bool, if True plots the satellite trajectory and exclusion zones on a map, default=False
  :param show_sat: bool, if True overlays the accepted and rejected satellite positions on the map
      (only used when show=True), default=False
  :returns: float, duty cycle as a fraction in [0, 1]
  """
  plt.rcParams.update({'font.size': 15})
  orbit_period = orbital_period_calc(alt)
  n_orbit = 1000
  n_val_per_orbit = 100
  time_vals = np.linspace(0, n_orbit * orbit_period, n_orbit * n_val_per_orbit)
  earth_ra_offset = earth_rotation_offset(time_vals)
  true_anomalies = true_anomaly_calc(time_vals, orbit_period)

  counter = 0
  lat_list = []
  long_list = []
  lat_list_rej = []
  long_list_rej = []
  countrej = 0
  for ite in range(len(time_vals)):
    decsat, rasat = orbitalparam2decra(inc, ohm, omega, nu=true_anomalies[ite])
    rasat = np.mod(rasat - earth_ra_offset[ite], 360)
    if not verif_rad_belts(decsat, rasat, alt):
      lat_list.append(90 - decsat)
      long_list.append(rasat)
      counter += 1
    else:
      lat_list_rej.append(90 - decsat)
      long_list_rej.append(rasat)
      countrej += 1
  print(f"Duty cycle for inc : {inc}, ohm : {ohm}, omega : {omega} for all files at {alt}")
  print(f"   === {counter / len(time_vals)}")

  if show:
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}, figsize=(15, 8))
    ax.set_global()

    dec_points = np.linspace(0, 180, 361)
    ra_points = np.linspace(0, 360, 720, endpoint=False)
    plottitle = f"All radiation belt {alt}km"
    # plottitle = f"Proton radiation belt {alt}km"
    nite = len(dec_points) * len(ra_points)
    ncount = 0
    cancel_dec = []
    cancel_ra = []
    for dec_p in dec_points:
      for ra_p in ra_points:
        ncount += 1
        if verif_rad_belts(dec_p, ra_p, alt):
          cancel_dec.append(dec_p)
          cancel_ra.append(ra_p)
        print(f"Calculation : {int(ncount / nite * 100)}%", end="\r")
    print("Calculation over")
    cancel_dec = 90 - np.array(cancel_dec)
    cancel_ra = np.array(cancel_ra)
    ax.scatter(cancel_ra, cancel_dec, s=1, color="blue")
    ax.set(title=plottitle)
    # Adding the coasts
    ax.coastlines()
    # print(cancel_dec)
    # print(cancel_ra)

    colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
    # Displaying the satellites (rejected and not rejected)
    if show_sat:
      ax.scatter(long_list, lat_list, s=1, color=colors[2])
      ax.scatter(long_list_rej, lat_list_rej, s=1, color="black")

    plt.show()
  return counter / len(time_vals)


def duty_variation_plot(alt=500):
  """
  Plots the duty cycle as a function of orbit inclination for a given altitude,
  sweeping inclinations from 0° to 98° in 0.25° steps.
  :param alt: float, altitude of the orbit [km], default=500
  """
  mpl.use("Qt5Agg")

  inclinations = np.linspace(0, 98, 401)
  duties = [100 * calc_duty(inc, 0, 0, alt) for inc in inclinations]

  fig, ax = plt.subplots(1, 1, figsize=(10, 6))
  ax.plot(inclinations, duties)
  ax.set(xlabel="Orbit inclination (°)", ylabel="Duty cycle (%)")
  plt.show()


def show_non_op_area(alt, zonetype="all"):
  """
  Plots a geographic map of the radiation belt exclusion zones at a given altitude,
  showing all positions where satellite operations are not allowed.
  :param alt: float, altitude of the verification [km]
  :param zonetype: str, type of radiation belt to display - "all", "electron" or "proton", default="all"
  """
  mpl.use("Qt5Agg")
  fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
  ax.set_global()

  theta_verif = np.linspace(0, 180, 181)
  phi_verif = np.linspace(0, 360, 360, endpoint=False)
  plottitle = f"Map of {zonetype} radiation belt at {alt} km"

  cancel_theta = []
  cancel_phi = []
  for theta in theta_verif:
    for phi in phi_verif:
      if verif_rad_belts(theta, phi, alt, zonetype=zonetype):
        cancel_theta.append(90 - theta)
        cancel_phi.append(phi if phi <= 180 else phi % 180 - 180)
  cancel_theta = np.array(cancel_theta)
  cancel_phi = np.array(cancel_phi)
  ax.scatter(cancel_phi, cancel_theta, s=1)
  ax.set(title=plottitle)
  # Adding the coasts
  ax.coastlines()
  plt.show()


def fov_const(parfile, museffdata, num_val=500, erg_cut=(10, 1000), show=True, save=False):
  """
  Plots maps of the sky coverage and effective area sensitivity for a satellite constellation.
  Produces three maps: number of satellites in sight, Compton effective area, and single
  event effective area, as well as printing the mean values of each quantity weighted
  by the solid angle of each sky cell.
  :param parfile: str, path to the constellation parameter file
  :param museffdata: object, mu100 and effective area data container
  :param num_val: int, number of angular grid points along each axis, default=500
  :param erg_cut: tuple, energy window (Emin, Emax) [keV], default=(10, 1000)
  :param show: bool, if True displays the figures interactively, default=True
  :param save: bool, if True saves the figures to disk instead of displaying, default=False
  :raises ValueError: if chosen_proj is not "carre" or "mollweide"
  """
  plt.rcParams.update({'font.size': 15})
  xlab = "Right ascention (°)"
  ylab = "Declination (°)"
  title1 = "Constellation sky coverage map"
  title2 = "Constellation sky sensitivity map for Compton events"
  title3 = "Constellation sky sensitivity map for single events"
  bar1 = "Number of satellites covering the area"
  bar2 = "Effective area for Compton events (cm²)"
  bar3 = "Effective area for single events (cm²)"
  # chosen_proj = "carre"
  chosen_proj = "mollweide"

  suffix = parfile.split("/")[-1].split("polGBM.par")[0]

  sat_info = read_grbpar(parfile)[-1]
  n_sat = len(sat_info)
  # museffdata = MuSeffContainer(mu100par, erg_cut, armcut)

  phi_world = np.linspace(0, 360, num_val, endpoint=False)
  # theta will be converted in sat coord with grb_decra_worldf2satf, which takes dec in world coord with 0 being north pole and 180 the south pole !
  theta_world = np.linspace(0, 180, num_val)
  detection = np.zeros((n_sat, num_val, num_val))
  detection_compton = np.zeros((n_sat, num_val, num_val))
  detection_single = np.zeros((n_sat, num_val, num_val))

  # cancel_theta = np.zeros((num_val, num_val))
  # cancel_phi = np.zeros((num_val, num_val))
  # ok_theta = np.zeros((num_val, num_val))
  # ok_phi = np.zeros((num_val, num_val))

  nite = num_val ** 2 * len(sat_info)
  ncount = 0
  for ite, info_sat in enumerate(sat_info):
    for ite_theta, theta in enumerate(theta_world):
      for ite_phi, phi in enumerate(phi_world):
        #
        # if ite == 0:
        #   if verif_rad_belts(theta, phi, info_sat[3], zonetype="all"):
        #     cancel_theta[ite_theta][ite_phi] = (90 - theta)
        #     cancel_phi[ite_theta][ite_phi] = (phi if phi <= 180 else phi % 180 - 180)
        #   else:
        #     ok_theta[ite_theta][ite_phi] = (90 - theta)
        #     ok_phi[ite_theta][ite_phi] = (phi if phi <= 180 else phi % 180 - 180)
        ncount += 1
        detection_compton[ite][ite_theta][ite_phi], detection_single[ite][ite_theta][ite_phi], detection[ite][ite_theta][ite_phi] = eff_area_func(theta, phi, info_sat, museffdata)
        print(f"Calculation : {int(ncount / nite * 100)}%", end="\r")
  print("Calculation over")

  detec_sum = np.sum(detection, axis=0)
  detec_sum_compton = np.sum(detection_compton, axis=0)
  detec_sum_single = np.sum(detection_single, axis=0)

  phi_plot, theta_plot = np.meshgrid(np.where(phi_world <= 180, phi_world, phi_world % 180 - 180), 90 - theta_world)
  detec_min = int(np.min(detec_sum))
  detec_max = int(np.max(detec_sum))
  detec_min_compton = int(np.min(detec_sum_compton))
  detec_max_compton = int(np.max(detec_sum_compton))
  detec_min_single = int(np.min(detec_sum_single))
  detec_max_single = int(np.max(detec_sum_single))
  cmap_det = mpl.cm.Blues_r
  cmap_compton = mpl.cm.Greens_r
  cmap_single = mpl.cm.Oranges_r

  ##################################################################################################################
  # Map for number of satellites in sight
  ##################################################################################################################
  levels = range(detec_min, detec_max + 1, max(1, int((detec_max + 1 - detec_min) / 15)))

  if chosen_proj == "carre":
    fig1, ax1 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
  elif chosen_proj == "mollweide":
    fig1, ax1 = plt.subplots(subplot_kw={'projection': ccrs.Mollweide()}, figsize=(10, 6))
  else:
    raise ValueError("Use a correct value for proj 'carre' or 'mollweide'")
  # ax1.set_global()
  # ax1.coastlines()
  h1 = ax1.pcolormesh(phi_plot, theta_plot, detec_sum, cmap=cmap_det, transform=ccrs.PlateCarree())
  # ax1.axis('scaled')
  # ax1.set(xlabel=xlab, ylabel=ylab, title=title1)
  ax1.set(xlabel=xlab, ylabel=ylab)
  # ax1.scatter(cancel_phi, cancel_theta, s=1, color="red", transform=ccrs.PlateCarree())
  # ax1.scatter(ok_phi, ok_theta, s=1, color="green", transform=ccrs.PlateCarree())

  cbar = fig1.colorbar(h1, ticks=levels)
  cbar.set_label(bar1, rotation=270, labelpad=20)
  plt.tight_layout()
  if save:
    fig1.savefig(f"{suffix}in_sight_erg{erg_cut[0]}-{erg_cut[1]}")
    plt.close(fig1)
  if show:
    plt.show()

  ##################################################################################################################
  # Map of constellation's compton effective area
  ##################################################################################################################
  levels_compton = range(detec_min_compton, detec_max_compton + 1, max(1, int((detec_max_compton + 1 - detec_min_compton) / 15)))

  if chosen_proj == "carre":
    fig2, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
  elif chosen_proj == "mollweide":
    fig2, ax2 = plt.subplots(subplot_kw={'projection': ccrs.Mollweide()}, figsize=(10, 6))
  else:
    raise ValueError("Use a correct value for proj 'carre' or 'mollweide'")
  # ax2.set_global()
  # ax2.coastlines()
  h3 = ax2.pcolormesh(phi_plot, theta_plot, detec_sum_compton, cmap=cmap_compton, transform=ccrs.PlateCarree())
  # ax2.axis('scaled')
  # ax2.set(xlabel=xlab, ylabel=ylab, title=title2)
  ax2.set(xlabel=xlab, ylabel=ylab)
  cbar = fig2.colorbar(h3, ticks=levels_compton)
  cbar.set_label(bar2, rotation=270, labelpad=20)
  plt.tight_layout()
  if save:
    fig2.savefig(f"{suffix}compton_seff_erg{erg_cut[0]}-{erg_cut[1]}")
    plt.close(fig2)
  if show:
    plt.show()

  ##################################################################################################################
  # Map of constellation's single effective area
  ##################################################################################################################
  levels_single = range(detec_min_single, detec_max_single + 1, max(1, int((detec_max_single + 1 - detec_min_single) / 15)))

  if chosen_proj == "carre":
    fig3, ax3 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
  elif chosen_proj == "mollweide":
    fig3, ax3 = plt.subplots(subplot_kw={'projection': ccrs.Mollweide()}, figsize=(10, 6))
  else:
    raise ValueError("Use a correct value for proj 'carre' or 'mollweide'")
  # ax3.set_global()
  # ax3.coastlines()
  h5 = ax3.pcolormesh(phi_plot, theta_plot, detec_sum_single, cmap=cmap_single, transform=ccrs.PlateCarree())
  # ax3.axis('scaled')
  # ax3.set(xlabel=xlab, ylabel=ylab, title=title3)
  ax3.set(xlabel=xlab, ylabel=ylab)
  cbar = fig3.colorbar(h5, ticks=levels_single)
  cbar.set_label(bar3, rotation=270, labelpad=20)
  plt.tight_layout()
  if save:
    fig3.savefig(f"{suffix}single_seff_erg{erg_cut[0]}-{erg_cut[1]}")
    plt.close(fig3)
  if show:
    plt.show()

  correction_values = (1 + np.sin(np.deg2rad(theta_world)) * (num_val - 1)) / num_val

  print(f"The mean number of satellites in sight is :       {np.average(np.mean(detec_sum, axis=1), weights=correction_values):.4f} satellites")
  print(f"The mean effective area for Compton events is :  {np.average(np.mean(detec_sum_compton, axis=1), weights=correction_values):.4f} cm²")
  print(f"The mean effective area for single events is :   {np.average(np.mean(detec_sum_single, axis=1), weights=correction_values):.4f} cm²")


def get_mean_seff_values(parfile, museffdata, num_val=100, erg_cut=(10, 1000)):
  """
  Computes and prints the time-averaged sky-mean effective area and sky coverage for a
  satellite constellation over a full orbital period. The averages are weighted by solid
  angle to account for the non-uniform density of grid points on the sphere.
  :param parfile: str, path to the constellation parameter file
  :param museffdata: object, mu100 and effective area data container
  :param num_val: int, number of angular grid points along each axis, default=100
  :param erg_cut: tuple, energy window (Emin, Emax) [keV], default=(10, 1000)
  """
  sat_info = read_grbpar(parfile)[-1]
  n_sat = len(sat_info)
  # museffdata = MuSeffContainer(mu100par, erg_cut, armcut)

  phi_world = np.linspace(0, 360, num_val, endpoint=False)
  # theta will be converted in sat coord with grb_decra_worldf2satf, which takes dec in world coord with 0 being north pole and 180 the south pole !
  theta_world = np.linspace(0, 180, num_val)

  burst_times = np.linspace(0, orbital_period_calc(sat_info[0][3]), 100, endpoint=False)
  mean_coverage = []
  mean_seff_compton = []
  mean_seff_single = []
  nite = num_val ** 2 * len(sat_info) * 100
  ncount = 0
  for burst_time in burst_times:
    detection = np.zeros((n_sat, num_val, num_val))
    detection_compton = np.zeros((n_sat, num_val, num_val))
    detection_single = np.zeros((n_sat, num_val, num_val))

    for ite, info_sat in enumerate(sat_info):
      for ite_theta, theta in enumerate(theta_world):
        for ite_phi, phi in enumerate(phi_world):
          ncount += 1
          detection_compton[ite][ite_theta][ite_phi], detection_single[ite][ite_theta][ite_phi], detection[ite][ite_theta][ite_phi] = eff_area_func(theta, phi, info_sat, museffdata, burst_time=burst_time)
          print(f"Calculation : {int(ncount / nite * 100)}%", end="\r")
    print("Calculation over")

    detec_sum = np.sum(detection, axis=0)
    detec_sum_compton = np.sum(detection_compton, axis=0)
    detec_sum_single = np.sum(detection_single, axis=0)

    correction_values = (1 + np.sin(np.deg2rad(theta_world)) * (num_val - 1)) / num_val

    mean_coverage.append(np.average(np.mean(detec_sum, axis=1), weights=correction_values))
    mean_seff_compton.append(np.average(np.mean(detec_sum_compton, axis=1), weights=correction_values))
    mean_seff_single.append(np.average(np.mean(detec_sum_single, axis=1), weights=correction_values))

  print(f"mean detection sensitivity values for {parfile} and ergcut {erg_cut[0]}-{erg_cut[1]}")
  print(f"The mean number of satellites in sight is :       {np.mean(mean_coverage):.4f} satellites")
  print(f"The mean effective area for Compton events is :  {np.mean(mean_seff_compton):.4f} cm²")
  print(f"The mean effective area for single events is :   {np.mean(mean_seff_single):.4f} cm²")

