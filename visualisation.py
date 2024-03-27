# Autor Nathan Franel
# Date 06/12/2023
# Version 2 :
# Module to have functions to make so usual actions not using GRB simulated data

# Package imports
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from apexpy import Apex
# Developped modules imports
from funcmod import *
from catalog import Catalog
from MLogData import LogData


# TODO mix this file and trajectories.py ? Change the name of this file. Move some functions from trajectories to funcmod

############################################################
# Usefull functions :
############################################################
def bkg_data_map(field, bkgdata, altitude, dec_range=np.linspace(0, 180, 181), ra_range=np.linspace(0, 360, 361)):
  """
  TODO testing for the detectors !!!
  :param field: Field ploted on the map :
      compton_cr
      single_cr
      calor
      dsssd
      side
      total_hits
  :param bkgdata: background data obtained with MBkgContainer
  :param altitude: altitude for the background
  :param dec_range: range of declinations for the map
  :param ra_range: range of right ascensions for the map
  """
  x_long, y_lat = np.meshgrid(ra_range, 90 - dec_range)
  field_list = np.zeros((len(dec_range), len(ra_range)))
  apex15 = Apex(date=2025)
  if field == "compton_cr":
    item_legend = "compton events count rate (counts/s)"
    field_index = 0
  elif field == "single_cr":
    item_legend = "single events count rate (counts/s)"
    field_index = 1
  elif field == "calor":
    item_legend = "bottom calorimeter count number (counts)"
    field_index = 2
  elif field == "dsssd":
    item_legend = "DSSSD count number (counts)"
    field_index = 3
  elif field == "side":
    item_legend = "side detector count number (counts)"
    field_index = 4
  elif field == "total_hits":
    item_legend = "total count number (counts)"
    field_index = 5
  else:
    raise ValueError("Wrong name given for the background field")

  for row, dec in enumerate(dec_range):
    for col, ra in enumerate(ra_range):
      lat = 90 - dec
      # Geodetic to apex, scalar input
      mag_lat, mag_lon = apex15.convert(lat, ra, 'geo', 'apex', height=altitude)
      # print(f"init : {lat:.12f}, {ra:.12f}              final : {mag_lat:.12f}, {mag_lon:.12f}")
      mag_dec, mag_ra = 90 - mag_lat, mag_lon
      bkg_values = closest_bkg_values(mag_dec, mag_ra, altitude, bkgdata)
      field_list[row][col] = bkg_values[field_index]

  fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
  p1 = ax.pcolormesh(x_long, y_lat, field_list, cmap="Blues")
  ax.coastlines()
  ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title=f"Background map for {item_legend} at {altitude}km")
  cbar = fig.colorbar(p1)
  cbar.set_label(f"Background {item_legend}", rotation=270, labelpad=20)
  plt.show()


def magnetic_latitude_convert(altitude, lat_range=np.linspace(90, -90, 721), lon_range=np.linspace(0, 360, 361)):
  """
  TODO testing !!!
  :param altitude: altitude for the background
  :param lat_range: range of latitudes for the map
  :param lon_range: range of longitudes for the map
  """
  apex15 = Apex(date=2025)
  # WITH SCATTER
  # geo_lat, geo_lon = np.zeros((len(lat_range), len(lon_range))), np.zeros((len(lat_range), len(lon_range)))
  # for ite_lat, mag_lat in enumerate(lat_range):
  #   for ite_lon, mag_lon in enumerate(lon_range):
  #     geo_lat[ite_lat, ite_lon], geo_lon[ite_lat, ite_lon] = apex15.convert(mag_lat, mag_lon, 'apex', 'geo', height=altitude)
  #
  # fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
  # for ite_lat in range(len(lat_range)):
  #   ax.scatter(geo_lon[ite_lat], geo_lat[ite_lat], color='navy', s=1)
  # ax.coastlines()
  # ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title=f"Lines of constant geomagnetic latitudes")
  # plt.show()
  # WITH CONTOUR
  x_lat, y_lat = np.meshgrid(lon_range, lat_range)
  mag_lat = np.zeros((len(lat_range), len(lon_range)))
  for ite_lat, lat in enumerate(lat_range):
    for ite_lon, lon in enumerate(lon_range):
      mag_lat[ite_lat, ite_lon] = apex15.convert(lat, lon, 'geo', 'apex', height=altitude)[0]
  fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
  levels = np.linspace(-90, 90, 19)
  p1 = ax.contour(x_lat, y_lat, mag_lat, levels=levels)
  cbar = fig.colorbar(p1)
  cbar.set_label(f"Geomagnetic latitudes (deg)", rotation=270, labelpad=20)
  cbar.set_ticks(levels)
  ax.coastlines()
  ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title=f"Lines of constant geomagnetic latitudes")
  plt.show()


def calc_duty(inc, ohm, omega, alt, show=False):
  """
  Calculates the duty cycle caused by the radiation belts
  :param inc: inclination of the orbit [deg]
  :param ohm: longitude/ra of the ascending node of the orbit [deg]
  :param omega: argument of periapsis of the orbit [deg]
  :param alt: altitude of the orbit
  :param show: If True shows the trajectory of a satellite on this orbit and the exclusion zones
  """
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
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    ax.set_global()

    dec_points = np.linspace(0, 180, 181)
    ra_points = np.linspace(0, 360, 360, endpoint=False)
    plottitle = f"All radiation belt {alt}km, inc : {inc}"
    cancel_dec = []
    cancel_ra = []
    for dec_p in dec_points:
      for ra_p in ra_points:
        if verif_rad_belts(dec_p, ra_p, alt):
          cancel_dec.append(dec_p)
          cancel_ra.append(ra_p)
    cancel_dec = 90 - np.array(cancel_dec)
    cancel_ra = np.array(cancel_ra)
    ax.scatter(cancel_ra, cancel_dec, s=1, color="red")
    ax.set(title=plottitle)
    # Adding the coasts
    ax.coastlines()
    print(cancel_dec)
    print(cancel_ra)

    colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
    # Creating the lists of sat coordinates
    ax.scatter(long_list, lat_list, s=1, color=colors[2])
    ax.scatter(long_list_rej, lat_list_rej, s=1, color="black")

    plt.show()
  return counter / len(time_vals)


def fov_const(num_val=500, show=True, save=False):
  """
  Plots a map of the sensibility over the sky for number of sat in sight, single events and compton events
  :param num_val: number of value to
  """
  phi_world = np.linspace(0, 360, num_val, endpoint=False)
  # theta will be converted in sat coord with grb_decra_worldf2satf, which takes dec in world coord with 0 being north pole and 180 the south pole !
  theta_world = np.linspace(0, 180, num_val)
  detection = np.zeros((self.n_sat, num_val, num_val))
  detection_compton = np.zeros((self.n_sat, num_val, num_val))
  detection_single = np.zeros((self.n_sat, num_val, num_val))

  # for ite in range(self.n_sat):
  #   detection_pola[ite] = np.array([[eff_area_compton_func(grb_decra_worldf2satf(theta, phi, self.sat_info[ite][0], self.sat_info[ite][1])[0], self.sat_info[ite][2], func_type="cos") for phi in phi_world] for theta in theta_world])
  #   detection_spectro[ite] = np.array([[eff_area_single_func(grb_decra_worldf2satf(theta, phi, self.sat_info[ite][0], self.sat_info[ite][1])[0], self.sat_info[ite][2], func_type="data") for phi in phi_world] for theta in theta_world])

  for ite, info_sat in enumerate(self.sat_info):
    for ite_theta, theta in enumerate(theta_world):
      for ite_phi, phi in enumerate(phi_world):
        detection_compton[ite][ite_theta][ite_phi], detection_single[ite][ite_theta][ite_phi], detection[ite][ite_theta][ite_phi] = eff_area_func(theta, phi, info_sat, self.muSeffdata)
  detec_sum = np.sum(detection, axis=0)
  detec_sum_compton = np.sum(detection_compton, axis=0)
  detec_sum_single = np.sum(detection_single, axis=0)

  phi_plot, theta_plot = np.meshgrid(phi_world, theta_world)
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
  # Map for number of satellite in sight
  ##################################################################################################################
  levels = range(detec_min, detec_max + 1, max(1, int(detec_max + 1 - detec_min) / 15))

  # fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
  fig1, ax1 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
  ax1.set_global()
  ax1.coastlines()
  h1 = ax1.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum, cmap=cmap_det)
  ax1.axis('scaled')
  ax1.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
  cbar = fig1.colorbar(h1, ticks=levels)
  cbar.set_label("Number of satellite in sight", rotation=270, labelpad=20)
  if save:
    fig1.savefig(f"{self.result_prefix}_n_sight")
  if show:
    plt.show()

  # plt.subplot(projection="mollweide")
  # h2 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum, cmap=cmap_det)
  # plt.axis('scaled')
  # plt.xlabel("Right ascention (rad)")
  # plt.ylabel("Declination (rad)")
  # cbar = plt.colorbar(ticks=levels)
  # cbar.set_label("Number of satellite in sight", rotation=270, labelpad=20)
  # if save:
  #   plt.savefig(f"{self.result_prefix}_n_sight_proj")
  # if show:
  #   plt.show()

  ##################################################################################################################
  # Map of constellation's compton effective area
  ##################################################################################################################
  levels_compton = range(detec_min_compton, detec_max_compton + 1, max(1, int(detec_max_compton + 1 - detec_min_compton) / 15))

  # fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
  fig2, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
  ax2.set_global()
  ax2.coastlines()
  h3 = ax2.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_compton, cmap=cmap_compton)
  ax2.axis('scaled')
  ax2.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
  cbar = fig2.colorbar(h3, ticks=levels_compton)
  cbar.set_label("Effective area at for compton events (cm²)", rotation=270, labelpad=20)
  if save:
    fig2.savefig(f"{self.result_prefix}_compton_seff")
  if show:
    plt.show()

  # plt.subplot(projection="mollweide")
  # h4 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_compton, cmap=cmap_compton)
  # plt.axis('scaled')
  # plt.xlabel("Right ascention (rad)")
  # plt.ylabel("Declination (rad)")
  # cbar = plt.colorbar(ticks=levels_compton)
  # cbar.set_label("Effective area at for compton events (cm²)", rotation=270, labelpad=20)
  # if save:
  #   plt.savefig(f"{self.result_prefix}_compton_seff_proj")
  # if show:
  #   plt.show()

  ##################################################################################################################
  # Map of constellation's compton effective area
  ##################################################################################################################
  levels_single = range(detec_min_single, detec_max_single + 1, max(1, int(detec_max_single + 1 - detec_min_single) / 15))

  # fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
  fig3, ax3 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
  ax3.set_global()
  ax3.coastlines()
  h5 = ax3.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_single, cmap=cmap_single)
  ax3.axis('scaled')
  ax3.set(xlabel="Right ascention (rad)", ylabel="Declination (rad)")
  cbar = fig3.colorbar(h5, ticks=levels_single)
  cbar.set_label("Effective area for single events (cm²)", rotation=270, labelpad=20)
  if save:
    fig3.savefig(f"{self.result_prefix}_single_seff")
  if show:
    plt.show()

  # plt.subplot(projection="mollweide")
  # h6 = plt.pcolormesh(phi_plot, np.pi / 2 - theta_plot, detec_sum_single, cmap=cmap_single)
  # plt.axis('scaled')
  # plt.xlabel("Right ascention (rad)")
  # plt.ylabel("Declination (rad)")
  # cbar = plt.colorbar(ticks=levels_single)
  # cbar.set_label("Effective area for single events (cm²)", rotation=270, labelpad=20)
  # if save:
  #   plt.savefig(f"{self.result_prefix}_single_seff")
  # if show:
  #   plt.show()

  print(f"The mean number of satellite in sight is :       {np.mean(detec_sum):.4f} satellites")
  print(f"The mean effective area for compton events is :  {np.mean(detec_sum_compton):.4f} cm²")
  print(f"The mean effective area for single events is :   {np.mean(detec_sum_single):.4f} cm²")


def pers_plot(xdata, ydata, title, xlabel, ylabel, figsize=(10, 6), xscale="linear", yscale="linear", xticks=None, yticks=None, xlim=None, ylim=None, projection=None, legend=True, grid=True, save=False):
  """

  """
  pass


def pers_scatter(xdata, ydata, title, xlabel, ylabel, figsize=(10, 6), s=2, xscale="linear", yscale="linear", xticks=None, yticks=None, xlim=None, ylim=None, projection=None, legend=True, grid=True, save=False):
  """

  """
  pass


def pers_hist():
  """

  """
  pass


def pers_duohist():
  """

  """
  pass


def coverage_maps():
  """

  """
  pass