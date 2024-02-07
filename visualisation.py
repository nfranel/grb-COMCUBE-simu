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