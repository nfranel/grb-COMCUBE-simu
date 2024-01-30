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


def magnetic_latitude_convert(altitude, lat_range=np.linspace(90, -90, 19), lon_range=np.linspace(-180, 180, 361)):
  """
  TODO testing !!!
  :param altitude: altitude for the background
  :param lat_range: range of latitudes for the map
  :param lon_range: range of longitudes for the map
  """
  apex15 = Apex(date=2025)
  geo_lat, geo_lon = np.zeros((len(lat_range), len(lon_range))), np.zeros((len(lat_range), len(lon_range)))
  for ite_lat, mag_lat in enumerate(lat_range):
    for ite_lon, mag_lon in enumerate(lon_range):
      geo_lat[ite_lat, ite_lon], geo_lon[ite_lat, ite_lon] = apex15.convert(mag_lat, mag_lon, 'apex', 'geo', height=altitude)

  fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
  for ite_lat in range(len(lat_range)):
    ax.plot(geo_lon[ite_lat], geo_lat[ite_lat], color='black')
  ax.coastlines()
  ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)", title=f"Lines of constant geomagnetic latitudes")
  plt.show()

def coverage_maps():
  """

  """
  pass