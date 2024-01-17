# import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
from funcmod import *


def orbitalparam2cartesian(i, ohm, omega):
  """
  Calculates the cartesian coordinates of a point defined my orbital parameters (at a unit distance from the center)
  :param i: inclination of the orbit [deg]
  :param ohm: longitude/ra of the ascending node of the orbit [deg]
  :param omega: argument of periapsis of the orbit [deg]
  """
  x = np.cos(omega) * np.cos(ohm) - np.sin(omega) * np.cos(i) * np.sin(ohm)
  y = np.cos(omega) * np.sin(ohm) + np.sin(omega) * np.cos(i) * np.cos(ohm)
  z = np.sin(omega) * np.sin(i)
  return x, y, z


def orbit(inc, ohm, omega, nsat, alt):
  """
  Plots a 3D repartition of the satellites
  :param inc: list of inclination of the orbits [deg]
  :param ohm: list of longitude/ra of the ascending node of the orbits [deg]
  :param omega: list of argument of periapsis of the orbits [deg]
  :param nsat: list of number of satellite per orbit
  :param alt: float, altitude of the orbits
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

# mpl.rcParams.update({'font.size': 22})
# omega = 0
# orbit([0, 0, 0], [0, 0, 0], omega, [9, 9, 9], 400)
# orbit([5, 5, 45], [0, 180, 0], omega, [9, 9, 9], 400)
# orbit([0, 45, 83], [0, 0, 0], omega, [9, 9, 9], 400)


def trajectory(inc, ohm, nsat, alt, excludefile=None, omega=0, projection="carre"):
  """
  TODO add the earth rotation to the trajectory of the satellites
  Plots the trajectory of different orbits, its satellites and the exclusion zones due to radiation belts
  :param inc: list of inclination of the orbits [deg]
  :param ohm: list of longitude/ra of the ascending node of the orbits [deg]
  :param nsat: list of number of satellite per orbit
  :param alt: float, altitude of the orbits
  :param excludefile: name of the file if the exclusion zone needed is a specific one, if None the it is the whole radiation belt
  :param omega: argument of periapsis of the orbits [deg]
  :param projection: type of projection to use "carre" or "mollweide" (this one doesn't work well)
  """
  # Create the plot with the correct projection
  if projection == "mollweide":
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Mollweide(central_longitude=0)})
  elif projection == "carre":
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
  else:
    print("Please use either carre or mollweide for projection")
    return None
  ax.set_global()

  theta_list = []
  phi_list = []
  theta_line = []
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
      temp_theta.append(theta)
      temp_phi.append(phi)
    line_temp_theta, line_temp_phi = orbitalparam2decra(inc[ite_orbit], ohm[ite_orbit], np.linspace(0, 360, 100))
    theta_line.append(line_temp_theta)
    phi_line.append(line_temp_phi)
    theta_list.append(temp_theta)
    phi_list.append(temp_phi)
  latitude_list = 90 - np.array(theta_list)
  phi_list = np.array(phi_list)
  latitude_line = 90 - np.array(theta_line)
  phi_line = np.array(phi_line)
  for ite_orbit in range(len(inc)):
    ax.scatter(phi_list[ite_orbit], latitude_list[ite_orbit], color=colors[ite_orbit])
    ax.scatter(phi_line[ite_orbit], latitude_line[ite_orbit], s=1, color=colors[ite_orbit])

  theta_verif = np.linspace(-90, 90, 181)
  phi_verif = np.linspace(-180, 180, 360, endpoint=False)
  if excludefile is not None:
    if excludefile == "all":
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
    else:
      plottitle = excludefile.split("/")[-1]
      cancel_theta = []
      cancel_phi = []
      for theta in theta_verif:
        for phi in phi_verif:
          if verif_zone_file(theta, phi, excludefile):
            cancel_theta.append(theta)
            cancel_phi.append(phi)
      cancel_theta = np.array(cancel_theta)
      cancel_phi = np.array(cancel_phi)
    if projection == "carre":
      ax.scatter(cancel_phi, cancel_theta, s=1)
      ax.set(title=plottitle)
  # Adding the coasts
  ax.coastlines()
  plt.show()

# files = ["./bkg/exclusion/400km/AE8max_400km.out", "./bkg/exclusion/400km/AP8min_400km.out",
#          "./bkg/exclusion/500km/AE8max_500km.out", "./bkg/exclusion/500km/AP8min_500km.out"]

# for file in files:
#     trajectory([5, 5, 45], [0, 180, 90], [12, 12, 12], 400, excludefile=file, projection="carre")
# for alt in [400, 500]:
#     trajectory([5, 5, 45], [0, 180, 90], [9, 9, 9], alt, excludefile="all", projection="carre")

trajectory([0], [0], [27], 500, excludefile="all", projection="carre")

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
  for ite in range(len(time_vals)):
    decsat, rasat = orbitalparam2decra(inc, ohm, omega, nu=true_anomalies[ite])
    rasat -= earth_ra_offset[ite]
    lat_list.append(90 - decsat)
    long_list.append(rasat)
    if not verif_rad_belts(90 - decsat, rasat, alt):
      counter += 1
  print(f"Duty cycle for inc : {inc}, ohm : {ohm}, omega : {omega} for all files at {alt}")
  print(f"   === {counter / len(time_vals)}")

  if show:
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    ax.set_global()

    colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
    # Creating the lists of sat coordinates
    ax.scatter(long_list, lat_list, s=1, color=colors[2])

    theta_verif = np.linspace(-90, 90, 181)
    phi_verif = np.linspace(-180, 180, 360, endpoint=False)
    plottitle = f"All radiation belt {alt}km, inc : {inc}"
    cancel_theta = []
    cancel_phi = []
    for theta in theta_verif:
      for phi in phi_verif:
        # if verif_zone_file(lat, long, file):
        if verif_rad_belts(theta, phi, alt):
          cancel_theta.append(theta)
          cancel_phi.append(phi)
    cancel_theta = np.array(cancel_theta)
    cancel_phi = np.array(cancel_phi)
    ax.scatter(cancel_phi, cancel_theta, s=1, color="red")
    ax.set(title=plottitle)
    # Adding the coasts
    ax.coastlines()
    plt.show()
  return counter / len(time_vals)


def calc_partial_duty(inc, ohm, omega, alt, exclusionfile):
  """
  Calculates the duty cycle caused by an exclusion zone
  :param inc: inclination of the orbit [deg]
  :param ohm: longitude/ra of the ascending node of the orbit [deg]
  :param omega: argument of periapsis of the orbit [deg]
  :param alt: altitude of the orbit
  :param exclusionfile: file describing the exclusion zone
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
  for ite in range(len(time_vals)):
    decsat, rasat = orbitalparam2decra(inc, ohm, omega, nu=true_anomalies[ite])
    rasat -= earth_ra_offset[ite]
    lat_list.append(90 - decsat)
    long_list.append(rasat)
    if not verif_zone_file(90 - decsat, rasat, exclusionfile):
      counter += 1
  print(f"Duty cycle for inc : {inc}, ohm : {ohm}, omega : {omega} for {exclusionfile.split('/')[-1]}  at {alt}")
  print(f"   === {counter / len(time_vals)}")

  fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
  ax.set_global()

  colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
  # Creating the lists of sat coordinates
  ax.scatter(long_list, lat_list, s=1, color=colors[2])

  theta_verif = np.linspace(-90, 90, 181)
  phi_verif = np.linspace(-180, 180, 360, endpoint=False)
  plottitle = f"{exclusionfile.split('/')[-1]} {alt}km, inc : {inc}"
  cancel_theta = []
  cancel_phi = []
  for theta in theta_verif:
    for phi in phi_verif:
      if verif_zone_file(theta, phi, exclusionfile):
        cancel_theta.append(theta)
        cancel_phi.append(phi)
  cancel_theta = np.array(cancel_theta)
  cancel_phi = np.array(cancel_phi)
  ax.scatter(cancel_phi, cancel_theta, s=1, color="red")
  ax.set(title=plottitle)
  # Adding the coasts
  ax.coastlines()
  plt.show()


# files = ["./bkg/exclusion/400km/AE8max_400km.out", "./bkg/exclusion/400km/AP8min_400km.out",
#          "./bkg/exclusion/500km/AE8max_500km.out", "./bkg/exclusion/500km/AP8min_500km.out"]
# # for file in files:
# #   calc_partial_duty(90, 0, 0, 400, file)
# # calc_duty(90, 0, 0, 400)
#
# incl = np.linspace(0, 90, 46)
# duty_list = []
# for incli in incl:
#   duty_list.append(calc_duty(incli, 0, 0, 500))
# plt.figure()
# plt.plot(incl, duty_list)
# plt.grid()
# plt.xticks(np.arange(0, 91, 5))
# plt.yticks(np.linspace(0.8, 1, 21))
# plt.show()

# calc_duty(0, 0, 0, 400)
# calc_duty(0, 0, 0, 500)
# calc_duty(5, 0, 0, 400)
# calc_duty(5, 0, 0, 500)
# calc_duty(45, 0, 0, 400)
# calc_duty(45, 0, 0, 500)
# calc_duty(82.5, 0, 0, 400)
# calc_duty(82.5, 0, 0, 500)
# calc_duty(83, 0, 0, 400)
# calc_duty(83, 0, 0, 500)
# calc_duty(90, 0, 0, 500)
