from MAllSourceData import AllSourceData
from time import time
import matplotlib.pyplot as plt
import numpy as np

init_time = time()
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--400km--0-0-0--27sat/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--400km--0-45-97--27sat/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--400km--5-5-45--27sat/polGBM.par"
grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-45-97.5--27sat/polGBM.par"
# grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--5-5-45--27sat/polGBM.par"
bkg_param = "./bkg/bkg-v15.par"
mu_param = "./mu100/mu100-v15.par"
erg = (30, 1000)
arm = 180
test = AllSourceData(grb_sim_param, bkg_param, mu_param, erg, arm, parallel="all")
test.make_const()
test.verif_const()
test.analyze()
test.verif_const()
print("=======================================")
print("processing time : ", time()-init_time, "seconds")
print("=======================================")
test.mdp_histogram()
test.count_triggers()

bkg_cr_list400 = []
bkg_compton_cr_list400 = []
dec_list400 = []

bkg_cr_list500 = []
bkg_compton_cr_list500 = []
dec_list500 = []
for bkg in test.bkgdata:
   if bkg.alt == 400:
      bkg_cr_list400.append(bkg.compton_cr)
      bkg_compton_cr_list400.append(bkg.single_cr)
      dec_list400.append(bkg.dec)
   elif bkg.alt == 500:
      bkg_cr_list500.append(bkg.compton_cr)
      bkg_compton_cr_list500.append(bkg.single_cr)
      dec_list500.append(bkg.dec)

bkg_cr_list400 = np.array(bkg_cr_list400)
bkg_compton_cr_list400 = np.array(bkg_compton_cr_list400)
lat_list400 = 90 - np.array(dec_list400)

bkg_cr_list500 = np.array(bkg_cr_list500)
bkg_compton_cr_list500 = np.array(bkg_compton_cr_list500)
lat_list500 = 90 - np.array(dec_list500)

figure, axs = plt.subplots(1, 2, figsize=(16, 12))
figure.suptitle("Background count rates as function of latitude")
axs[0].plot(lat_list400, bkg_cr_list400, color="blue", label="Single event background count rate at 400km")
axs[1].plot(lat_list400, bkg_compton_cr_list400, color="blue", label="Compton event background count rate at 400km")
axs[0].plot(lat_list500, bkg_cr_list500, color="lightsteelblue", label="Single event background count rate at 500km")
axs[1].plot(lat_list500, bkg_compton_cr_list500, color="lightsteelblue", label="Compton event background count rate at 500km")
for ax in axs:
   ax.legend()
   ax.set(xlabel="Latitude(°)", ylabel="Count rate")
plt.show()




from MAllSourceData import AllSourceData
from MBkgContainer import BkgContainer
from MmuSeffContainer import MuSeffContainer
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
bkg_cr_list500 = []
bkg_compton_cr_list500 = []
dec_list500 = []
for bkg in bkgdata:
  if bkg.alt == 500:
    bkg_cr_list500.append(bkg.single_cr)
    bkg_compton_cr_list500.append(bkg.compton_cr)
    dec_list500.append(bkg.dec)
bkg_cr_list500 = np.array(bkg_cr_list500)
bkg_compton_cr_list500 = np.array(bkg_compton_cr_list500)
lat_list500 = 90 - np.array(dec_list500)

phi_world = np.linspace(0, 360, 181)

x_long, y_lat = np.meshgrid(phi_world, lat_list500)
bkg_cr_list = np.ones((len(lat_list500), len(phi_world)))*np.reshape(bkg_cr_list500, (len(lat_list500), 1))

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
p1 = ax.pcolormesh(x_long, y_lat, bkg_cr_list, cmap="Blues")
ax.coastlines()
#plt.axis('scaled')
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Background count rates map")
cbar = fig.colorbar(p1)
cbar.set_label("count rate for single events (counts/s)", rotation=270, labelpad=20)
plt.show()

phi_sat = np.linspace(0, 360, 181)
theta_sat = np.linspace(0, 114, 115)

mu100list = np.zeros((len(theta_sat), len(phi_sat)))
for i, theta in enumerate(theta_sat):
  for j, phi in enumerate(phi_sat):
    mu100list[i, j] = closest_mufile(theta, phi, muSeffdata)[0]

theta_sat = 90 - theta_sat
x_mu, y_mu = np.meshgrid(phi_sat, theta_sat)

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertConformal(central_longitude=0, central_latitude=0)})
p2 = ax.pcolormesh(x_mu, y_mu, mu100list, cmap="Blues", transform=ccrs.PlateCarree())
#plt.axis('scaled')
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Background count rates map")
ax.gridlines(xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 20))
# ax.set_xticks(np.arange(0, 360, 30), crs=ccrs.PlateCarree())
# ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
cbar = fig.colorbar(p2, ax=ax)
cbar.set_label("count rate for single events (counts/s)", rotation=270, labelpad=20)
plt.show()





grb_sim_param = "/pdisk/ESA/COMCUBEv15--500km--0-0-0--27sat/polGBM.par"
bkg_param = "./bkg/bkg-v15.par"
mu_param = "./mu100/mu100-v15.par"
ergcut = (30, 1000)
armcut = 180
muSeffdata = MuSeffContainer(mu_param, ergcut, armcut)

sat_info = read_grbpar(grb_sim_param)[-1]

num_val = 500
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
    detection[ite] = np.array([[eff_area_func(theta, phi, sat_dec_wf, sat_ra_wf, sat_info[ite][3], muSeffdata, func_type="FoV") for phi in phi_world] for theta in theta_world])
    detection_compton[ite] = np.array([[eff_area_func(theta, phi, sat_dec_wf, sat_ra_wf, sat_info[ite][3], muSeffdata, func_type="compton") for phi in phi_world] for theta in theta_world])
    detection_single[ite] = np.array([[eff_area_func(theta, phi, sat_dec_wf, sat_ra_wf, sat_info[ite][3], muSeffdata, func_type="single") for phi in phi_world] for theta in theta_world])
sat_lat_wf_list = 90 - np.array(sat_dec_wf_list)
sat_ra_wf_list = np.array(sat_ra_wf_list)
phi_plot, theta_plot = np.meshgrid(phi_world, 90 - theta_world)

detec_sum = np.sum(detection, axis=0)
detec_sum_compton = np.sum(detection_compton, axis=0)
detec_sum_single = np.sum(detection_single, axis=0)

detec_min = int(np.min(detec_sum))
detec_max = int(np.max(detec_sum))
cmap = mpl.cm.Blues_r
levels = np.arange(int(detec_min), int(detec_max) + 1, int((int(detec_max) + 1 - int(detec_min)) / 10))

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
ax.set_global()
h1 = ax.pcolormesh(phi_plot, theta_plot, detec_sum, cmap=cmap)
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Constellation coverage map")
cbar = fig.colorbar(h1, ax=ax, ticks=levels)
cbar.set_label("Number of satellite in sight", rotation=270, labelpad=20)
colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
ax.scatter(sat_ra_wf_list, sat_lat_wf_list, color=colors[3])


detec_min_compton = int(np.min(detec_sum_compton))
detec_max_compton = int(np.max(detec_sum_compton))
cmap_compton = mpl.cm.Blues_r
levels_compton = np.arange(int(detec_min_compton), int(detec_max_compton) + 1, int((int(detec_max_compton) + 1 - int(detec_min_compton)) / 10))

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
ax.set_global()
h1 = ax.pcolormesh(phi_plot, theta_plot, detec_sum, cmap=cmap_compton)
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Constellation effective area for compton events map")
cbar = fig.colorbar(h1, ax=ax, ticks=levels_compton)
cbar.set_label("Effective area for compton events (cm²)", rotation=270, labelpad=20)
colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
ax.scatter(sat_ra_wf_list, sat_lat_wf_list, color=colors[3])


detec_min_single = int(np.min(detec_sum_single))
detec_max_single = int(np.max(detec_sum_single))
cmap_single = mpl.cm.Blues_r
levels_single = np.arange(int(detec_min_single), int(detec_max_single) + 1, int((int(detec_max_single) + 1 - int(detec_min_single)) / 10))

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
ax.set_global()
h1 = ax.pcolormesh(phi_plot, theta_plot, detec_sum, cmap=cmap_single)
ax.set(xlabel="Right ascension (deg)", ylabel="Latitude (deg)", title="Constellation effective area for single events map")
cbar = fig.colorbar(h1, ax=ax, ticks=levels_single)
cbar.set_label("Effective area for single events (cm²)", rotation=270, labelpad=20)
colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
ax.scatter(sat_ra_wf_list, sat_lat_wf_list, color=colors[3])


# theta_line = []
# phi_line = []
# Creating the lists of sat coordinates
# for ite_orbit in range(len(inc)):
#   line_temp_theta, line_temp_phi = orbitalparam2decra(inc[ite_orbit], ohm[ite_orbit], np.linspace(0, 360, 100))
#   theta_line.append(line_temp_theta)
#   phi_line.append(line_temp_phi)
# latitude_line = 90 - np.array(theta_line)
# phi_line = np.array(phi_line)

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
ax.coastlines()
plt.show()

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
