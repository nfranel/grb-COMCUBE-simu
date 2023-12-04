import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from funcmod import *

def coor(i, om, omega):
    # x = -np.cos(om) * np.sin(omega) - np.sin(om) * np.cos(i) * np.cos(omega)
    # y = -np.sin(om) * np.sin(omega) + np.cos(om) * np.cos(i) * np.cos(omega)
    # z = np.sin(i) * np.cos(omega)
    x = np.cos(omega) * np.cos(om) - np.sin(omega) * np.cos(i) * np.sin(om)
    y = np.cos(omega) * np.sin(om) + np.sin(omega) * np.cos(i) * np.cos(om)
    z = np.sin(omega) * np.sin(i)
    return x, y, z


def orbit(inc, ohm, omega, nsat, alt):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # fig, ax = plt.subplot( 111, projection='3d')

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

    x1, y1, z1 = coor(np.deg2rad(inc[0]), np.deg2rad(ohm[0]), omega + np.linspace(0, 2 * np.pi, nsat[0], endpoint=False))
    x2, y2, z2 = coor(np.deg2rad(inc[1]), np.deg2rad(ohm[1]), omega + np.linspace(0 + 2 * np.pi / (3 * nsat[1]), 2 * np.pi + 2 * np.pi / (3 * nsat[1]), nsat[1], endpoint=False))
    x3, y3, z3 = coor(np.deg2rad(inc[2]), np.deg2rad(ohm[2]), omega + np.linspace(0 + (2 * np.pi / (3 * nsat[1])) * 2, 2 * np.pi + (2 * np.pi / (3 * nsat[1])) * 2, nsat[2], endpoint=False))

    xl1, yl1, zl1 = coor(np.deg2rad(inc[0]), np.deg2rad(ohm[0]), np.linspace(0, 2 * np.pi, 100))
    xl2, yl2, zl2 = coor(np.deg2rad(inc[1]), np.deg2rad(ohm[1]), np.linspace(0, 2 * np.pi, 100))
    xl3, yl3, zl3 = coor(np.deg2rad(inc[2]), np.deg2rad(ohm[2]), np.linspace(0, 2 * np.pi, 100))

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
#
# orbit([0, 0, 0], [0, 0, 0], omega, [9, 9, 9], 500)
# orbit([5, 5, 45], [0, 180, 0], omega, [9, 9, 9], 500)
# orbit([0, 45, 82.5], [0, 0, 0], omega, [9, 9, 9], 500)
# omega = np.deg2rad(0)
# orbit([5, 5, 45], [0, 180, 0], omega, [1, 1, 1], 400)
# omega = np.deg2rad(20)
# orbit([5, 5, 45], [0, 180, 0], omega, [1, 1, 1], 400)
# omega = np.deg2rad(40)
# orbit([5, 5, 45], [0, 180, 0], omega, [1, 1, 1], 400)
# omega = np.deg2rad(60)
# orbit([5, 5, 45], [0, 180, 0], omega, [1, 1, 1], 400)
# omega = np.deg2rad(80)
# orbit([5, 5, 45], [0, 180, 0], omega, [1, 1, 1], 400)

def trajectory(inc, ohm, nsat, projection="carre"):
    # Create the plot with the correct projection
    if projection == "mollweide":
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Mollweide(central_longitude=0)})
    elif projection == "carre":
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    else:
        print("Please use either carre or mollweide for projection")
        return None

    theta_list = []
    phi_list = []
    theta_line = []
    phi_line = []
    colors = ["lightsteelblue", "cornflowerblue", "royalblue", "blue", "navy"]
    # Creating the lists of sat coordinates
    for ite_orbit in range(len(inc)):
        temp_theta = []
        temp_phi = []
        starting = 360/nsat[ite_orbit] / len(inc) * ite_orbit # offset in the repartition of sat of different orbits so that there is a constant repartition in omega and that the first sat of each orbit are not at the same place
        for ite_sat in range(nsat[ite_orbit]):
            separation = 360/nsat[ite_orbit] * ite_sat # separation between sats of a same orbit
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
    # print(phi_list, theta_list)
    ax.set_global()
    for ite_orbit in range(len(inc)):
        ax.scatter(phi_list[ite_orbit], theta_list[ite_orbit], color=colors[ite_orbit])
        ax.scatter(phi_line[ite_orbit], theta_line[ite_orbit], s=1, color=colors[ite_orbit])
        print(theta_line[ite_orbit])
        print()

    if projection == "carre":
        theta_verif = np.linspace(-90, 90, 91)
        phi_verif = np.linspace(-180, 180, 181)
        # Pour plot la maille :
        # ax.scatter(phi_verif, theta_verif, s=1)
        # theta_verif = theta_verif.reshape((91, 1)) * np.ones((91, 181))
        # phi_verif = phi_verif * np.ones((91, 181))
        cancel_theta = []
        cancel_phi = []
        for theta in theta_verif:
            for phi in phi_verif:
                if verif_zone(theta, phi):
                    cancel_theta.append(theta)
                    cancel_phi.append(phi)
        cancel_theta = np.array(cancel_theta)
        cancel_phi = np.array(cancel_phi)
        ax.scatter(cancel_phi, cancel_theta, s=1)
    # Adding the coasts
    ax.coastlines()

    plt.show()

# trajectory([0, 0, 0], [0, 0, 0], [9, 9, 9])
# trajectory([5, 5, 45], [0, 180, 0], [9, 9, 9])
# trajectory([0, 45, 82.5], [0, 0, 0], [9, 9, 9])
omega = 0
trajectory([5, 5, 45], [0, 180, 90], [12, 12, 12])


# from matplotlib.animation import FuncAnimation
#
#
# inc_set = [0, 0, 0]
# ohm_set = [0, 0, 0]
# nbsat = [9, 9, 9]
# alt = 400
# omega = np.linspace(0, 5547, 20)
#
# r = 6371 / (6371 + alt)
# x = []
# y = []
# z = []
# n_ite_eq = 100
# for theta in np.linspace(-np.pi / 2, np.pi / 2, int(n_ite_eq / 2)):
#     if theta == -np.pi / 2 or theta == np.pi / 2:
#         phi_list = [0]
#     else:
#         phi_list = np.linspace(0, 2 * np.pi, int(np.cos(theta) * n_ite_eq))
#     for phi in phi_list:
#         x.append(r * np.cos(phi) * np.cos(theta))
#         y.append(r * np.sin(phi) * np.cos(theta))
#         z.append(r * np.sin(theta))
#
# xl1, yl1, zl1 = coor(np.deg2rad(inc_set[0]), np.deg2rad(ohm_set[0]), np.linspace(0, 2 * np.pi, 100))
# xl2, yl2, zl2 = coor(np.deg2rad(inc_set[1]), np.deg2rad(ohm_set[1]), np.linspace(0, 2 * np.pi, 100))
# xl3, yl3, zl3 = coor(np.deg2rad(inc_set[2]), np.deg2rad(ohm_set[2]), np.linspace(0, 2 * np.pi, 100))
#
# # Function to generate each frame
#
# def orbit_frame(frame):
#     # ax.clear()
#     ax = fig.add_subplot(projection='3d')
#     ax.plot([0, 1], [0, 0], [0, 0], c="black")
#     ax.plot([0, 0], [0, 1], [0, 0], c="black")
#     ax.plot([0, 0], [0, 0], [0, 1], c="black")
#     ax.scatter(x, y, z, s=1, c='slategrey', alpha=0.1)
#     ax.axis('off')
#     ax.grid(visible=None)
#     # ax.legend(bbox_to_anchor=(0.1, 1.15), loc='upper left')
#     ax.set(xlabel="x", ylabel="y", zlabel="z")
#     ax.set_box_aspect([1.0, 1.0, 1.0])
#
#     x1, y1, z1 = coor(np.deg2rad(inc_set[0]), np.deg2rad(ohm_set[0]), omega[frame] + np.linspace(0, 2 * np.pi, nbsat[0], endpoint=False))
#     x2, y2, z2 = coor(np.deg2rad(inc_set[1]), np.deg2rad(ohm_set[1]), omega[frame] + np.linspace(0 + 2 * np.pi / (3 * nbsat[1]), 2 * np.pi + 2 * np.pi / (3 * nbsat[1]), nbsat[1], endpoint=False))
#     x3, y3, z3 = coor(np.deg2rad(inc_set[2]), np.deg2rad(ohm_set[2]), omega[frame] + np.linspace(0 + (2 * np.pi / (3 * nbsat[1])) * 2, 2 * np.pi + (2 * np.pi / (3 * nbsat[1])) * 2, nbsat[2], endpoint=False))
#
#     ax.scatter3D(x1, y1, z1, s=40, c="blue")
#     ax.scatter3D(x2, y2, z2, s=40, c='cornflowerblue')
#     ax.scatter3D(x3, y3, z3, s=40, c='darkblue')
#     ax.plot(xl1, yl1, zl1, c="blue", label=f'Inclination : {inc_set[0]}° \nRA of the ascending node : {ohm_set[0]}°')
#     ax.plot(xl2, yl2, zl2, c=f'cornflowerblue', label=f'Inclination : {inc_set[1]}° \nRA of the ascending node : {ohm_set[1]}°')
#     ax.plot(xl3, yl3, zl3, c=f'darkblue', label=f'Inclination : {inc_set[2]}° \nRA of the ascending node : {ohm_set[2]}°')
#
#
# # Initialise figure
# fig = plt.figure()
#
# # Créer l'animation
# animation = FuncAnimation(fig, orbit_frame, frames=len(omega), repeat=False)
# # Afficher l'animation
# plt.show()
