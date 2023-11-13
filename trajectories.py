import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def coor(i, om, nu):
    # x = -np.cos(om) * np.sin(nu) - np.sin(om) * np.cos(i) * np.cos(nu)
    # y = -np.sin(om) * np.sin(nu) + np.cos(om) * np.cos(i) * np.cos(nu)
    # z = np.sin(i) * np.cos(nu)
    x = np.cos(nu) * np.cos(om) - np.sin(nu) * np.cos(i) * np.sin(om)
    y = np.cos(nu) * np.sin(om) + np.sin(nu) * np.cos(i) * np.cos(om)
    z = np.sin(nu) * np.sin(i)
    return x, y, z


def trajectory(inc, ohm, nsat, alt):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot([0, 1], [0, 0], [0, 0], c="black")
    ax.plot([0, 0], [0, 1], [0, 0], c="black")
    ax.plot([0, 0], [0, 0], [0, 1], c="black")
    ax.axis('off')
    ax.grid(visible=None)

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

    x1, y1, z1 = coor(np.deg2rad(inc[0]), np.deg2rad(ohm[0]), np.linspace(0, 2 * np.pi, nsat[0], endpoint=False))
    x2, y2, z2 = coor(np.deg2rad(inc[1]), np.deg2rad(ohm[1]), np.linspace(0 + 2 * np.pi / (3 * nsat[1]), 2 * np.pi + 2 * np.pi / (3 * nsat[1]), nsat[1], endpoint=False))
    x3, y3, z3 = coor(np.deg2rad(inc[2]), np.deg2rad(ohm[2]), np.linspace(0 + (2 * np.pi / (3 * nsat[1])) * 2, 2 * np.pi + (2 * np.pi / (3 * nsat[1])) * 2, nsat[2], endpoint=False))

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
    # set_axes_equal(ax)
    fig.show()

mpl.rcParams.update({'font.size': 22})

trajectory([0, 0, 0], [0, 0, 0], [9, 9, 9], 400)
trajectory([5, 5, 45], [0, 180, 0], [9, 9, 9], 400)
trajectory([0, 45, 83], [0, 0, 0], [9, 9, 9], 400)

trajectory([0, 0, 0], [0, 0, 0], [9, 9, 9], 500)
trajectory([5, 5, 45], [0, 180, 0], [9, 9, 9], 500)
trajectory([0, 45, 82.5], [0, 0, 0], [9, 9, 9], 500)
