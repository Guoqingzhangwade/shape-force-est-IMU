#!/usr/bin/env python3
"""
Continuum–robot illustration

Changes compared with the previous draft
────────────────────────────────────────
✓ Disks use light Lambert shading  → clear 3-D depth cue  
✓ Backbone drawn as separate segments from one disk hub to the next,
  stopped exactly at each face                       → no mis-alignment
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import pi

# ───────────────────────────── geometry ──────────────────────────────
TOTAL_ARC_LEN = 70.0            # mm (scales dimensionless s → mm)

def skew(u):
    return np.array([[0, -u[2],  u[1]],
                     [u[2],   0, -u[0]],
                     [-u[1], u[0],  0]])

def twist(u, e3):
    return np.block([[skew(u), e3.reshape(3, 1)],
                     [np.zeros((1, 3)), 0]])

def product_of_exponentials(m, s, phi, k=10):
    """SE(3) product-of-exponentials, uniform subdivision."""
    Δ = s / k
    T = np.eye(4)
    e3 = np.array([0, 0, 1])
    for i in range(k):
        T = T @ expm(twist(phi(i*Δ) @ m, e3) * Δ)
    return T

def fk(m, s, phi, k=10):
    T = product_of_exponentials(m, s, phi, k)
    T[:3, 3] *= TOTAL_ARC_LEN
    return T

def phi_quad(s):
    return np.array([[1, s, s**2, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, s, s**2, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, s, s**2]])

def frames_along(m, s_vals, phi, k=10):
    pos, rot = [], []
    for s in s_vals:
        T = fk(m, s, phi, k)
        pos.append(T[:3, 3])
        rot.append(T[:3, :3])
    return np.asarray(pos), rot

# ───────────────────── disk with simple Lambert shading ──────────────
def lambert_shade(normal, view_dir, base_rgb, k_ambient=0.3):
    """Return RGB after cosine shading."""
    cosθ = np.clip(np.dot(normal, view_dir), 0, 1)
    shade = k_ambient + (1-k_ambient) * cosθ
    return np.clip(base_rgb * shade, 0, 1)

def make_disk(origin, R, radius, thickness, n=48):
    θ = np.linspace(0, 2*pi, n, endpoint=False)
    circle = np.vstack((radius*np.cos(θ),
                        radius*np.sin(θ),
                        np.zeros_like(θ)))
    top    = R @ (circle + np.array([[0],[0],[+thickness/2]])) + origin[:,None]
    bottom = R @ (circle + np.array([[0],[0],[-thickness/2]])) + origin[:,None]

    top_poly    = list(top.T)
    bottom_poly = list(bottom.T[::-1])
    sides = []
    for i in range(n):
        j = (i+1) % n
        quad = [bottom[:,i], bottom[:,j], top[:,j], top[:,i]]
        sides.append(quad)
    return top_poly, bottom_poly, sides

def add_disk(ax, origin, R, *, radius=5, thickness=1,
             base_colour=(0.75, 0.75, 0.75), edge_colour='black', lw=0.7):
    view_dir = np.array([*ax.get_proj()[:,2]])[:3]    # viewer direction in data coords
    view_dir /= np.linalg.norm(view_dir)

    top, bottom, sides = make_disk(origin, R, radius, thickness)

    # face normals (all the same for top/bottom)
    n_top =  R[:,2]
    n_bot = -R[:,2]

    shade_top = lambert_shade(n_top, view_dir, np.array(base_colour))
    shade_bot = lambert_shade(n_bot, view_dir, np.array(base_colour))

    faces = [(top,    shade_top),
             (bottom, shade_bot)] + [(s, shade_top*0.9) for s in sides]

    for verts, col in faces:
        ax.add_collection3d(
            Poly3DCollection([verts],
                             facecolors=[col],
                             edgecolors=edge_colour,
                             linewidths=lw)
        )

# ───────────────────── other drawing helpers ────────────────────────
def draw_arrow(ax, o, v, c):
    ax.plot([o[0], o[0]+v[0]],
            [o[1], o[1]+v[1]],
            [o[2], o[2]+v[2]], c, linewidth=2)

def draw_frame(ax, o, R, L=6):
    draw_arrow(ax, o, R[:,0]*L, 'r')
    draw_arrow(ax, o, R[:,1]*L, 'g')
    draw_arrow(ax, o, R[:,2]*L, 'b')

def set_axes_equal(ax):
    lims = np.array([ax.get_xlim3d(),
                     ax.get_ylim3d(),
                     ax.get_zlim3d()])
    centre  = lims.mean(axis=1)
    radius  = (lims[:,1]-lims[:,0]).max() / 2
    ax.set_xlim3d(centre[0]-radius, centre[0]+radius)
    ax.set_ylim3d(centre[1]-radius, centre[1]+radius)
    ax.set_zlim3d(centre[2]-radius, centre[2]+radius)

# ────────────────────────── main figure ─────────────────────────────
def plot_robot(m):
    fig = plt.figure(figsize=(7,9))
    ax  = fig.add_subplot(111, projection='3d')

    # parameters
    disk_r, disk_t   = 5.0, 1.0
    attach_r         = 3.0
    wire_angles      = np.deg2rad([0, 90, 180, 270])

    # discrete backbone samples
    s_disk = np.linspace(0, 1, 9)
    s_fine = np.linspace(0, 1, 400)

    pos_d, R_d = frames_along(m, s_disk,  phi_quad, k=20)
    pos_f, R_f = frames_along(m, s_fine,  phi_quad, k=20)

    # 1) BACKBONE – draw **between** neighbouring disks
    for i in range(len(pos_d)-1):
        p0 = pos_d[i]   + R_d[i][:,2]*(disk_t/2)
        p1 = pos_d[i+1] - R_d[i+1][:,2]*(disk_t/2)
        ax.plot([p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color='dimgray', linewidth=4)

    # 2) WIRES
    for ang in wire_angles:
        pts = []
        for p, R in zip(pos_f, R_f):
            pts.append(R @ np.array([attach_r*np.cos(ang),
                                     attach_r*np.sin(ang), 0]) + p)
        pts = np.asarray(pts)
        ax.plot(pts[:,0], pts[:,1], pts[:,2],
                color='royalblue', linewidth=5)

    # 3) DISKS (opaque, shaded)
    for o, R in zip(pos_d, R_d):
        add_disk(ax, o, R, radius=disk_r, thickness=disk_t)

    # 4) FRAMES
    draw_frame(ax, pos_d[0],  R_d[0])
    draw_frame(ax, pos_d[4],  R_d[4])
    draw_frame(ax, pos_d[-1], R_d[-1])
    draw_frame(ax, np.array([30, -10, 0]), np.eye(3))

    # tidy up
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_visible(False); a._axinfo['grid']['linewidth'] = 0
        a._axinfo['axisline']['linewidth'] = 0
    set_axes_equal(ax)
    ax.view_init(elev=20, azim=110)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# ────────────────────────────── run  ────────────────────────────────
if __name__ == '__main__':
    np.random.seed(35)
    m_true = np.array([-1, 6, -4, 1, 3, -5, 0, 0, 0])
    plot_robot(m_true)















   






