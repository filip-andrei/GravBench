import math
import numpy as np
from matplotlib import pyplot

G = 6.67408e-11  # m3 / (kg s^2)
central_mass = 5.972e24  # kg
central_radius = 6371e3  # m


def get_orbit_energy(pos, vel):
    energies = []
    for i in range(len(pos)):
        v = math.sqrt(vel[i][0] ** 2 + vel[i][1] ** 2)
        r = math.sqrt(pos[i][0] ** 2 + pos[i][1] ** 2)
        energies.append((v**2)/2 - (G*central_mass/r))

    return energies


def get_acc(pos):
    dist = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
    direction = -(pos / dist)

    return ((G * central_mass) / (dist ** 2)) * direction


def propag_forward_euler(pos, vel, dt):
    acc = get_acc(pos)
    new_vel = vel + acc * dt
    new_pos = pos + vel * dt

    return new_pos, new_vel


def propag_backward_euler(pos, vel, dt):
    new_pos = pos + vel * dt
    new_vel = vel + get_acc(new_pos) * dt

    return new_pos, new_vel


def propag_rk2(pos, vel, dt):
    pos_e, vel_e = propag_forward_euler(pos, vel, dt)
    new_pos = pos + dt * (vel + vel_e)/2
    new_vel = vel + dt * (get_acc(pos) + get_acc(pos_e))/2

    return new_pos, new_vel


def propag_rk4(pos, vel, dt):
    k1v = get_acc(pos)
    k1r = vel
    k2v = get_acc(pos + k1r * dt/2)
    k2r = vel + k1v * dt/2
    k3v = get_acc(pos + k2r * dt/2)
    k3r = vel + k2v * dt/2
    k4v = get_acc(pos + k3r * dt)
    k4r = vel + k3v * dt

    new_vel = vel + dt/6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    new_pos = pos + dt/6 * (k1r + 2 * k2r + 2 * k3r + k4r)
    return new_pos, new_vel


def propagate(pos_0, vel_0, propag_func, dt, max_steps):
    positions = [pos_0]
    velocities = [vel_0]

    current_pos = pos_0
    current_vel = vel_0

    for i in range(max_steps):
        current_pos, current_vel = propag_func(current_pos, current_vel, dt)
        positions.append(current_pos)
        velocities.append(current_vel)

    return positions, velocities


pos_0 = np.array([0, 42e6])
vel_0 = np.array([3.1e3, 0])

step_size = 3600
steps = 24

positions, velocities = propagate(pos_0, vel_0, propag_backward_euler, step_size, steps)
energies = get_orbit_energy(positions, velocities)
x = [pos[0] for pos in positions]
y = [pos[1] for pos in positions]
pyplot.plot(x, y, linestyle="-", color="blue")
#pyplot.plot(energies, linestyle="-", color="blue")

positions, velocities = propagate(pos_0, vel_0, propag_rk2, step_size, steps)
energies = get_orbit_energy(positions, velocities)
x = [pos[0] for pos in positions]
y = [pos[1] for pos in positions]
pyplot.plot(x, y, linestyle="--", color="red")
#pyplot.plot(energies, linestyle="--", color="red")

positions, velocities = propagate(pos_0, vel_0, propag_rk4, step_size, steps)
energies = get_orbit_energy(positions, velocities)
x = [pos[0] for pos in positions]
y = [pos[1] for pos in positions]
pyplot.plot(x, y, linestyle="-.", color="black")
#pyplot.plot(energies, linestyle="-.", color="pink")

pyplot.show()
