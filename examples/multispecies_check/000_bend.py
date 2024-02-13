import xtrack as xt
import numpy as np

from scipy.constants import e as qe
from scipy.constants import c as clight

p_ref1 = xt.Particles(
    mass0=xt.PROTON_MASS_EV,
    q0=2,
    gamma0=1.2)

p_ref2 = xt.Particles(
    mass0=2*xt.PROTON_MASS_EV,
    q0=3,
    gamma0=1.5)

# Build a particle referred to reference 1
p1 = p_ref1.copy()
p1.x = 1e-3
p1.y = 2e-3
p1.zeta = 1e-2
p1.delta = 0.5
P_p1 = (1 + p1.delta) * p1.p0c

# Build the same particle referred to reference 2
p1_ref2 = p_ref2.copy()
p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

p1_ref2.x = p1.x
p1_ref2.y = p1.y
p1_ref2.zeta = p1.zeta / p1.beta0 * p1_ref2.beta0
p1_ref2.charge_ratio = p1_ref2_charge_ratio
p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

assert np.isclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

assert np.isclose(p1.rpp, 1 / (1 + p1.delta), atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.rpp, 1 / (1 + p1_ref2.delta), atol=0, rtol=1e-14)



L_bend = 1.
B_T = 0.2

P0_J_ref1 = p_ref1.p0c[0] / clight * qe
h_bend_ref1 = B_T * qe * p_ref1.charge / P0_J_ref1 # This is brho
theta_bend_ref1 = h_bend_ref1 * L_bend

P0_J_ref2 = p_ref2.p0c[0] / clight * qe
h_bend_ref2 = B_T * qe * p_ref2.charge / P0_J_ref2 # This is brho
theta_bend_ref2 = h_bend_ref2 * L_bend

dipole_ref1 = xt.Multipole(knl=[theta_bend_ref1],
                    length=L_bend, hxl=0.2)
dipole_ref2 = xt.Multipole(knl=[theta_bend_ref2],
                    length=L_bend, hxl=0.2)

n_slices = 100

dipole_ref1 = xt.Multipole(knl=[theta_bend_ref1/n_slices],
                    length=L_bend/n_slices, hxl=0.2/n_slices)
dipole_ref2 = xt.Multipole(knl=[theta_bend_ref2/n_slices],
                    length=L_bend/n_slices, hxl=0.2/n_slices)

ele_ref1 = []
for ii in range(n_slices):
    ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))
    ele_ref1.append(dipole_ref1)
    ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))

ele_ref2 = []
for ii in range(n_slices):
    ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))
    ele_ref2.append(dipole_ref2)
    ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))

line_ref1 = xt.Line(elements=ele_ref1)
line_ref2 = xt.Line(elements=ele_ref2)

line_ref1.build_tracker()
line_ref2.build_tracker()

line_ref1.track(p1)
line_ref2.track(p1_ref2)

assert np.isclose(p1.x, p1_ref2.x, atol=0, rtol=1e-14)
assert np.isclose(p1.y, p1_ref2.y, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
assert np.isclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)


# rpp : array_like of float, optional
#     m/m0 P0c / Pc = 1/(1+delta)