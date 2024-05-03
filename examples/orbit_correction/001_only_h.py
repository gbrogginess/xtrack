import xtrack as xt
import numpy as np
from numpy.matlib import repmat

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
tt = line.get_table()
line.twiss_default['co_search_at'] = 'ip7'

tw = line.twiss4d()

# Select monitors by names (starting by "bpm" and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
h_monitor_names = tt_monitors.name

# Select h correctors by names (starting by "mcb.", containing "h.", and ending by ".b1")
tt_h_correctors = tt.rows['mcb.*'].rows['.*h\..*'].rows['.*\.b1']
h_corrector_names = tt_h_correctors.name

# Add correction knobs to the correctors
h_correction_knobs = []
for nn_kick in h_corrector_names:
    corr_knob_name = f'orbit_corr_{nn_kick}'
    assert hasattr(line[nn_kick], 'knl')
    line.vars[corr_knob_name] = 0
    line.element_refs[nn_kick].knl[0] += line.vars[f'orbit_corr_{nn_kick}']
    h_correction_knobs.append(corr_knob_name)

# Build response matrix
betx_monitors = tw.rows[h_monitor_names].betx
betx_correctors = tw.rows[h_corrector_names].betx

mux_monitor = tw.rows[h_monitor_names].mux
mux_correctors = tw.rows[h_corrector_names].mux

n_h_monitors = len(h_monitor_names)
n_hcorrectors = len(h_corrector_names)

qx = tw.qx

bet_prod_x = np.atleast_2d(betx_monitors).T @ np.atleast_2d(betx_correctors)
mux_diff = (repmat(mux_monitor, n_hcorrectors, 1).T
                    - repmat(mux_correctors, n_h_monitors, 1))

# Slide 28
# https://indico.cern.ch/event/1328128/contributions/5589794/attachments/2786478/4858384/linearimperfections_2024.pdf
response_matrix_x = (np.sqrt(bet_prod_x) / 2 / np.sin(np.pi * qx)
                     * np.cos(np.pi * qx - 2*np.pi*np.abs(mux_diff)))


# Introduce some orbit perturbation
h_kicks = {'mcbh.15r7.b1': 1e-5, 'mcbh.21r7.b1':-3e-5}

kick_vect_x = np.zeros(n_hcorrectors)

for nn_kick, kick in h_kicks.items():
    line.element_refs[nn_kick].knl[0] -= kick
    i_h_kick = np.where(h_corrector_names == nn_kick)[0][0]
    kick_vect_x[i_h_kick] = kick

for iter in range(1):
    # Measure the orbit
    tw_meas = line.twiss4d(only_orbit=True)

    x_meas = tw_meas.rows[h_monitor_names].x
    s_x_meas = tw_meas.rows[h_monitor_names].s

    # Compute the correction with least squares
    correction_x, residual_x, rank_x, sval_x = np.linalg.lstsq(
                            response_matrix_x, -x_meas, rcond=None)

    # Apply correction
    for nn_knob, kick in zip(h_correction_knobs, correction_x):
        line.vars[nn_knob] -= kick # knl[0] is -kick

    tw_after = line.twiss4d(only_orbit=True)

    print('max x: ', tw_after.x.max())

# Extract kicks from the knobs
applied_kicks = np.array([line.vv[nn_knob] for nn_knob in h_correction_knobs])