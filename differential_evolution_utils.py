import numpy as np
from scipy.optimize import differential_evolution
from simulation_utils import run_metanet_sim  # your METANET simulator
from optimization_utils import smooth_inflow


# ---------------------------
# decode_params (DE vector -> arrays)
# ---------------------------
def decode_params(params, num_calibrated_segments):
    """
    decode contiguous blocks: (eta_high N, tau N, K N, rho_crit N, v_free N, a N, beta N, r_inflow N)
    This matches the order used in encode_params_from_ipopt below and build_param_config when all ramps present.
    Note: if some ramps were excluded from bounds they will not appear in DE vector; decode_params assumes full blocks.
    If you use variable-length param vectors (due to skipping ramps), prefer params_from_best (name-based parsing).
    """
    idx = 0
    def get_vector(n):
        nonlocal idx
        vec = params[idx:idx+n]
        idx += n
        return np.array(vec, dtype=float)

    eta_high = get_vector(num_calibrated_segments)
    tau = get_vector(num_calibrated_segments)
    K = get_vector(num_calibrated_segments)
    rho_crit = get_vector(num_calibrated_segments)
    v_free = get_vector(num_calibrated_segments)
    a = get_vector(num_calibrated_segments)
    beta = get_vector(num_calibrated_segments)
    r_inflow = get_vector(num_calibrated_segments)

    return eta_high, tau, K, rho_crit, v_free, a, beta, r_inflow

# ---------------------------
# params_from_best (name-based parsing)
# ---------------------------
def params_from_best(param_names, best_params, num_calibrated_segments):
    """
    Robust parser: constructs arrays by scanning param_names (the authoritative ordering)
    Useful when build_param_config omitted some parameters (e.g., fixed ramps).
    Returns dict with arrays keyed by 'eta_high','tau','K','rho_crit','v_free','a','beta','r_inflow' and optional 'lanes'.
    """
    calibrated = {k: np.zeros(num_calibrated_segments) for k in
                  ['eta_high','tau','K','rho_crit','v_free','a','beta','r_inflow']}
    lanes_val = None

    idx = 0
    for name in param_names:
        val = best_params[idx]
        idx += 1
        if name == 'lanes':
            lanes_val = int(round(val))
            continue
        parts = name.split('_')
        base = '_'.join(parts[:-1]) if len(parts) > 1 else parts[0]
        try:
            seg_idx = int(parts[-1])
        except Exception:
            # if it's weird, skip
            continue

        # map possible base names to canonical keys
        if base in ['rho_crit', 'rho_crit']:
            key = 'rho_crit'
        elif base in ['r_inflow', 'r']:
            key = 'r_inflow'
        else:
            key = base

        if key in calibrated:
            calibrated[key][seg_idx] = val
        else:
            calibrated[key] = calibrated.get(key, [])
            calibrated[key].append((seg_idx, val))

    if lanes_val is not None:
        calibrated['lanes'] = lanes_val

    return calibrated

# ---------------------------
# loss_fn (same as yours)
# ---------------------------
def loss_fn(v_hat, rho_hat, q_hat, v_pred, rho_pred, q_pred):
    v_max = np.max(v_hat)
    rho_max = np.max(rho_hat)
    q_max = np.max(q_hat)
    total = 0.0
    # print("vhat shape:", v_hat.shape)
    # print("vpred shape:", v_pred.shape)
    for t in range(len(v_hat)):
        for i in range(len(v_hat[0])):
            total += (20 * ((v_pred[t, i] - v_hat[t, i]) / (v_max + 1e-12)) ** 2) \
                     + ((rho_pred[t, i] - rho_hat[t, i]) / (rho_max + 1e-12)) ** 2 \
                     + ((q_pred[t, i] - q_hat[t, i]) / (q_max + 1e-12)) ** 2
    return total

def objective_function(params, rho_hat, q_hat, T, l, num_calibrated_segments,
                       lane_mapping=None, smoothing=True,
                       opt=False, params_obj=None, sep_boundary_conditions=None):
    """
    Works for:
      - params vector (DE optimization)
      - params dictionary (exact simulation)
    """

    # -----------------------------
    # Decode params if vector
    # -----------------------------
    # print("Objective function called with params:", params)
    if isinstance(params, np.ndarray) or isinstance(params, list):
        eta_high, tau, K, rho_crit, v_free, a, beta, r_inflow = decode_params(
            np.array(params), num_calibrated_segments
        )

        sim_params = {
            'eta_high': eta_high,
            'tau': tau,
            'K': K,
            'rho_crit': rho_crit,
            'v_free': v_free,
            'a': a,
            'beta': beta,
            'r': r_inflow,
        }

        # PATCH: ensure ramp arrays exist
        if len(sim_params['beta']) == 0:
            sim_params['beta'] = np.zeros(num_calibrated_segments)
        if len(sim_params['r']) == 0:
            sim_params['r'] = np.zeros(num_calibrated_segments)

    elif isinstance(params, dict):
        # assume params dict
        sim_params = params
        sim_params['r'] = sim_params.get('r_inflow', np.zeros(num_calibrated_segments))
        sim_params['rho_crit'] = sim_params.get('rho_crit', np.zeros(num_calibrated_segments))
    
    if params_obj is not None:
        # assert that sim_params equals params_obj
        for key in params_obj:
            if key in sim_params:
                assert np.allclose(sim_params[key], params_obj[key]), f"Mismatch in param {key}"
    
    # print("Simulation parameters:", sim_params)
    # -----------------------------
    # Interior lanes and slices
    # -----------------------------
    if sep_boundary_conditions is not None:
        data_inflow = sep_boundary_conditions.get('upstream_flow', None)
        downstream_density = sep_boundary_conditions.get('downstream_density', None)
        num_lanes_array = lane_mapping
        scaled_rho_hat = rho_hat / num_lanes_array  # all segments
        v_hat_init = q_hat / rho_hat
        init_traffic_state = (scaled_rho_hat[0,:], v_hat_init[0,:], data_inflow[0] if data_inflow is not None else 0, 0)
    else:
        num_lanes_array = lane_mapping[1:-1]  # interior segments
        scaled_rho_hat = rho_hat[:, 1:-1] / num_lanes_array

        v_hat_init = q_hat[:, 1:-1] / rho_hat[:, 1:-1]

        # Inflow and downstream density
        data_inflow = smooth_inflow(q_hat[:, 0]) if smoothing else q_hat[:, 0]
        downstream_density = smooth_inflow(rho_hat[:, -1]) / num_lanes_array[-1] if smoothing else rho_hat[:, -1] / num_lanes_array[-1]

        # Initial traffic state
        init_traffic_state = (scaled_rho_hat[0,:], v_hat_init[0,:], data_inflow[0], 0)

    # Lane dictionary
    lanes_dict = {i: num_lanes_array[i] for i in range(num_calibrated_segments)}

    # -----------------------------
    # Run simulation
    # -----------------------------
    # print("Running METANET simulation for objective function evaluation...")
    rho_sim, v_sim, _, tts_sim = run_metanet_sim(
        T, l,
        init_traffic_state,
        data_inflow,
        downstream_density,
        sim_params,
        vsl_speeds=None,
        lanes=lanes_dict,
        plotting=True,
        real_data=True
    )
    # print("Simulation completed.")
    # Compute predicted flow
    rho_pred = rho_sim[:-1]
    v_pred = v_sim[:-1]
    q_pred_perlane = rho_pred * v_pred
    if sep_boundary_conditions is not None:
        # Keep all of q_hat
        q_hat_loss = q_hat / num_lanes_array
    else:
        q_hat_loss = q_hat[:, 1:-1] / num_lanes_array

    # Compute loss
    total_rmse = loss_fn(v_hat_init, scaled_rho_hat, q_hat_loss,
                         v_pred, rho_pred, q_pred_perlane)

    if opt:
        return rho_pred, v_pred, q_pred_perlane, total_rmse
    return total_rmse

import numpy as np
from scipy.optimize import differential_evolution
from simulation_utils import run_metanet_sim
from optimization_utils import smooth_inflow

# ---------------------------
# build_param_config
# ---------------------------
def build_param_config(num_calibrated_segments,
                       include_ramping=True,
                       lane_mapping=None,
                       on_ramp_mapping=None,
                       off_ramp_mapping=None):
    """
    Returns (param_names, bounds, fixed) for DE.
    Only includes ramps where allowed by on_ramp_mapping / off_ramp_mapping.
    """
    param_names = []
    bounds = []
    fixed = {}

    # Core METANET parameters per segment
    core_specs = [
        ('eta_high',  (5.0, 40.0)),
        ('tau',       (1.0/3600, 60.0/3600)),
        ('K',         (5.0, 60.0)),
        ('rho_crit',  (20.0, 120.0)),
        ('v_free',    (60.0, 120.0)),
        ('a',         (0.1, 4.0)),
    ]
    for base_name, b in core_specs:
        for i in range(num_calibrated_segments):
            param_names.append(f"{base_name}_{i}")
            bounds.append(b)

    if include_ramping:
        for i in range(num_calibrated_segments):
            # beta (off-ramp)
            param_names.append(f"beta_{i}")
            if off_ramp_mapping is None or off_ramp_mapping[i] == 1:
                
                bounds.append((1e-3, 0.9))
            else:
                bounds.append((0.0, 0.0))
        
        for i in range(num_calibrated_segments):
            param_names.append(f"r_inflow_{i}")
            # r_inflow (on-ramp)
            if on_ramp_mapping is None or on_ramp_mapping[i] == 1:
                
                bounds.append((1e-3, 2000.0))
            else:
                bounds.append((0.0, 0.0))
    else:
        # Fix ramps to zero
        for i in range(num_calibrated_segments):
            fixed[f"beta_{i}"] = 0.0
        for i in range(num_calibrated_segments):
            fixed[f"r_inflow_{i}"] = 0.0

    fixed['lanes'] = lane_mapping

    return param_names, bounds, fixed

# ---------------------------
# run_calibration
# ---------------------------
def run_calibration(rho_hat, q_hat, T, l,
                    num_calibrated_segments=1,
                    include_ramping=True,
                    lane_mapping=None,
                    on_ramp_mapping=None,
                    off_ramp_mapping=None,
                    smoothing=True,
                    de_opts=None,
                    sep_boundary_conditions=None,
                    opt=False):
    """
    Run differential evolution calibration.
    Returns calibrated_params dict with arrays for each parameter.
    """

    # Build DE config
    param_names, bounds, fixed = build_param_config(
        num_calibrated_segments,
        include_ramping=include_ramping,
        lane_mapping=lane_mapping,
        on_ramp_mapping=on_ramp_mapping,
        off_ramp_mapping=off_ramp_mapping
    )
    print("Parameter names for calibration:", param_names, len(param_names))

    # DE options
    de_opts = de_opts or {}
    default_opts = dict(
        strategy='randtobest1bin',   
        maxiter=200,                 
        popsize=30,                  
        tol=1e-7,                    
        mutation=(0.7, 1.9),         
        recombination=0.9,           
        disp=True,
        polish=True,                
        workers=-1,
        init='latinhypercube'
    )
    # Run differential evolution
    # print("Starting differential evolution calibration...")
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(rho_hat, q_hat, T, l, num_calibrated_segments, lane_mapping, smoothing, opt, None,sep_boundary_conditions),
        **{**default_opts, **de_opts}
    )
    # print("Differential evolution completed.")

    # Parse best parameters robustly
    from differential_evolution_utils import params_from_best
    best_vector = result.x
    # print final loss
    print(f"Final calibrated loss: {result.fun}")
    calibrated_raw = params_from_best(param_names, best_vector, num_calibrated_segments)

    # Build final calibrated params dict
    calibrated_params = {}
    for key in ['eta_high','tau','K','rho_crit','v_free','a','beta','r_inflow']:
        if key in calibrated_raw and calibrated_raw[key].size > 0:
            calibrated_params[key] = np.array(calibrated_raw[key], dtype=float)
        elif key in fixed:
            calibrated_params[key] = np.array(fixed[key], dtype=float)
        else:
            calibrated_params[key] = np.zeros(num_calibrated_segments, dtype=float)

    calibrated_params['lanes'] = np.array(lane_mapping, dtype=float)
    return calibrated_params
