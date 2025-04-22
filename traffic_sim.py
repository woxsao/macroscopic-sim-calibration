import numpy as np

# Network parameters
tau = 18/3600 #33
p_crit = 37.45 #27.6 #31.1
a = 1.4 #2.5 #2.9
v_free = 120
p_max = 180 #84.5
K = 40 #10
eta_high = 30 #21.3
eta_low = 60
q_capacity = 2200
v_start = 100 # starting speed when vehicles enter the freeway

## Dynamics for METANET
def density_dynamics(current, inflow, outflow, lanes, T, l):
    return current + T/(l * lanes) * (inflow - outflow)

def flow_dynamics(density, velocity, lanes):
    # if density * velocity > q_capacity:
    #     print(density, velocity, lanes)
    return density * velocity * lanes

def queue_dynamics(current, demand, flow_origin, T):
    return current + T * (demand - flow_origin)

def calculate_V(rho, v_ctrl):
    return min(v_free * np.exp(-1/a * (rho / p_crit)**a), v_ctrl) #if rho == 0 else min(v_free * np.exp(-1/a * (rho / p_crit)**a), v_ctrl, 2100/rho)

def velocity_dynamics_MN(current, prev_state, density, next_density, current_density, v_ctrl, T, l):
    next =  current + T/tau * (calculate_V(density, v_ctrl) - current) + T/l * current * (prev_state - current) - (eta_high * T) / (tau * l) * (next_density - density) / (density + K)
    return max(0, next) #if current_density == 0 else min(max(0, next), q_capacity/current_density)

def origin_flow_dynamics_MN(demand, density_first, queue, lanes, T):
    return min(demand + queue / T, lanes * q_capacity * (p_max - density_first)/(p_max - p_crit), lanes * q_capacity)

# From linearized METANET
# def velocity_dynamics(current, current_estimate, prev_state, density, next_density, V_term, measured_density, T, l):
#     next =  current + T/tau * (V_term - current) + T/l * current_estimate * (prev_state - current) - (eta_high * T) / (tau * l) * (next_density - density) / (measured_density + K)
#     return next #max(0, next)

def metanet_sim(T, l, init_traffic_state, vsl_speeds, demand, downstream_density, lanes=None, plotting=False):
    if not lanes:
        lanes = {i: 1 for i in range(vsl_speeds.shape[1])}

    initial_density, initial_velocity, initial_flow_or, initial_queue = init_traffic_state

    time_steps, num_segments = vsl_speeds.shape
    # Initialize state variables (velocity, density, flow, demand, queue, origin flow) for simulation
    velocity = np.zeros((time_steps + 1, num_segments))
    density = np.zeros((time_steps + 1, num_segments))
    flow = np.zeros((time_steps + 1, num_segments)) 
    queue = np.zeros((time_steps + 1, 1))
    flow_origin = np.zeros((time_steps + 1, 1))

    # initial conditions
    density[0] = initial_density
    velocity[0] = initial_velocity
    flow[0] = np.array([initial_density[i] * initial_velocity[i] * lanes[i] for i in range(num_segments)])
    flow_origin[0, 0] = initial_flow_or
    queue[0, 0] = initial_queue

    # Step forward simulation of velocity, density, and flow
    for t in range(0, time_steps):
        # Update density
        for i in range(num_segments):
            if i == 0:
                density[t + 1, i] = density_dynamics(density[t, i], flow_origin[t, 0], flow[t, i], lanes[i], T, l)
            else:
                density[t + 1, i] = density_dynamics(density[t, i], flow[t, i-1], flow[t, i], lanes[i], T, l)

        # Update velocity
        for i in range(num_segments):
            if i == 0:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i], density[t, i], density[t, i + 1], density[t+1, i], vsl_speeds[t, i], T, l)
            elif i == num_segments - 1:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i - 1], density[t, i], downstream_density[t], density[t+1, i], vsl_speeds[t, i], T, l)
            else:
                velocity[t + 1, i] = velocity_dynamics_MN(velocity[t, i], velocity[t, i - 1], density[t, i], density[t, i + 1], density[t+1, i], vsl_speeds[t, i], T, l)

        # Update queue
        queue[t + 1, 0] = queue_dynamics(queue[t, 0], demand[t], flow_origin[t, 0], T)

        # Update flow
        for i in range(num_segments):
            flow[t+1, i] = flow_dynamics(density[t+1, i], velocity[t+1, i], lanes[i])
        
        # Update origin flow
        flow_origin[t+1, 0] = origin_flow_dynamics_MN(demand[t+1], density[t+1, 0], queue[t+1, 0], lanes[0], T)
    
    total_travel_time = T * (sum([np.sum(density[:, i]) * lanes[i] * l for i in range(num_segments)]) + np.sum(queue))
    # print(density)
    if plotting:
        return density, velocity, total_travel_time, flow_origin
    else:
        return (density[-1], velocity[-1], flow_origin[-1, 0], queue[-1, 0]), total_travel_time