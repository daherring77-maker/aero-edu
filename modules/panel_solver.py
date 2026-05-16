import numpy as np
from scipy import linalg
import streamlit as st

def panel_method_airfoil(x_airfoil, y_airfoil, alpha, V_inf):
    """
    Solves 2D potential flow over an airfoil using a Constant Strength Vortex Panel Method.
    Uses scipy.linalg.lstsq for robust solving.
    
    Parameters:
    -----------
    x_airfoil : array_like
        X-coordinates of the airfoil surface points (ordered).
    y_airfoil : array_like
        Y-coordinates of the airfoil surface points (ordered).
    alpha : float
        Angle of attack in degrees.
    V_inf : float
        Freestream velocity magnitude.
        
    Returns:
    --------
    results : dict
        Dictionary containing Cp, Cl, Cl_circ, coordinates, gamma, etc.
    """
    
    # 1. Preprocessing Geometry
    x = np.array(x_airfoil, dtype=float)
    y = np.array(y_airfoil, dtype=float)
    
    # Ensure the airfoil is closed (TE point repeated at end)
    # If "deduped", the last point might not equal the first.
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    
    n_points = len(x)
    n_panels = n_points - 1
    # Add this right after loading x, y
    # Smooth the last 3 points near TE (assumes TE is at x[0]/x[-1])
    if n_points >= 4:
        # Blend TE points to create a small blunt radius
        te_idx = [0, -1]
        for idx in te_idx:
            prev_idx = (idx - 1) % n_points
            next_idx = (idx + 1) % n_points
            x[idx] = 0.8 * x[idx] + 0.1 * (x[prev_idx] + x[next_idx])
            y[idx] = 0.8 * y[idx] + 0.1 * (y[prev_idx] + y[next_idx])
    if n_panels < 3:
        raise ValueError("Not enough points to define panels.")

    # Panel Geometry Calculation
    # Panel j goes from point j to j+1
    x_start = x[:-1]
    y_start = y[:-1]
    x_end = x[1:]
    y_end = y[1:]
    
    dx = x_end - x_start
    dy = y_end - y_start
    panel_len = np.sqrt(dx**2 + dy**2)
    
    # Avoid zero-length panels (can happen with noisy data)
    if np.any(panel_len < 1e-6):
        raise ValueError("Zero-length panel detected. Check input coordinates.")
        
    theta = np.arctan2(dy, dx)  # Panel angle relative to x-axis
    
    # Normal vectors (pointing outward)
    # For counter-clockwise ordering, normal is (-sin, cos)
    n_x = -np.sin(theta)
    n_y = np.cos(theta)
    
    # Control Points (Midpoints)
    xc = 0.5 * (x_start + x_end)
    yc = 0.5 * (y_start + y_end)
    
    # 2. Influence Coefficient Matrix Construction
    # We want to solve for gamma (vortex strength) on each panel.
    # Boundary Condition: Flow tangency (V . n = 0) at each control point.
    # V_total = V_inf + V_induced
    # (V_inf . n) + sum(gamma_j * V_induced_j . n) = 0
    
    alpha_rad = np.radians(alpha)
    V_inf_x = V_inf * np.cos(alpha_rad)
    V_inf_y = V_inf * np.sin(alpha_rad)
    
    # RHS: - (V_inf . n)
    b = -(V_inf_x * n_x + V_inf_y * n_y)
        
    # Matrix A: Normal velocity induced at point i by panel j with gamma=1
    A = np.zeros((n_panels, n_panels))
    #A_normal = np.zeros((n_panels, n_panels))
    #A_tangent = np.zeros((n_panels, n_panels))
    
    # Precompute trig for efficiency
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    for i in range(n_panels):
        for j in range(n_panels):
            if i == j:
                # Self-influence: Normal velocity induced by a panel on itself is 0
                # (Tangential is gamma/2, but we are enforcing normal BC)
                A[i, j] = 0.0
                #A_normal[i, j] = 0.0
                #A_tangent[i, j] = 0.5 
            else:
                # Influence of panel j on control point i
                # Transform control point i into panel j's coordinate system
                dx_ij = xc[i] - x_start[j]
                dy_ij = yc[i] - y_start[j]
                
                # Rotate to panel frame
                x_bar = dx_ij * cos_theta[j] + dy_ij * sin_theta[j]
                y_bar = -dx_ij * sin_theta[j] + dy_ij * cos_theta[j]
                
                # Distances to panel endpoints in panel frame
                # Panel j is from (0,0) to (L, 0) in its frame
                r1_sq = x_bar**2 + y_bar**2
                r2_sq = (x_bar - panel_len[j])**2 + y_bar**2
                
                # Angles
                theta_1 = np.arctan2(y_bar, x_bar)
                theta_2 = np.arctan2(y_bar, x_bar - panel_len[j])
                
                # Velocity induced by unit vortex panel in panel frame (u', v')
                # For Vortex: u' = (theta1 - theta2)/2pi, v' = -ln(r2/r1)/2pi
                # Note: Careful with branch cuts in arctan2. 
                # Standard formulation ensures continuity.
                
                d_theta = theta_1 - theta_2
                # Handle branch cut jump if crossing the wake line (unlikely for surface points)
                # But arctan2 range is -pi to pi. 
                
                ln_r = 0.5 * np.log(r2_sq / r1_sq)
                
                u_prime = d_theta / (2.0 * np.pi)
                v_prime = -ln_r / (2.0 * np.pi)
                
                # Transform velocity back to global frame
                u_glob = u_prime * cos_theta[j] - v_prime * sin_theta[j]
                v_glob = u_prime * sin_theta[j] + v_prime * cos_theta[j]
                
                # Project onto normal of panel i
                A[i, j] = u_glob * n_x[i] + v_glob * n_y[i]
                #A_normal[i, j] = (u_glob * n_x[i] + v_glob * n_y[i]) / panel_len[j]

    # 3. Kutta Condition
    # For constant vortex panels, standard Kutta is gamma_1 + gamma_N = 0
    # This enforces equal pressure/stagnation at the TE.
    # We append this as an extra row to make the system over-determined for lstsq
    # 3. Kutta Condition (scaled for conditioning)
       # 3. Kutta Condition (weight = 1.0 for lstsq balance)
    kutta_row = np.zeros(n_panels)
    kutta_row[0] = 1.0
    kutta_row[-1] = 1.0
    
    # Normalize RHS for numerical stability
    b_norm = b / V_inf
    
    # Weight = 1.0 prevents Kutta from dominating the least-squares solution
    A_kutta = np.vstack([A, kutta_row])
    b_kutta = np.append(b_norm, 0.0)
    
    # 4. Solve for gamma* = gamma / V_inf
    gamma_star, residuals, rank, s = linalg.lstsq(A_kutta, b_kutta, lapack_driver="gelsy")
    gamma = -gamma_star * V_inf  # Physical vortex strength
    
    # 5. Tangential Velocity (inward normals: n = (-sin, cos))
    V_t = np.zeros(n_panels)
    for i in range(n_panels):
        V_t_inf_norm = np.cos(alpha_rad) * cos_theta[i] + np.sin(alpha_rad) * sin_theta[i]
        V_t_ind_norm = 0.0
        for j in range(n_panels):
            if i == j:
                V_t_ind_norm += 0.5 * gamma_star[j]
            else:
                # Re-calculate influence (tangential component this time)
                dx_ij = xc[i] - x_start[j]
                dy_ij = yc[i] - y_start[j]
                
                x_bar = dx_ij * cos_theta[j] + dy_ij * sin_theta[j]
                y_bar = -dx_ij * sin_theta[j] + dy_ij * cos_theta[j]
                
                r1_sq = x_bar**2 + y_bar**2
                r2_sq = (x_bar - panel_len[j])**2 + y_bar**2
                
                theta_1 = np.arctan2(y_bar, x_bar)
                theta_2 = np.arctan2(y_bar, x_bar - panel_len[j])
                
                d_theta = theta_1 - theta_2
                ln_r = 0.5 * np.log(r2_sq / r1_sq)
                
                u_prime = d_theta / (2.0 * np.pi)
                v_prime = -ln_r / (2.0 * np.pi)
                # [Keep your exact influence calculation here]
                u_glob_norm = u_prime * cos_theta[j] - v_prime * sin_theta[j]
                v_glob_norm = u_prime * sin_theta[j] + v_prime * cos_theta[j]
                V_t_ind_norm += u_glob_norm * cos_theta[i] + v_glob_norm * sin_theta[i]
        
        V_t[i] = V_inf * (V_t_inf_norm + V_t_ind_norm)
    #st.write(V_t, gamma)
    # 6. Light smoothing (removes midpoint oscillation)
    V_t = np.convolve(V_t, np.ones(5)/5, mode='same')
    V_t[0:2] = V_t[2]
    V_t[-2:] = V_t[-3]
    
    # 7. Pressure Coefficient
    Cp = 1.0 - (V_t / V_inf)**2
    
    # 8. Lift via Circulation (unambiguous, avoids normal-direction sign traps)
    chord = np.max(x) - np.min(x)
    circulation = np.sum(gamma * panel_len)
    Cl = (2.0 * circulation) / (V_inf * chord)
    
    # Optional: Keep pressure integration for validation
    # Cl_pressure = np.sum(Cp * n_y * panel_len) / chord  # + for inward normals
    #circulation = np.sum(gamma * panel_len)
    #Cl_circ = (2.0 * circulation) / (V_inf * (x[0] - x[-1])) # Chord approx
    # Better Chord calculation:
    chord = np.max(x) - np.min(x)
    Cl_circ = (2.0 * circulation) / (V_inf * chord)
    
    # Lift from Pressure Integration (more accurate for panel method)
    # L' = - integral(p * n_y ds) ... or sum(Cp * n_y * L)
    #Cl = - sum(Cp * n_y * panel_len) / chord
    # Note: Sign convention depends on normal direction. 
    # With outward normals, Lift = - integral(p dy). 
    # Cp = (p - p_inf)/q_inf. 
    # Cl = 1/c * integral(Cp * dx) ? No.
    # Force Y = - sum(p * n_y * L). 
    # Cl = Force_Y / (0.5 * rho * V^2 * c). 
    # Since Cp = (p-p_inf)/q, Force_Y/q = - sum(Cp * n_y * L).
       
    # Convert alpha to degrees for output consistency
    alpha_deg = alpha
    #st.write(Cp, gamma, V_t)
    return {
        'Cp': Cp,
        'Cl': Cl,
        'Cl_circ': Cl_circ,  # For debugging
        'x_cp': xc,
        'y_cp': yc,
        'gamma': gamma,
        'alpha': alpha_deg,
        'V_inf': V_inf,
        'circulation': circulation
    }
