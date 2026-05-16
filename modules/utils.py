import numpy as np
import streamlit as st
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def create_2d_laplacian(nx, ny, dx, dy):
    """Create 2D Laplacian with Dirichlet BCs (identity rows on boundaries)."""
    N = nx * ny
    A = lil_matrix((N, N))
    
    for j in range(ny):
        for i in range(nx):
            p = j * nx + i
            
            # Check if boundary node
            is_boundary = (i == 0 or i == nx-1 or j == 0 or j == ny-1)
            
            if is_boundary:
                A[p, p] = 1.0  # Dirichlet BC: phi = value (set via RHS)
            else:
                # Interior: 5-point stencil
                A[p, p] = -2.0/dx**2 - 2.0/dy**2
                A[p, p+1] = 1.0/dx**2    # right
                A[p, p-1] = 1.0/dx**2    # left
                A[p, p+nx] = 1.0/dy**2   # top
                A[p, p-nx] = 1.0/dy**2   # bottom
    
    return csr_matrix(A)

# -----------------------------------------------------------------------------
# Stabilized Navier-Stokes Solver with Diagnostics
# -----------------------------------------------------------------------------
#@njit(nopython=True)  # ← Add this line
def run_cavity_flow(nx, ny, Re, dt, t_end, upwind=True, verbose=False):
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    
    # Initial conditions
    psi = np.zeros((ny, nx))
    omega = np.zeros((ny, nx))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    
    A_lap = create_2d_laplacian(nx, ny, dx, dy)
    
    t_history = []
    kinetic_energy = []
    max_vorticity = []
    
    # CFL limits
    cfl_diff = Re * min(dx**2, dy**2) / 4 * 0.5  # 50% safety for stability
    cfl_adv = min(dx, dy) / 1.0 * 0.5  # Assume max vel = 1 (lid)
    dt_max = min(cfl_diff, cfl_adv)
    
    if verbose:
        st.info(f"📊 Grid: {nx}×{ny}, dx={dx:.4f}, dy={dy:.4f}")
        st.info(f"📊 CFL limits: diffusion={cfl_diff:.5f}, advection={cfl_adv:.5f}")
        st.info(f"📊 Your dt={dt:.5f}, Recommended dt≤{dt_max:.5f}")
    
    if dt > dt_max:
        st.warning(f"⚠️ **dt exceeds CFL limit!** Expected: dt ≤ {dt_max:.5f}")
    
    # Smooth lid startup (critical for stability!)
    lid_ramp_steps = max(100, int(0.5 / dt))  # Ramp over 0.5 time units
    lid_ramp_steps = min(lid_ramp_steps, int(t_end / dt))
    #lid_ramp_steps = 1000  # First 1000 steps: ramp up
    
    for step in range(int(t_end / dt)):
        t = step * dt
        t_history.append(t)
        
        # --- Step 1: Velocity from streamfunction (interior only) ---
        u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dy)
        v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
        
        # Lid BC with smooth ramp (CRITICAL!)
        lid_factor = min(1.0, step / lid_ramp_steps)
        u[-1, 1:-1] = lid_factor * 1.0  # Don't set corners!
        u[-1, 0] = 0.0  # Left corner stationary
        u[-1, -1] = 0.0  # Right corner stationary
        
        # --- Step 2: Advect vorticity (interior only) ---
        omega_new = omega.copy()
        
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if upwind:
                    # First-order upwind (more diffusive but stable)
                    if u[j,i] >= 0:
                        dwdx = (omega[j,i] - omega[j,i-1]) / dx
                    else:
                        dwdx = (omega[j,i+1] - omega[j,i]) / dx
                    
                    if v[j,i] >= 0:
                        dwdy = (omega[j,i] - omega[j-1,i]) / dy
                    else:
                        dwdy = (omega[j+1,i] - omega[j,i]) / dy
                else:
                    # Central difference (less stable)
                    dwdx = (omega[j,i+1] - omega[j,i-1]) / (2*dx)
                    dwdy = (omega[j+1,i] - omega[j-1,i]) / (2*dy)
                
                conv = u[j,i] * dwdx + v[j,i] * dwdy
                lap_omega = (omega[j,i+1] + omega[j,i-1] - 2*omega[j,i]) / dx**2 + \
                           (omega[j+1,i] + omega[j-1,i] - 2*omega[j,i]) / dy**2
                
                omega_new[j,i] = omega[j,i] + dt * (-conv + lap_omega / Re)
        
            
        # --- Step 3: Vorticity BCs (Thom's formula - CRITICAL!) ---
        # Bottom wall (j=0)
        omega_new[0, 1:-1] = -2 * (psi[1, 1:-1] - psi[0, 1:-1]) / dy**2
        # Top wall (lid, j=ny-1)
        omega_new[-1, 1:-1] = -2 * (psi[-2, 1:-1] - psi[-1, 1:-1]) / dy**2 - 2 * lid_factor / dy
        # Left wall (i=0)
        omega_new[1:-1, 0] = -2 * (psi[1:-1, 1] - psi[1:-1, 0]) / dx**2
        # Right wall (i=nx-1)
        omega_new[1:-1, -1] = -2 * (psi[1:-1, -2] - psi[1:-1, -1]) / dx**2
        
        # Corner vorticity (average of adjacent walls)
        omega_new[0, 0] = 0.5 * (omega_new[0, 1] + omega_new[1, 0])
        omega_new[0, -1] = 0.5 * (omega_new[0, -2] + omega_new[1, -1])
        omega_new[-1, 0] = 0.5 * (omega_new[-1, 1] + omega_new[-2, 0])
        omega_new[-1, -1] = 0.5 * (omega_new[-1, -2] + omega_new[-2, -1])
        
        omega = omega_new
        
        # --- Check for NaN/Inf BEFORE Poisson solve ---
        if np.any(np.isnan(omega)) or np.any(np.isinf(omega)):
            st.error(f"💥 **Vorticity exploded at step {step} (t={t:.4f})**")
            st.error("Cause: dt too large or BC error")
            break
        
        # --- Step 4: Solve Poisson for streamfunction ---
        rhs = -omega.flatten()
        
        # Apply BCs to RHS (psi = 0 on all walls)
        for j in range(ny):
            for i in range(nx):
                if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                    p = j * nx + i
                    rhs[p] = 0.0  # Dirichlet BC
        
        try:
            psi_flat = spsolve(A_lap, rhs)
            if np.any(np.isnan(psi_flat)) or np.any(np.isinf(psi_flat)):
                st.error("💥 **Poisson solver failed**")
                break
            psi = psi_flat.reshape((ny, nx))
        except Exception as e:
            st.error(f"💥 **Linear solve failed: {e}**")
            break
        
        # --- Kinetic energy (interior only, exclude BCs) ---
        ke = 0.5 * np.sum(u[1:-1, 1:-1]**2 + v[1:-1, 1:-1]**2) / ((nx-2)*(ny-2))
        kinetic_energy.append(ke)
        max_vorticity.append(np.max(np.abs(omega)))
    
    return x, y, u, v, psi, omega, t_history, kinetic_energy, max_vorticity

# Enhanced particle tracking with speed coloring
def track_particles_with_speed(u, v, x, y, seed_points, dt, n_steps):
    """Track particles and record speed along each trajectory."""
    from scipy.interpolate import RegularGridInterpolator
    
    u_interp = RegularGridInterpolator((y, x), u, method='linear', bounds_error=False, fill_value=0)
    v_interp = RegularGridInterpolator((y, x), v, method='linear', bounds_error=False, fill_value=0)
    
    trajectories = []
    
    for x0, y0 in seed_points:
        traj = []
        speeds = []
        x_curr, y_curr = x0, y0
        
        for i in range(n_steps):
            if x_curr < 0 or x_curr > 1 or y_curr < 0 or y_curr > 1:
                break
            
            traj.append([x_curr, y_curr])
            
            u_curr = u_interp([y_curr, x_curr])[0]
            v_curr = v_interp([y_curr, x_curr])[0]
            speeds.append(np.sqrt(u_curr**2 + v_curr**2))
            
            x_curr += u_curr * dt
            y_curr += v_curr * dt
        
        if len(traj) >= 2:
            trajectories.append({
                'path': np.array(traj),
                'speed': np.array(speeds)
            })
    
    return trajectories

def recover_pressure(u, v, Re, dx, dy):
    """
    Recover pressure from velocity field using Poisson equation.
    FIXED: Proper bounds checking for finite differences.
    """
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    
    ny, nx = u.shape
    N = nx * ny
    
    # Build Laplacian matrix
    A = lil_matrix((N, N))
    
    for j in range(ny):
        for i in range(nx):
            p = j * nx + i
            
            # Check if boundary node
            is_boundary = (i == 0 or i == nx-1 or j == 0 or j == ny-1)
            
            if is_boundary:
                A[p, p] = 1.0  # Neumann BC: dp/dn = 0
            else:
                A[p, p] = -2.0/dx**2 - 2.0/dy**2
                A[p, p+1] = 1.0/dx**2
                A[p, p-1] = 1.0/dx**2
                A[p, p+nx] = 1.0/dy**2
                A[p, p-nx] = 1.0/dy**2
    
    A = csr_matrix(A)
    
    # Compute RHS (source term from velocity gradients)
    rhs = np.zeros((ny, nx))
    
    # ✅ FIXED: Only loop over INTERIOR points (not boundaries)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            # ∂²(u²)/∂x²
            d2u2dx2 = (u[j,i+1]**2 - 2*u[j,i]**2 + u[j,i-1]**2) / dx**2
            # ∂²(v²)/∂y²
            d2v2dy2 = (v[j+1,i]**2 - 2*v[j,i]**2 + v[j-1,i]**2) / dy**2
            # 2∂²(uv)/∂x∂y (mixed derivative)
            d2uvdxdy = ((u[j+1,i+1]*v[j+1,i+1] - u[j+1,i-1]*v[j+1,i-1]) -
                        (u[j-1,i+1]*v[j-1,i+1] - u[j-1,i-1]*v[j-1,i-1])) / (4*dx*dy)
            
            rhs[j,i] = -(d2u2dx2 + 2*d2uvdxdy + d2v2dy2)
    
    # Apply Neumann BCs for pressure (dp/dn = 0)
    # For homogeneous Neumann, RHS at boundaries = 0 (already initialized)
    
    # Solve Poisson equation
    try:
        p_flat = spsolve(A, rhs.flatten())
        p = p_flat.reshape((ny, nx))
        
        # Normalize (pressure is relative)
        p -= np.mean(p)
        
        return p
    except Exception as e:
        st.error(f"❌ Pressure solve failed: {e}")
        return np.zeros_like(u)  # Fallback
    
def extract_ghia_profiles(u, v, x, y):
    """
    Extract centerline profiles for Ghia benchmark comparison.
    
    Returns
    -------
    y_centerline, u_along_vertical : u at x=0.5 vs y
    x_centerline, v_along_horizontal : v at y=0.5 vs x
    """
    nx, ny = len(x), len(y)
    
    # Find centerline indices
    i_center = nx // 2  # x = 0.5
    j_center = ny // 2  # y = 0.5
    
    # u along vertical centerline (x = 0.5)
    u_vertical = u[:, i_center]
    
    # v along horizontal centerline (y = 0.5)
    v_horizontal = v[j_center, :]
    
    return y, u_vertical, x, v_horizontal

