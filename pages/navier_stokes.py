import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import plotly.io as pio
from modules.plotting import create_cavity_results_zip
from modules.utils import recover_pressure, run_cavity_flow, track_particles_with_speed
import time, datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 2D Laplacian with Proper BC Handling
# -----------------------------------------------------------------------------
# At top of app.py
from modules.plotting import (
    create_velocity_plot, 
    create_streamline_plot, 
    create_vorticity_plot,
    create_ke_plot,
    create_pressure_plot,
    plot_pathlines
)

# -----------------------------------------------------------------------------
# Streamlit UI with Proper Precision
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="2D Aerodynamics Lab")
st.title("✈️ 2D Navier-Stokes: Streamfunction-Vorticity")
st.markdown("*Pure Python • Windows-native • Educational*")

st.sidebar.header("🎛️ Parameters")
# Initialize
if 'results' not in st.session_state:
    st.session_state.results = None
if 'params' not in st.session_state:
    st.session_state.params = None
# Grid
nx = st.sidebar.slider("Grid Points (X)", 32, 128, 48, 8)
ny = st.sidebar.slider("Grid Points (Y)", 32, 128, 48, 8)

# Physics
Re = st.sidebar.number_input("Reynolds Number", value=30.0, step=10.0, format="%.1f")

# ⚠️ CRITICAL: Proper format string for small dt
dt = st.sidebar.number_input("Time Step (dt)", value=0.0001, step=0.00001, format="%.5f")
#t_end = st.sidebar.number_input("End Time", value=2.0, step=0.5, format="%.1f")
t_end = st.sidebar.number_input("End Time", value=10.0, step=1.0, format="%.1f")

# Numerics
upwind = st.sidebar.checkbox("Upwind Scheme (more stable)", value=True)
verbose = st.sidebar.checkbox("Show CFL Diagnostics", value=True)
auto_dt = st.sidebar.checkbox("🛡️ Auto-limit dt to CFL", value=True)

# Auto-stability override
if auto_dt:
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    cfl_diff = Re * min(dx**2, dy**2) / 4 * 0.5
    cfl_adv = min(dx, dy) * 0.5
    dt_max = min(cfl_diff, cfl_adv)
    dt = min(dt, dt_max)
    st.sidebar.info(f"🔒 dt limited to {dt:.5f}")

# Known stable presets
if st.sidebar.button("🎯 Load Stable Preset (Re=30)"):
    nx, ny = 48, 48
    Re = 30.0
    dt = 0.00005
    t_end = 2.0
    st.sidebar.success("Loaded stable configuration!")

if st.sidebar.button("▶️ Run Simulation"):
    with st.spinner(f"Solving Re={Re} on {nx}×{ny} grid..."):
        start = time.time()
        try:
            x, y, u, v, psi, omega, t_hist, ke_hist, vort_hist = run_cavity_flow(
                nx, ny, Re, dt, t_end, upwind, verbose
            )
            elapsed = time.time() - start
            # Store in session_state
            # ✅ With params
            # Store BOTH results and trajectories
            st.session_state.results = {
            'x': x, 'y': y, 'u': u, 'v': v,
            'psi': psi, 'omega': omega,
            't_hist': t_hist, 'ke_hist': ke_hist, 
            'params': {'Re': Re, 'dt': dt, 'grid': (nx, ny)},
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
             }
            # Generate trajectories
            seed_points = [(0.1, 0.5), (0.1, 0.3), (0.1, 0.7)]
            trajs = track_particles_with_speed(u, v, x, y, seed_points, dt=0.01, n_steps=200)
            st.session_state.trajectories = trajs
            st.success(f"✅ Solved in {elapsed:.1f}s ({len(t_hist)} steps)")
            
            # Stability check
            if len(ke_hist) < len(t_hist):
                st.error("⚠️ Simulation stopped early due to instability")
            elif np.any(np.isnan(ke_hist)) or np.any(np.isinf(ke_hist)):
                st.error("⚠️ Kinetic energy contains NaN/Inf")
            else:
                st.info(f"📊 Final KE: {ke_hist[-1]:.4e}, Max ω: {vort_hist[-1]:.2f}")
            
           # In your Streamlit app, AFTER the simulation completes:
            if st.session_state.results is not None:
                results = st.session_state.results
                params = st.session_state.params
                trajs = st.session_state.trajectories
                #trajs = st.session_state.get('trajectories', None)  # Store trajs in session_state too!
                u, v = results['u'], results['v']
                Re = results['params']['Re']
                dt = results['params']['dt']
                grid = results['params']['grid']

                # Show parameters
                st.info(f"📊 Re={Re}, Grid={grid[0]}x{grid[1]}, dt={dt:.5f}")
                # Grid spacing
                # Robust grid extraction
                ny, nx = u.shape
                x_raw, y_raw = results['x'], results['y']

                st.write(f"x_raw shape: {x_raw.shape}, ndim: {x_raw.ndim}")
                st.write(f"y_raw shape: {y_raw.shape}, ndim: {y_raw.ndim}")
                
               # Handle different storage formats
                if x_raw.ndim == 2:
                    # Meshgrid format (ny, nx)
                    x_1d = x_raw[0, :]  # First row
                    y_1d = y_raw[:, 0]  # First column
                elif x_raw.ndim == 1:
                    # 1D array format
                    x_1d = x_raw
                    y_1d = y_raw
                else:
                    # Fallback: generate from velocity shape
                    x_1d = np.linspace(0, 1, nx)
                    y_1d = np.linspace(0, 1, ny)
                
                st.write(f"x_1d shape: {x_1d.shape}, len: {len(x_1d)}")
                st.write(f"y_1d shape: {y_1d.shape}, len: {len(y_1d)}")
                
                # ✅ Safe grid spacing calculation with bounds checking
                if len(x_1d) > 1:
                    dx = x_1d[1] - x_1d[0]
                else:
                    dx = 1.0 / (nx - 1) if nx > 1 else 0.01
                
                if len(y_1d) > 1:
                    dy = y_1d[1] - y_1d[0]
                else:
                    dy = 1.0 / (ny - 1) if ny > 1 else 0.01
                
                st.write(f"Grid: {nx}×{ny}, dx={dx:.5f}, dy={dy:.5f}")
                x = x_raw[0, :] if x_raw.ndim == 2 else x_raw
                y = y_raw[:, 0] if y_raw.ndim == 2 else y_raw
                
                #st.write("### DEBUG: Results Structure")
                #st.write(f"**u.shape**: {results['u'].shape}")
                #st.write(f"**x.shape**: {results['x'].shape}, **x.ndim**: {results['x'].ndim}")
                #st.write(f"**y.shape**: {results['y'].shape}, **y.ndim**: {results['y'].ndim}")
                #st.write(f"**x[0]**: {results['x'].flat[0] if results['x'].size > 0 else 'empty'}")
                #st.write(f"**x[-1]**: {results['x'].flat[-1] if results['x'].size > 0 else 'empty'}")
                #st.write(f"**len(x)**: {len(results['x'])}")
                #st.write(f"**nx from u.shape**: {results['u'].shape[1]}")
                
                # ✅ Recover pressure (post-processing, NOT a new simulation)
                with st.spinner("Recovering pressure field..."):
                    p = recover_pressure(u, v, Re, dx, dy)
                
                # Store in results for plotting/export
                st.session_state.results['pressure'] = p
                st.success("✅ Pressure recovered!")
                
                                    
                fig_ke = create_ke_plot(t_hist, ke_hist, for_export=False)
                st.plotly_chart(fig_ke, width='stretch')
                
                fig_vort = create_vorticity_plot(omega, x, y)
                st.plotly_chart(fig_vort, width='stretch')
               
                # Velocity plot
                fig_u = create_velocity_plot(u, x, y)
                st.plotly_chart(fig_u, width='stretch')
                                                      
                # Streamlines plot
                fig_psi = create_streamline_plot(psi, x, y)
                st.plotly_chart(fig_psi, width='stretch')

                fig_p = create_pressure_plot(p, results['x'], results['y'], for_export=False)
                st.plotly_chart(fig_p, width='stretch')

                # Usage in Streamlit:
                # In your Streamlit app
                if trajs is not None:
                    fig_trajs = plot_pathlines(trajs, title="Particle Pathlines", for_export=False)
                    st.plotly_chart(fig_trajs, width='stretch')
             
             
            # === ZIP DOWNLOAD BUTTON ===
            zip_data = create_cavity_results_zip(results, trajs=trajs)
            st.download_button(
                label="📦 Download All Results (ZIP)",
                data=zip_data,
                file_name=f"cavity_Re{results['params']['Re']}_results.zip",
                mime="application/zip"
    )                          
              
            
                                    
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("💡 Try: dt=0.00005, Re=30, grid=48×48, upwind=ON")
else:
    st.info("👈 Adjust parameters and click **Run Simulation**")
    
    with st.expander("📚 Stable Configuration Guide"):
        st.markdown("""
        ### ✅ Known Stable Settings
        
        | Reynolds | Grid | dt | t_end | Notes |
        |----------|------|------|-------|-------|
        | 10 | 48×48 | 0.0001 | 2.0 | Very stable |
        | 30 | 48×48 | 0.00005 | 2.0 | Stable |
        | 100 | 48×48 | 0.00002 | 1.0 | Marginal |
        | 1000 | 96×96 | 0.000005 | 0.5 | Needs upwind |
        
        ### 🔍 Debugging Checklist
        1. Check dt format shows 5 decimals (not 0.00)
        2. Enable "Show CFL Diagnostics"
        3. Start with Re=10, then increase
        4. Use "Load Stable Preset" button
        5. Enable upwind scheme
        
        ### ⚠️ Common Pitfalls
        - dt input shows 0.00 (actually 0.00000) → use format="%.5f"
        - Lid starts at full speed → causes shock → use ramp
        - Corner BCs not handled → vorticity spikes
        - CFL limit exceeded → exponential growth
        """)