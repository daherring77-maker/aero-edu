# In your ghia_validation.py page
import streamlit as st
import numpy as np
from modules.ghia_data import get_ghia_data
from modules.plotting import plot_ghia_comparison
from modules.utils import run_cavity_flow

st.title("🔬 Ghia Benchmark Validation")
st.markdown("*Validate cavity flow solver against Ghia et al. (1982)*")

# Sidebar parameters
st.sidebar.header("🎛️ Validation Parameters")
Re = st.sidebar.selectbox("Reynolds Number", [100, 400, 1000], index=0)
nx = st.sidebar.slider("Grid X", 32, 128, 64)
ny = st.sidebar.slider("Grid Y", 32, 128, 64)
dt = st.sidebar.number_input("Time Step", value=0.001, format="%.4f")
t_end = st.sidebar.number_input("End Time", value=5.0)

# Run button
if st.sidebar.button("▶️ Run Validation"):
    with st.spinner(f"Solving Re={Re}..."):
        # Run simulation
        x, y, u, v, psi, omega, t_hist, ke_hist, vort_hist = run_cavity_flow(
            nx, ny, Re, dt, t_end
        )
        
        # Extract centerline profiles
        i_center = nx // 2
        j_center = ny // 2
        u_vertical = u[:, i_center]
        v_horizontal = v[j_center, :]
        
        # Get Ghia reference data
        ghia_ref = get_ghia_data(Re)
        
        # Plot comparison
        fig = plot_ghia_comparison(y, u_vertical, x, v_horizontal, Re, ghia_data=ghia_ref)
        st.plotly_chart(fig, width='stretch')
        fig.write_image("ghia_comparison.png")
        # Compute error metrics
        if ghia_ref is not None:
            # Interpolate Ghia data to simulation grid for error calculation
            from scipy.interpolate import interp1d
            u_ghia_interp = interp1d(ghia_ref['y'], ghia_ref['u'], kind='linear', fill_value='extrapolate')
            v_ghia_interp = interp1d(ghia_ref['x'], ghia_ref['v'], kind='linear', fill_value='extrapolate')
            
            u_ghia_on_grid = u_ghia_interp(y)
            v_ghia_on_grid = v_ghia_interp(x)
            
            u_rms = np.sqrt(np.mean((u_vertical - u_ghia_on_grid)**2))
            v_rms = np.sqrt(np.mean((v_horizontal - v_ghia_on_grid)**2))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Reynolds Number", Re)
            col2.metric("u-velocity RMS Error", f"{u_rms:.4f}")
            col3.metric("v-velocity RMS Error", f"{v_rms:.4f}")
            
            # Pass/fail criteria
            if u_rms < 0.02 and v_rms < 0.02:
                st.success("✅ **VALIDATION PASSED!** Errors within acceptable range.")
            else:
                st.warning("⚠️ **Validation Warning:** Errors higher than expected. Try:")
                st.write("- Increase grid resolution")
                st.write("- Reduce dt (CFL condition)")
                st.write("- Increase t_end (ensure steady state)")


