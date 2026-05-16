# In app.py - new page for airfoil flow
import streamlit as st
import numpy as np
from modules.plotting import create_lift_visualization_plot, create_cp_distribution_plot, create_streamlines_panel_plot
from modules.panel_solver import panel_method_airfoil
from modules.naca_airfoil import generate_naca_4digit, remove_duplicate_points
from modules.aircraft_3d_demo import create_aircraft_3d_demo
from modules.plotting import create_aero_results_zip, add_plot_download_button
import io, zipfile, plotly.io as pio
import plotly.graph_objects as go
st.title("✈️ Airfoil Aerodynamics: Visualizing Lift")

st.sidebar.header("🎛️ Flow Parameters")
# Initialize
if 'results' not in st.session_state:
    st.session_state.aero_results = None
if 'params' not in st.session_state:
    st.session_state.params = None
# === PLOT STYLE CONFIGURATION ===
PLOT_CONFIG = {
    'interactive': {
        'width': None,
        'height': None,
        'font_size': 14,
        'bg_color': 'rgba(0,0,0,0)',
        'show_grid': False,
        'show_markers': True
    },
    'export': {
        'width': 800,
        'height': 600,
        'font_size': 14,
        'font_family': 'Arial',
        'bg_color': 'white',
        'show_grid': True,
        'show_markers': False,
        'scale': 2  # For retina displays
    }
}

# Airfoil selection
series = st.sidebar.text_input("NACA Code", value="2412", max_chars=4)
alpha = st.sidebar.slider("Angle of Attack (°)", -10, 20, 4, 1)
V_inf = st.sidebar.number_input("Freestream Velocity", value=1.0)

if len(series) == 4 and series.isdigit():
    # Generate geometry
    xu, yu, xl, yl = generate_naca_4digit(series, num_points=100)
    
    # ✅ CORRECT: Reverse upper surface, keep lower as-is
    x_airfoil = np.concatenate([xu[::-1], xl[1:]])  # TE→LE + LE→TE
    y_airfoil = np.concatenate([yu[::-1], yl[1:]])
    
    # Remove consecutive duplicates
    x_airfoil, y_airfoil = remove_duplicate_points(x_airfoil, y_airfoil)
    
    # ✅ Only close if not already closed
    if not (np.isclose(x_airfoil[0], x_airfoil[-1]) and 
            np.isclose(y_airfoil[0], y_airfoil[-1])):
        x_airfoil = np.append(x_airfoil, x_airfoil[0])
        y_airfoil = np.append(y_airfoil, y_airfoil[0])
        st.info("ℹ️ Closed the loop")
    else:
        st.info("ℹ️ Airfoil already closed")
    
    # Debug final geometry
    #st.write("### Final Airfoil Geometry")
    #st.write(f"Points: {len(x_airfoil)}")
    #st.write(f"Start: ({x_airfoil[0]:.4f}, {y_airfoil[0]:.4f})")
    #st.write(f"End:   ({x_airfoil[-1]:.4f}, {y_airfoil[-1]:.4f})")
    # === PANEL GEOMETRY DEBUG (using your existing variable names) ===
    #st.write("### 🔍 Panel Geometry Debug")

    # 1. NACA Generator Output
    #st.write(f"**NACA{series} Generator Output**")
    #col1, col2 = st.columns(2)
    #with col1:
    #    st.write("**Upper surface (xu, yu):**")
    #    st.write(f"  Points: {len(xu)}")
    #    st.write(f"  Start (LE?): ({xu[0]:.4f}, {yu[0]:.4f})")
    #    st.write(f"  End (TE?):   ({xu[-1]:.4f}, {yu[-1]:.4f})")
    #with col2:
    #    st.write("**Lower surface (xl, yl):**")
    #    st.write(f"  Points: {len(xl)}")
    #    st.write(f"  Start (LE?): ({xl[0]:.4f}, {yl[0]:.4f})")
    #    st.write(f"  End (TE?):   ({xl[-1]:.4f}, {yl[-1]:.4f})")

    # 2. Combined Airfoil
    #st.write(f"**Combined Airfoil (x_airfoil, y_airfoil):**")
    #st.write(f"  Total points: {len(x_airfoil)}")
    #st.write(f"  Start (should be TE): ({x_airfoil[0]:.4f}, {y_airfoil[0]:.4f})")
    #st.write(f"  End (should be TE):   ({x_airfoil[-1]:.4f}, {y_airfoil[-1]:.4f})")

    with st.expander("📖 How to Read These Plots"):
        st.subheader("1. What are Cₚ and Cₗ?")
        st.write("""
        - **Cₚ (Pressure Coefficient)** is a dimensionless measure of local pressure relative to the freestream.  
        In incompressible flow: `Cₚ = 1 - (V/V∞)²`  
        - `Cₚ < 0` → faster flow, lower pressure (**suction**)  
        - `Cₚ > 0` → slower flow, higher pressure (**positive pressure**)
        
        - **Cₗ (Lift Coefficient)** is a dimensionless measure of total upward aerodynamic force:  
        `Cₗ = L' / (½ρV∞²c)`  
        It summarizes the airfoil's lifting performance at a given angle of attack.
        """)
        
        st.subheader("2. How Cₚ and Cₗ Are Connected")
        st.write("""
        They aren't independent — **Cₗ is the integral of Cₚ over the entire airfoil surface**.  
        Every local pressure difference contributes to total lift. If you plot Cₚ vs x/c,  
        the **area between upper and lower curves** is directly proportional to Cₗ.
        """)
        
        st.subheader("3. Where Lift Actually Comes From")
        st.write("""
        Lift is the **net result of pressure differences across the entire chord**:
        - **Upper surface**: typically suction (Cₚ < 0) → "pulls" wing upward
        - **Lower surface**: typically higher pressure (Cₚ ≥ 0) → "pushes" wing upward
        
        Both contribute! The exact split depends on angle of attack and camber.
    """)

    # 3. Panel Ordering Check (counter-clockwise?)
    n_panels = len(x_airfoil) - 1
    if n_panels > 10:
        st.write("**Panel Ordering Check (counter-clockwise?):**")
        idx_upper = n_panels // 4   # Should be upper surface (y > 0)
        idx_lower = 3 * n_panels // 4  # Should be lower surface (y < 0)
        
        st.write(f"  Point at n/4 (should be upper, y>0): ({x_airfoil[idx_upper]:.4f}, {y_airfoil[idx_upper]:.4f})")
        st.write(f"  Point at 3n/4 (should be lower, y<0): ({x_airfoil[idx_lower]:.4f}, {y_airfoil[idx_lower]:.4f})")
        
        if y_airfoil[idx_upper] < 0:
            st.error("❌ Panel ordering may be CLOCKWISE! Upper surface has y < 0")
        elif y_airfoil[idx_lower] > 0:
            st.error("❌ Panel ordering may be CLOCKWISE! Lower surface has y > 0")
        else:
            st.success("✅ Panel ordering looks counter-clockwise")

    # 4. Panel Lengths
    dx_panels = np.diff(x_airfoil)
    dy_panels = np.diff(y_airfoil)
    ds_panels = np.sqrt(dx_panels**2 + dy_panels**2)

    st.write("**Panel Lengths:**")
    st.write(f"  Min: {np.min(ds_panels):.6f}")
    st.write(f"  Max: {np.max(ds_panels):.6f}")
    st.write(f"  Mean: {np.mean(ds_panels):.6f}")

    if np.any(ds_panels < 1e-6):
        zero_idx = np.where(ds_panels < 1e-6)[0]
        st.error(f"❌ Zero-length panels at indices: {zero_idx[:5]}")

    # 5. Normal Vectors (for lift sign check)
    theta_panels = np.arctan2(dy_panels, dx_panels)
    nx_panels = np.sin(theta_panels)
    ny_panels = -np.cos(theta_panels)

    st.write("**Normal Vectors (should point OUTWARD):**")
    st.write(f"  Top surface (n/4): nx={nx_panels[idx_upper]:.3f}, ny={ny_panels[idx_upper]:.3f}")
    st.write(f"  Bottom surface (3n/4): nx={nx_panels[idx_lower]:.3f}, ny={ny_panels[idx_lower]:.3f}")

    if ny_panels[idx_upper] < 0:
        st.warning("⚠️ Top surface normal points DOWN (inward?) — may flip lift sign")
    if ny_panels[idx_lower] > 0:
        st.warning("⚠️ Bottom surface normal points UP (inward?) — may flip lift sign")

    # 6. Freestream Components
    alpha_rad = np.radians(alpha)
    u_inf = 1.0 * np.cos(alpha_rad)  # V_inf = 1.0 default
    v_inf = 1.0 * np.sin(alpha_rad)

    st.write("**Freestream Components (V_inf=1.0):**")
    st.write(f"  alpha = {alpha}° → {alpha_rad:.4f} rad")
    st.write(f"  u_inf = {u_inf:.4f}, v_inf = {v_inf:.4f}")

    if alpha > 0 and v_inf < 0:
        st.error("❌ v_inf is negative for positive alpha! Check np.radians() sign")

    # 7. Expected vs. Actual Lift Sign
    st.write("**Lift Sign Expectation:**")
    if alpha > 0:
        st.write("  For α > 0: Cl should be POSITIVE (upward lift)")
    else:
        st.write("  For α < 0: Cl should be NEGATIVE (downward lift)")
    st.info("💡 If Cl sign is wrong, check: panel ordering, normal vectors, or dy calculation")
    # Check panel lengths
    ds = np.sqrt(np.diff(x_airfoil)**2 + np.diff(y_airfoil)**2)
    st.write(f"Min panel: {np.min(ds):.6f}, Max panel: {np.max(ds):.6f}")
    st.write(f"Zero-length panels: {np.sum(ds < 1e-4)}")
    
    if np.min(ds) < 1e-6:
        st.error("❌ Zero-length panels detected!")
        # Show where they are
        zero_idx = np.where(ds < 1e-6)[0]
        st.write(f"Zero panels at indices: {zero_idx}")
        for idx in zero_idx[:5]:  # Show first 5
            st.write(f"  Panel {idx}: ({x_airfoil[idx]:.6f}, {y_airfoil[idx]:.6f}) → ({x_airfoil[idx+1]:.6f}, {y_airfoil[idx+1]:.6f})")
    else:
        # Solve
        try:
            # 1. Reverse the Geometry and Cp arrays
            # [::-1] reverses the array along the first axis
            x_rev = x_airfoil[::-1]
            y_rev = y_airfoil[::-1]
            results = panel_method_airfoil(x_rev, y_rev, alpha, V_inf)
            st.success(f"✅ Cl = {results['Cl']:.3f}")
        except Exception as e:
            st.error(f"❌ {e}")
  
        with st.expander("📖 How to Read These Plots"):
            st.subheader("1. What are Cₚ and Cₗ?")
            st.write("""
            - **Cₚ (Pressure Coefficient)** is a dimensionless measure of local pressure relative to the freestream.  
            In incompressible flow: `Cₚ = 1 - (V/V∞)²`  
            - `Cₚ < 0` → faster flow, lower pressure (**suction**)  
            - `Cₚ > 0` → slower flow, higher pressure (**positive pressure**)
            
            - **Cₗ (Lift Coefficient)** is a dimensionless measure of total upward aerodynamic force:  
            `Cₗ = L' / (½ρV∞²c)`  
            It summarizes the airfoil's lifting performance at a given angle of attack.
            """)
            
            st.subheader("2. How Cₚ and Cₗ Are Connected")
            st.write("""
            They aren't independent — **Cₗ is the integral of Cₚ over the entire airfoil surface**.  
            Every local pressure difference contributes to total lift. If you plot Cₚ vs x/c,  
            the **area between upper and lower curves** is directly proportional to Cₗ.
            """)
            
            st.subheader("3. Where Lift Actually Comes From")
            st.write("""
            Lift is the **net result of pressure differences across the entire chord**:
            - **Upper surface**: typically suction (Cₚ < 0) → "pulls" wing upward
            - **Lower surface**: typically higher pressure (Cₚ ≥ 0) → "pushes" wing upward
            
            Both contribute! The exact split depends on angle of attack and camber.
        """)
        Cp_rev = results['Cp'][::-1]    
        # Create figure once
        fig_cp = create_cp_distribution_plot(
        results['x_cp'], 
        results['y_cp'], 
        Cp_rev,
        title=f"Cp Distribution (Cl={results['Cl']:.3f})",
        #show_markers=False,  # Cleaner for publication
        for_export=False  # Export mode
        )
        st.plotly_chart(fig_cp, width='stretch')
        
        st.metric("Lift Coefficient", f"{results['Cl']:.3f}")
        st.metric("Max Suction", f"{np.min(results['Cp']):.3f}")
        st.metric("Max Pressure", f"{np.max(results['Cp']):.3f}")
        # Lift visualization
        fig_lift = create_lift_visualization_plot(      
        x_rev, y_rev,
        Cp_rev, 
        results['Cl'], 
        alpha,
        V_inf=V_inf,
        title=f"Lift Visualization (NACA{series}, α={alpha}°)",
        for_export=False,  # Interactive mode
        show_pressure_arrows=True,
        arrow_scale=0.06  # Slightly larger arrows for visibility
        )
        st.plotly_chart(fig_lift, width='stretch')
        fig_lift.write_image("lift_viz.png")

        fig_cp = go.Figure()
        fig_cp.add_scatter(x=results['x_cp'], y=results['Cp'], mode="lines+markers", 
                       line=dict(color="steelblue", width=2),
                       marker=dict(size=4, color="steelblue"),
                       name="Panel Solver")
        fig_cp.update_layout(
             yaxis=dict(autorange="reversed", title="Cₚ", zeroline=True),
            xaxis=dict(title="x/c", range=[0, 1]),
            title="Cp vs x/c (Compare with NeuralFoil)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cp, width='stretch')
        
        # Streamlines
        fig_stream = create_streamlines_panel_plot(
        x_airfoil, y_airfoil, alpha,
        V_inf=1.0,
        n_streamlines=20,
        title=f"Streamlines (NACA{series}, α={alpha}°)",
        for_export=False  # Interactive mode
    )
        st.plotly_chart(fig_stream, width='stretch')

# === ZIP DOWNLOAD BUTTON ===
        # In your Streamlit app, after panel method solves:
        if st.session_state.aero_results is not None:
            results = st.session_state.aero_results
            x_airfoil = st.session_state.x_airfoil
            y_airfoil = st.session_state.y_airfoil
    
        # === ZIP Download Button ===
        zip_data = create_aero_results_zip(results, x_airfoil, y_airfoil)
        
        # ✅ Clean filename using alpha (not Re!)
        series = results.get('series', 'XXXX')
        alpha = results.get('alpha', 0)
        
        st.download_button(
            label="📦 Download All Results (ZIP)",
            data=zip_data,
            file_name=f"NACA{series}_alpha{int(alpha)}_results.zip",
            mime="application/zip"
        )
                  

# Streamlit controls for pitch/yaw/roll
st.sidebar.subheader("🎮 Aircraft Attitude")
pitch = st.sidebar.slider("Pitch (°)", -30, 30, 0, 1)
yaw = st.sidebar.slider("Yaw (°)", -30, 30, 0, 1)
roll = st.sidebar.slider("Roll (°)", -45, 45, 0, 1)

fig_3d = create_aircraft_3d_demo(pitch, yaw, roll)
st.plotly_chart(fig_3d, width='stretch')

st.markdown("""
### Understanding Aircraft Motions

| Motion | Axis | Effect on Flight |
|--------|------|-----------------|
| **Pitch** 📈 | Lateral (Y) | Nose up/down → climb/descend |
| **Yaw** 🔄 | Vertical (Z) | Nose left/right → turn coordination |
| **Roll** 🌀 | Longitudinal (X) | Wing up/down → banked turn |

**In 2D airfoil analysis**, we only see **pitch** (angle of attack).  
**Full 3D flight** requires all three motions working together!
""")  


 