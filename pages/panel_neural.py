import streamlit as st
import numpy as np
import plotly.graph_objects as go
from modules.panel_solver import panel_method_airfoil 
from modules.naca_airfoil import generate_naca_4digit, remove_duplicate_points, naca4_coords, naca4_coords_neuralfoil

# ─────────────────────────────────────────────────────────────
# 1. Common Inputs (run once)
# ─────────────────────────────────────────────────────────────
st.title("AeroEdu: Panel Method vs NeuralFoil")

naca = st.text_input("NACA 4-digit", "2412")
alpha = st.slider("Angle of attack (°)", -10.0, 15.0, 4.0)
Re = st.number_input("Reynolds number", 10000, 10000000, 400000, step=100000)
V_inf = 1.0

if st.button("Run Both Solvers"):
    # Generate coordinates once
    if len(naca) == 4 and naca.isdigit():
        # Generate geometry
        xu, yu, xl, yl = generate_naca_4digit(naca, num_points=100)
        #coords = generate_naca_4digit(naca, num_points=100)
    x_airfoil = np.concatenate([xu[::-1], xl[1:]])  # TE→LE + LE→TE
    y_airfoil = np.concatenate([yu[::-1], yl[1:]])
    
    # Remove consecutive duplicates
    x_airfoil, y_airfoil = remove_duplicate_points(x_airfoil, y_airfoil)
    # 1. Reverse the Geometry and Cp arrays
    # [::-1] reverses the array along the first axis
    x_rev = x_airfoil[::-1]
    y_rev = y_airfoil[::-1]
    # ─────────────────────────────────────────────────────────
    # 2. Run Both Solvers
    # ─────────────────────────────────────────────────────────
    with st.spinner("Running Panel Method..."):
        results_panel = panel_method_airfoil(x_rev, y_rev, alpha, V_inf)
    
    with st.spinner("Running NeuralFoil..."):
        import neuralfoil as nf
        m, p, t = int(naca[0])/100, int(naca[1])/10, int(naca[2:])/100
        coords = naca4_coords_neuralfoil(m, p, t, N=150)  # Cartesian (N,2)
        results_nf = nf.get_aero_from_coordinates(coords, alpha=alpha, Re=Re, model_size="xlarge"  # highest accuracy
    )
    
    # ─────────────────────────────────────────────────────────
    # 3. Display in Tabs
    # ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Side-by-Side Cp", "🔵 Panel Method", "🟢 NeuralFoil"])
    
    with tab1:
        st.subheader("Cp Distribution Comparison")
        with st.expander("🔍 Why Do Panel Method and NeuralFoil Differ?"):
            st.write("""
            **Short answer**: One is inviscid theory, the other includes viscous reality.
            
            | Aspect | Panel Method (Inviscid) | NeuralFoil (Viscous) |
            |--------|---------------------------|----------------------|
            | **Physics** | Potential flow, no friction | Boundary layer + transition + separation |
            | **Cₚ Peaks** | Sharper, more extreme | Smoothed by boundary layer displacement |
            | **Drag** | C_D ≈ 0 (d'Alembert's paradox) | Realistic profile + pressure drag |
            | **Stall** | Never stalls (linear forever) | Predicts separation & lift drop-off |
            | **Re Dependence** | None | Strong (thicker BL at low Re) |
            
            **Why this matters for learning**:  
            Seeing both side-by-side shows *exactly where viscosity matters*.  
            At low α, they agree on C_L (lift is mostly inviscid).  
            Near stall or high drag, viscosity dominates — and NeuralFoil captures it.
            """)
        fig_compare = go.Figure()
        # Panel method (use your reversed arrays if needed)
        fig_compare.add_scatter(x=results_panel['x_cp'], y=results_panel['Cp'], 
                                mode="lines", name="Panel (Inviscid)", 
                                line=dict(color="royalblue", dash="dot"))
        # NeuralFoil (interpolate 32 points)
        def _safe_scalar(val):
            return float(np.atleast_1d(val)[0])
            
        Cp_nf_upper = [1 - _safe_scalar(results_nf[f"upper_bl_ue/vinf_{i}"])**2 for i in range(32)]
        Cp_nf_lower = [1 - _safe_scalar(results_nf[f"lower_bl_ue/vinf_{i}"])**2 for i in range(32)]
        fig_compare.add_scatter(x=nf.bl_x_points, y=Cp_nf_upper, 
                                mode="lines", name="NeuralFoil Upper (Viscous)", 
                                line=dict(color="red"))
        fig_compare.add_scatter(x=nf.bl_x_points, y=Cp_nf_lower, 
                                mode="lines", name="NeuralFoil Lower (Viscous)", 
                                line=dict(color="orange"))
        fig_compare.update_layout(yaxis=dict(autorange="reversed", title="Cₚ"), 
                                  xaxis=dict(title="x/c", range=[0, 1]))
        st.plotly_chart(fig_compare, width='stretch')
        fig_compare.write_image("Cp Distribution.png")
        # Metrics comparison
        c1, c2, c3 = st.columns(3)
        c1.metric("C_L Panel", f"{results_panel['Cl']:.3f}")
        c1.metric("C_L NeuralFoil", f"{float(np.atleast_1d(results_nf['CL'])[0]):.3f}")
        c2.metric("ΔC_L", f"{abs(results_panel['Cl'] - float(np.atleast_1d(results_nf['CL'])[0])):.3f}")
    
    with tab2:
        st.subheader("Panel Method Details")
        fig_panel_geo = go.Figure()
        fig_panel_geo.add_scatter(x=results_panel['x_cp'], y=results_panel['Cp'], mode="lines+markers", 
                       line=dict(color="steelblue", width=2),
                       marker=dict(size=4, color="steelblue"),
                       name="Panel Solver")
        fig_panel_geo.update_layout(
             yaxis=dict(autorange="reversed", title="Cₚ", zeroline=True),
            xaxis=dict(title="x/c", range=[0, 1]),
            title="Cp vs x/c (Compare with NeuralFoil)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_panel_geo, width='stretch')
    
    with tab3:
        st.subheader("NeuralFoil Details")
        # Safe extraction: works whether NeuralFoil returns a float or a 1-element array
        def _safe_float(val):
            return float(np.atleast_1d(val)[0])

        Cp_upper = [1 - _safe_float(results_nf[f"upper_bl_ue/vinf_{i}"])**2 for i in range(32)]
        Cp_lower = [1 - _safe_float(results_nf[f"lower_bl_ue/vinf_{i}"])**2 for i in range(32)]

        fig_nf_geo = go.Figure()
        fig_nf_geo.add_scatter(
            x=nf.bl_x_points, y=Cp_upper, 
            name="Upper", mode="lines+markers", 
            line=dict(color="royalblue", width=2),
            marker=dict(size=5, color="royalblue")
        )
        fig_nf_geo.add_scatter(
            x=nf.bl_x_points, y=Cp_lower, 
            name="Lower", mode="lines+markers", 
            line=dict(color="crimson", width=2),
            marker=dict(size=5, color="crimson")
        )
        fig_nf_geo.update_layout(
            yaxis=dict(autorange="reversed", title="Cₚ"),
            xaxis=dict(title="x/c", range=[0, 1]),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_nf_geo, width='stretch')