import streamlit as st
import numpy as np
import plotly.graph_objects as go
import neuralfoil as nf
from modules.naca_airfoil import naca4_coords

# ─────────────────────────────────────────────────────────────
# 2. Streamlit UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")
st.title("NeuralFoil + Streamlit MWE (Cartesian)")

col1, col2 = st.columns([1, 2])

with col1:
    naca = st.text_input("NACA 4-digit", "2412")
    alpha = st.slider("Angle of attack (°)", -10.0, 15.0, 4.0)
    Re = st.number_input("Reynolds number", 10000, 10000000, 1000000, step=100000)
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    m, p, t = int(naca[0])/100, int(naca[1])/10, int(naca[2:])/100
    coords = naca4_coords(m, p, t, N=150)  # Cartesian (N,2)

    # ─────────────────────────────────────────────────────────
    # 3. NeuralFoil call
    # ─────────────────────────────────────────────────────────
    aero = nf.get_aero_from_coordinates(
        coordinates=coords,
        alpha=alpha,
        Re=float(Re),
        model_size="xlarge"  # highest accuracy
    )
    #alphas = np.linspace(-5, 5, 11)
    #CLs = [float(nf.get_aero_from_coordinates(coords, alpha=a, Re=1e6)['CL']) for a in alphas]
    #slope = (CLs[-1] - CLs[0]) / (alphas[-1] - alphas[0])  # per degree
    #st.write(f"dCₗ/dα ≈ {slope:.3f}/deg (theory: ~0.11/deg)")
    with col2:
        st.subheader("Coefficients")
        # Safely extract scalars (works whether output is float or 1-element array)
        CL_val = float(np.atleast_1d(aero['CL'])[0])
        CD_val = float(np.atleast_1d(aero['CD'])[0])
        CM_val = float(np.atleast_1d(aero['CM'])[0])
        conf_val = float(np.atleast_1d(aero['analysis_confidence'])[0])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("C_L", f"{CL_val:.3f}")
        c2.metric("C_D", f"{CD_val:.4f}")
        c3.metric("C_M", f"{CM_val:.4f}")
        st.caption(f"Analysis confidence: {conf_val:.2f}")
        
        # ─────────────────────────────────────────────────────
        # 4. Cp Distribution - all 32 boundary layer points
        # ─────────────────────────────────────────────────────
        # Cp Distribution - all 32 boundary layer points
        st.subheader("Cp Distribution")

        # Safe extraction: works whether NeuralFoil returns a float or a 1-element array
        def _safe_float(val):
            return float(np.atleast_1d(val)[0])

        Cp_upper = [1 - _safe_float(aero[f"upper_bl_ue/vinf_{i}"])**2 for i in range(32)]
        Cp_lower = [1 - _safe_float(aero[f"lower_bl_ue/vinf_{i}"])**2 for i in range(32)]

        fig_cp = go.Figure()
        fig_cp.add_scatter(
            x=nf.bl_x_points, y=Cp_upper, 
            name="Upper", mode="lines+markers", 
            line=dict(color="royalblue", width=2),
            marker=dict(size=5, color="royalblue")
        )
        fig_cp.add_scatter(
            x=nf.bl_x_points, y=Cp_lower, 
            name="Lower", mode="lines+markers", 
            line=dict(color="crimson", width=2),
            marker=dict(size=5, color="crimson")
        )
        fig_cp.update_layout(
            yaxis=dict(autorange="reversed", title="Cₚ"),
            xaxis=dict(title="x/c", range=[0, 1]),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cp, width='stretch')

        # ─────────────────────────────────────────────────────
        # 5. Airfoil + Lift Vector
        # ─────────────────────────────────────────────────────
        st.subheader("Airfoil Geometry & Lift Direction")
        fig_af = go.Figure()
        fig_af.add_scatter(x=coords[:,0], y=coords[:,1], mode="lines", 
                           line=dict(color="white", width=2), name="Airfoil")
                      
        # Quarter-chord lift vector (scaled for visibility)
        x_qc, y_qc = 0.25, 0.0
        CL_val = float(np.atleast_1d(aero['CL'])[0])
        lift_scale = abs(CL_val) * 0.6

        # Arrow direction: positive CL = upward lift
        arrow_direction = -1 if CL_val >= 0 else 0
        arrow_color = "lime" if arrow_direction < 0 else "orange"
        text_label = f"Cₗ={CL_val:.2f}"

        fig_af.add_annotation(
            x=x_qc, y=y_qc,
            ax=x_qc, ay=y_qc + arrow_direction * lift_scale,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor=arrow_color,
            text=text_label,
            # ✅ Correct text positioning parameters:
            #textanchor="bottom" if arrow_direction > 0 else "top",  # Text above/below arrow
            xanchor="center",
            yanchor="bottom" if arrow_direction > 0 else "top",
            font=dict(size=11, color=arrow_color, family="monospace"),
            bgcolor="rgba(0,0,0,0.6)", #if is_dark_mode else "rgba(255,255,255,0.8)",
            borderpad=4,
            bordercolor=arrow_color,
            borderwidth=1,
            xref="x", yref="y", axref="x", ayref="y"
        )
        st.plotly_chart(fig_af, width='stretch')