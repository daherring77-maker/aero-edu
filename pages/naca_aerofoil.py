import streamlit as st
from modules.naca_airfoil import generate_naca_4digit, visualize_airfoil, get_airfoil_properties

# -----------------------------------------------------------------------------
# NACA Airfoil Page
# -----------------------------------------------------------------------------
st.title("✈️ NACA Airfoil Geometry Generator")
st.markdown("*Pure Python • No external meshers • Educational*")

st.sidebar.header("🎛️ Airfoil Parameters")

# Airfoil selection
series = st.sidebar.text_input("NACA 4-Digit Code", value="2412", max_chars=4)

# Validate input
if len(series) == 4 and series.isdigit():
    num_points = st.sidebar.slider("Surface Points", 50, 500, 200, 10)
    closed_te = st.sidebar.checkbox("Closed Trailing Edge", value=True)
    
    # Generate and display
    xu, yu, xl, yl = generate_naca_4digit(series, num_points, closed_te)
    
    # Properties panel
    with st.expander("📊 Airfoil Properties", expanded=True):
        props = get_airfoil_properties(series)
        col1, col2, col3 = st.columns(3)
        col1.metric("Max Camber", props['Max Camber'])
        col2.metric("Camber Location", props['Camber Location'])
        col3.metric("Max Thickness", props['Max Thickness'])
        st.info(f"**Typical Use:** {props['Typical Use']}")
    
    # Visualization
    fig = visualize_airfoil(series, num_points)
    st.plotly_chart(fig, width='stretch')
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        # PNG export
        import plotly.io as pio
        img_bytes = pio.to_image(fig, format="png", width=800, height=600, scale=2)
        st.download_button(
            label="📥 Download PNG",
            data=img_bytes,
            file_name=f"NACA_{series}_airfoil.png",
            mime="image/png"
        )
    
    with col2:
        # CSV export of coordinates
        import pandas as pd
        df_upper = pd.DataFrame({'x': xu, 'y': yu, 'surface': 'upper'})
        df_lower = pd.DataFrame({'x': xl, 'y': yl, 'surface': 'lower'})
        df = pd.concat([df_upper, df_lower], ignore_index=True)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Coordinates (CSV)",
            data=csv,
            file_name=f"NACA_{series}_coordinates.csv",
            mime="text/csv"
        )
    
    # Educational notes
    with st.expander("📚 Educational Notes"):
        st.markdown("""
        ### NACA 4-Digit Series Explained
        
        **Example: NACA 2412**
        - **2**: Maximum camber = 2% of chord
        - **4**: Location of max camber = 40% of chord
        - **12**: Maximum thickness = 12% of chord
        
        ### Why Cosine Spacing?
        Points are clustered near the **leading edge** where curvature is highest.
        This gives better resolution for CFD without increasing total point count.
        
        ### Trailing Edge Treatment
        - **Closed TE**: Standard for viscous flow (boundary layers meet)
        - **Open TE**: Theoretical inviscid flow (sharp edge)
        
        ### Next Steps
        1. Export coordinates to CSV
        2. Import into your CFD solver
        3. Generate mesh around airfoil (O-grid or C-grid)
        4. Apply boundary conditions
        5. Compute lift and drag!
        """)
    
    # Common airfoils quick-select
    st.subheader("🔍 Common NACA Airfoils")
    preset = st.selectbox(
        "Quick Load",
        ['0012 (Symmetric)', '2412 (General Aviation)', '4415 (High Lift)', 
            '0006 (Thin)', '23012 (Laminar)'],
        index=1
    )
    
    if st.button("Load Preset"):
        series_map = {
            '0012 (Symmetric)': '0012',
            '2412 (General Aviation)': '2412',
            '4415 (High Lift)': '4415',
            '0006 (Thin)': '0006',
            '23012 (Laminar)': '23012'  # Note: 5-digit, would need extension
        }
        st.session_state['naca_series'] = series_map[preset][:4]
        st.rerun()

else:
    st.error("❌ Please enter a valid 4-digit NACA code (e.g., '2412', '0012')")
    st.info("💡 **Tip:**" 
    "- First digit = camber (0 for symmetric)"
    "- Second digit = camber location (tenths of chord)"
    "- Last two digits = thickness (% of chord)")

