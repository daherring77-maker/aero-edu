import streamlit as st

# ----------------------------
# Main App
# ----------------------------
pages = {
        "Flow Dynamics":[   
           st.Page("pages/navier_stokes_theory.py", title="Navier Stokes Theory", icon="📊"),
           st.Page("pages/navier_stokes.py", title="Navier Stokes", icon="📊"),
           st.Page("pages/ghia_validation.py", title="Ghia Benchmark Validation", icon="📊"),
           st.Page("pages/sphere_flow.py", title="Sphere Flow", icon="📊"),
           st.Page("pages/naca_aerofoil.py", title="NACA Aerofoil", icon="📊"),   
           st.Page("pages/visualize_lift.py", title="Visualize Lift", icon="📊" ),  
           st.Page("pages/neural_foil.py", title="Neural Foil", icon="📊" ),
           st.Page("pages/panel_neural.py", title="Panel Method vs NeuralFoil", icon="📊" )
        
       
        ]
}
pg = st.navigation(pages)
pg.run()