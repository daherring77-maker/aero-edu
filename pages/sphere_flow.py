import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Potential Flow Past a Sphere",
    page_icon="🌐",
    layout="wide"
)

# Title and description
st.title("🌐 Potential Flow Past a Sphere")
st.markdown("""
This app demonstrates the potential flow theory for an inviscid, incompressible fluid flowing past a sphere.
The theory assumes irrotational flow and uses the method of images with a doublet to satisfy the boundary condition.
""")

# Theory section
with st.expander("📚 Theoretical Background"):
    st.markdown("""
    ### Potential Flow Theory for a Sphere
    
    For an inviscid, incompressible fluid, the velocity potential φ satisfies Laplace's equation:
    
    $$ \\nabla^2 \\phi = 0 $$
    
    The solution for flow past a sphere of radius R with freestream velocity U∞ is:
    
    $$ \\phi = U_\\infty r \\cos\\theta \\left(1 + \\frac{R^3}{2r^3}\\right) $$
    
    The velocity components are:
    
    $$ v_r = \\frac{\\partial \\phi}{\\partial r} = U_\\infty \\cos\\theta \\left(1 - \\frac{R^3}{r^3}\\right) $$
    
    $$ v_\\theta = \\frac{1}{r} \\frac{\\partial \\phi}{\\partial \\theta} = -U_\\infty \\sin\\theta \\left(1 + \\frac{R^3}{2r^3}\\right) $$
    
    The pressure coefficient is given by Bernoulli's equation:
    
    $$ C_p = \\frac{p - p_\\infty}{\\frac{1}{2}\\rho U_\\infty^2} = 1 - \\left(\\frac{v}{U_\\infty}\\right)^2 $$
    """)

# Parameters
st.sidebar.header("Parameters")
U_inf = st.sidebar.slider("Freestream Velocity (U∞)", 1.0, 10.0, 2.0, 0.1)
R = st.sidebar.slider("Sphere Radius (R)", 0.5, 3.0, 1.0, 0.1)
grid_size = st.sidebar.slider("Grid Size", 50, 200, 100, 10)
x_range = st.sidebar.slider("X-axis Range", 2.0, 10.0, 4.0, 0.5)
y_range = st.sidebar.slider("Y-axis Range", 2.0, 10.0, 4.0, 0.5)

# Create grid
x = np.linspace(-x_range, x_range, grid_size)
y = np.linspace(-y_range, y_range, grid_size)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Calculate velocity potential and components
mask = r > R  # Only calculate outside the sphere

phi = np.zeros_like(X)
vr = np.zeros_like(X)
vtheta = np.zeros_like(X)

phi[mask] = U_inf * r[mask] * np.cos(theta[mask]) * (1 + R**3 / (2 * r[mask]**3))
vr[mask] = U_inf * np.cos(theta[mask]) * (1 - R**3 / r[mask]**3)
vtheta[mask] = -U_inf * np.sin(theta[mask]) * (1 + R**3 / (2 * r[mask]**3))

# Convert to Cartesian coordinates
vx = vr * np.cos(theta) - vtheta * np.sin(theta)
vy = vr * np.sin(theta) + vtheta * np.cos(theta)

# Calculate pressure coefficient
Cp = np.zeros_like(X)
Cp[mask] = 1 - (vr[mask]**2 + vtheta[mask]**2) / U_inf**2

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Streamlines", "Velocity Field", "Pressure Distribution", "Theory Comparison"])

with tab1:
    st.header("Streamlines and Velocity Potential")
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Streamlines
    stream = ax1.streamplot(X, Y, vx, vy, density=2, color=np.sqrt(vx**2 + vy**2), 
                           cmap='viridis', linewidth=1)
    circle = Circle((0, 0), R, color='red', alpha=0.7)
    ax1.add_patch(circle)
    ax1.set_xlim(-x_range, x_range)
    ax1.set_ylim(-y_range, y_range)
    ax1.set_title('Streamlines')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig1.colorbar(stream.lines, ax=ax1, label='Velocity Magnitude')
    
    # Velocity potential contours
    contour = ax2.contourf(X, Y, phi, 50, cmap='coolwarm')
    circle = Circle((0, 0), R, color='red', alpha=0.7)
    ax2.add_patch(circle)
    ax2.set_xlim(-x_range, x_range)
    ax2.set_ylim(-y_range, y_range)
    ax2.set_title('Velocity Potential')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig1.colorbar(contour, ax=ax2, label='Potential')
    
    st.pyplot(fig1)

with tab2:
    st.header("Velocity Field Components")
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Radial velocity
    im1 = ax1.contourf(X, Y, vr, 50, cmap='RdBu_r')
    circle = Circle((0, 0), R, color='black', alpha=0.7)
    ax1.add_patch(circle)
    ax1.set_xlim(-x_range, x_range)
    ax1.set_ylim(-y_range, y_range)
    ax1.set_title('Radial Velocity (v_r)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig2.colorbar(im1, ax=ax1, label='v_r')
    
    # Tangential velocity
    im2 = ax2.contourf(X, Y, vtheta, 50, cmap='RdBu_r')
    circle = Circle((0, 0), R, color='black', alpha=0.7)
    ax2.add_patch(circle)
    ax2.set_xlim(-x_range, x_range)
    ax2.set_ylim(-y_range, y_range)
    ax2.set_title('Tangential Velocity (v_θ)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig2.colorbar(im2, ax=ax2, label='v_θ')
    
    st.pyplot(fig2)

with tab3:
    st.header("Pressure Distribution")
    
    fig3, ax = plt.subplots(figsize=(8, 6))
    
    # Pressure coefficient
    im = ax.contourf(X, Y, Cp, 50, cmap='RdBu_r')
    circle = Circle((0, 0), R, color='black', alpha=0.7)
    ax.add_patch(circle)
    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(-y_range, y_range)
    ax.set_title('Pressure Coefficient (C_p)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig3.colorbar(im, ax=ax, label='C_p')
    
    st.pyplot(fig3)

with tab4:
    st.header("Theory Comparison")
    
    # Compare with theoretical values at specific points
    st.subheader("Surface Pressure Distribution")
    
    # Calculate theoretical surface pressure
    theta_surface = np.linspace(0, 2*np.pi, 100)
    Cp_surface_theory = 1 - (9/4) * np.sin(theta_surface)**2
    
    # Extract numerical values at r = R + small epsilon
    epsilon = 0.01
    r_surface = R + epsilon
    x_surface = r_surface * np.cos(theta_surface)
    y_surface = r_surface * np.sin(theta_surface)
    
    # Interpolate numerical Cp values
    from scipy.interpolate import griddata
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = Cp.ravel()
    Cp_surface_numerical = griddata(points, values, (x_surface, y_surface), method='linear')
    
    # Plot comparison
    fig4, ax = plt.subplots(figsize=(10, 6))
    ax.plot(theta_surface, Cp_surface_theory, 'r-', label='Theoretical', linewidth=2)
    ax.plot(theta_surface, Cp_surface_numerical, 'b--', label='Numerical', linewidth=2)
    ax.set_xlabel('θ (radians)')
    ax.set_ylabel('C_p')
    ax.set_title('Surface Pressure Coefficient Comparison')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig4)
    fig4.savefig("sphere_flow.png", dpi=300, bbox_inches="tight")
    
    # Calculate and display error
    valid_mask = ~np.isnan(Cp_surface_numerical)
    if np.any(valid_mask):
        error = np.sqrt(np.mean((Cp_surface_numerical[valid_mask] - Cp_surface_theory[valid_mask])**2))
        st.metric("RMS Error", f"{error:.4f}")
        
        st.subheader("Key Observations")
        st.markdown(f"""
        - **Stagnation Points**: At θ = 0° and θ = 180°, C_p = 1 (theoretical) vs {Cp_surface_numerical[0]:.3f} and {Cp_surface_numerical[50]:.3f} (numerical)
        - **Maximum Velocity**: At θ = 90°, C_p = -1.25 (theoretical) vs {Cp_surface_numerical[25]:.3f} (numerical)
        - **Symmetry**: The flow is perfectly symmetric about the x-axis
        - **No Drag**: Potential flow theory predicts zero drag (d'Alembert's paradox)
        """)
    else:
        st.warning("Numerical values could not be calculated near the surface. Try increasing the grid size.")

# Additional information
st.sidebar.header("About")
st.sidebar.info("""
This simulation demonstrates potential flow theory, which is an idealized model
for inviscid, incompressible fluids. Real fluids exhibit viscous effects that
significantly alter the flow pattern, especially near the sphere surface.
""")

# Footer
st.markdown("---")
st.caption("Potential Flow Theory Demonstration | Created with Streamlit")
