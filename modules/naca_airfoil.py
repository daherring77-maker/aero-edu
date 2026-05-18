"""
NACA 4-Digit Airfoil Generator
Pure Python implementation for educational CFD
"""

import numpy as np
import plotly.graph_objects as go

def generate_naca_4digit(series, num_points=500, closed_trailing_edge=True):
    """
    Generate coordinates for NACA 4-digit airfoil.
    
    Parameters
    ----------
    series : str
        4-digit NACA code (e.g., '2412', '0012', '4415')
        - 1st digit: Maximum camber (% of chord)
        - 2nd digit: Location of maximum camber (tenths of chord)
        - 3rd-4th digits: Maximum thickness (% of chord)
    num_points : int
        Number of points along the airfoil surface
    closed_trailing_edge : bool
        If True, forces trailing edge to close (standard for CFD)
    
    Returns
    -------
    x_upper, y_upper, x_lower, y_lower : ndarray
        Coordinates of upper and lower surfaces
    """
    # Parse NACA digits
    m = int(series[0]) / 100.0      # Max camber
    p = int(series[1]) / 10.0       # Location of max camber
    t = int(series[2:]) / 100.0     # Max thickness
    
    # Generate x coordinates (cosine spacing for better resolution at leading edge)
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))  # Clusters points near leading/trailing edges
    
    # Thickness distribution (standard NACA formula)
    yt = 5 * t * (0.2969 * np.sqrt(x) 
                  - 0.1260 * x 
                  - 0.3516 * x**2 
                  + 0.2843 * x**3 
                  - 0.1015 * x**4)
    
    # Close trailing edge if requested
    if closed_trailing_edge:
        yt[-1] = 0.0  # Force zero thickness at TE
    
    # Camber line and its derivative
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    
    if m > 0 :  # Only compute if cambered (not symmetric)
        for i, xi in enumerate(x):
            if 0 <= xi < p:
                yc[i] = m / p**2 * (2 * p * xi - xi**2)
                dyc[i] = 2 * m / p**2 * (p - xi)
            elif p <= xi <= 1:
                yc[i] = m / (1 - p)**2 * ((1 - 2*p) + 2*p*xi - xi**2)
                dyc[i] = 2 * m / (1 - p)**2 * (p - xi)
    
    # Angle of camber line
    theta = np.arctan(dyc)
    
    # Upper and lower surfaces (perpendicular to camber line)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    return xu, yu, xl, yl

def visualize_airfoil(series='2412', num_points=200):
    """
    Create a Plotly figure of the NACA airfoil.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    xu, yu, xl, yl = generate_naca_4digit(series, num_points)
    
    fig = go.Figure()
    
    # Upper surface
    fig.add_trace(go.Scatter(
        x=xu, y=yu, 
        mode='lines', 
        name='Upper Surface',
        line=dict(color='blue', width=3),
        fill='tonextx'
    ))
    
    # Lower surface
    fig.add_trace(go.Scatter(
        x=xl, y=yl, 
        mode='lines', 
        name='Lower Surface',
        line=dict(color='blue', width=3)
    ))
    
    # Chord line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], 
        mode='lines', 
        name='Chord Line',
        line=dict(color='gray', width=1, dash='dot')
    ))
    
    # Leading/trailing edge markers
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], 
        mode='markers', 
        name='Edges',
        marker=dict(symbol='circle', color='red', size=8)
    ))
    
    fig.update_layout(
        title=f"NACA {series} Airfoil Geometry",
        xaxis_title="x/c (Chord)",
        yaxis_title="y/c",
        width=700,
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Equal aspect ratio (critical for airfoil shape!)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=[-0.05, 1.05])
    fig.update_yaxes(range=[-0.15, 0.15])
    
    return fig

def get_airfoil_properties(series):
    """
    Return a dictionary of airfoil properties for display.
    """
    m = int(series[0]) / 100.0
    p = int(series[1]) / 10.0
    t = int(series[2:]) / 100.0
    
    return {
        'Max Camber': f"{m*100:.0f}% of chord",
        'Camber Location': f"{p*100:.0f}% of chord",
        'Max Thickness': f"{t*100:.0f}% of chord",
        'Symmetric': 'Yes' if m == 0 else 'No',
        'Typical Use': get_airfoil_use_case(series)
    }

def get_airfoil_use_case(series):
    """Educational: typical applications for different NACA families."""
    m = int(series[0])
    t = int(series[2:])
    
    if m == 0:
        return "Symmetric: Aerobatics, tail surfaces, helicopter rotors"
    elif t < 12:
        return "Thin: High-speed aircraft, low drag"
    elif t < 15:
        return "Medium: General aviation, good lift/drag"
    else:
        return "Thick: Low-speed, high lift, structural depth"
    
def remove_duplicate_points(x, y, tol=1e-6):
    """Remove consecutive duplicate points from airfoil coordinates."""
    if len(x) < 2:
        return x, y
    
    # Keep track of unique points
    unique_x = [x[0]]
    unique_y = [y[0]]
    
    for i in range(1, len(x)):
        # Check if this point is different from the last kept point
        if not (np.isclose(x[i], unique_x[-1], atol=tol) and 
                np.isclose(y[i], unique_y[-1], atol=tol)):
            unique_x.append(x[i])
            unique_y.append(y[i])
    
    return np.array(unique_x), np.array(unique_y)   

# NACA 4-digit generator (Cartesian x,y)
# ─────────────────────────────────────────────────────────────
def naca4_coords_neuralfoil(m, p, t, N=120):
    x = np.linspace(0, 1, N)
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 
              0.2843*x**3 - 0.1015*x**4)
    
    if p == 0 or m == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(x <= p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))
        dyc_dx = np.where(x <= p, 2*m/p**2 * (p - x), 2*m/(1-p)**2 * (p - x))
        
    theta = np.arctan(dyc_dx)
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    
    # 1️⃣ Force exact TE closure (eliminates crossing/self-intersection)
    xu[-1] = xl[-1] = 1.0
    yu[-1] = yl[-1] = 0.0
    
    # 2️⃣ Remove duplicate LE point (x=0 appears in both upper & lower)
    xu, yu = xu[1:], yu[1:]
    
    # 3️⃣ Stack: Upper (near LE → TE) + Lower reversed (TE → LE)
    coords = np.vstack([
        np.column_stack([xu, yu]),
        np.column_stack([xl[::-1], yl[::-1]])
    ])
    
    return coords
def naca4_coords(m, p, t, N=120):
    x = np.linspace(0, 1, N)
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 
              0.2843*x**3 - 0.1015*x**4)
    
    # ⚠️ Explicit guard: np.where evaluates BOTH branches, so p=0 crashes here
    if p == 0 or m == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(x <= p,
                      m/p**2 * (2*p*x - x**2),  
                      m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))
        dyc_dx = np.where(x <= p,
                          2*m/p**2 * (p - x),
                          2*m/(1-p)**2 * (p - x))
                          
    theta = np.arctan(dyc_dx)
    xu, yu = x - yt*np.sin(theta), yc + yt*np.cos(theta)
    xl, yl = x + yt*np.sin(theta), yc - yt*np.cos(theta)
    
    # Order: LE -> TE (upper), then TE -> LE (lower) for closed loop
    return np.vstack([
        np.column_stack([xu, yu]), 
        np.column_stack([xl[::-1], yl[::-1]])
    ])