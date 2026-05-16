import streamlit as st
import io, zipfile, plotly.io as pio
import plotly.graph_objects as go
from modules.utils import recover_pressure
import numpy as np
"""
plotting.py - Reusable Plotly functions for both display and export
"""
def add_plot_download_button(fig, filename, label="Download PNG"):
    """Create a Streamlit download button for a Plotly figure as PNG."""
    import plotly.io as pio
    # Convert figure to PNG bytes
    img_bytes = pio.to_image(fig, format="png", width=800, height=600, scale=2)
    # Streamlit download button
    return st.download_button(
        label=label,
        data=img_bytes,
        file_name=f"{filename}.png",
        mime="image/png"
    )


def create_velocity_plot(u, x, y, title="Horizontal Velocity (u)", 
                         cmap='RdBu_r', zmin=None, zmax=None, for_export=False):
    """
    Create a heatmap of horizontal velocity.
    
    Parameters
    ----------
    u : ndarray (ny, nx)
        Horizontal velocity field
    x : ndarray (nx,)
        X coordinates
    y : ndarray (ny,)
        Y coordinates
    title : str
        Plot title
    cmap : str
        Plotly colorscale name
    zmin, zmax : float
        Color scale limits (auto-calculated if None)
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # Auto-scale if not provided
    if zmin is None:
        zmin = np.min(u)
    if zmax is None:
        zmax = np.max(u)
    
    fig = go.Figure(data=go.Heatmap(
        z=u, 
        x=x, 
        y=y,
        colorscale='viridis',
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title="u-velocity", thickness=20)
    ))
    # Layout settings
    if for_export:
        fig.update_layout(
            title={'text': title, 'font': {'size': 16}},
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial', 'size': 14},
            margin={'l': 60, 'r': 80, 't': 60, 'b': 60}  # Extra right margin for colorbar
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
    
    # Equal aspect ratio (critical for CFD!)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def create_streamline_plot(psi, x, y, title="Streamlines", 
                           n_contours=20, cmap='Viridis', for_export=False):
    """
    Create contour plot of streamfunction (streamlines).
    
    Parameters
    ----------
    psi : ndarray (ny, nx)
        Streamfunction field
    x : ndarray (nx,)
        X coordinates
    y : ndarray (ny,)
        Y coordinates
    title : str
        Plot title
    n_contours : int
        Number of contour lines
    cmap : str
        Plotly colorscale name
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = go.Figure(data=go.Contour(
        z=psi,
        x=x,
        y=y,
        contours_coloring='lines',
        contours=dict(
            start=np.min(psi),
            end=np.max(psi),
            size=(np.max(psi) - np.min(psi)) / n_contours,
            #line_width=2
        ),
        colorscale=cmap,
        #showlines=False
    ))
    # Layout settings
    if for_export:
        fig.update_layout(
            title={'text': title, 'font': {'size': 16}},
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial', 'size': 14},
            margin={'l': 60, 'r': 80, 't': 60, 'b': 60}  # Extra right margin for colorbar
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def create_vorticity_plot(omega, x, y, title="Vorticity (ω)", 
                          cmap='RdBu_r', zmin=None, zmax=None, for_export=False):
    """
    Create heatmap of vorticity field.
    
    Parameters
    ----------
    omega : ndarray (ny, nx)
        Vorticity field
    x : ndarray (nx,)
        X coordinates
    y : ndarray (ny,)
        Y coordinates
    title : str
        Plot title
    cmap : str
        Plotly colorscale name
    zmin, zmax : float
        Color scale limits (auto-calculated if None)
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # Auto-scale with symmetric limits for better visualization
    if zmin is None or zmax is None:
        max_abs = np.max(np.abs(omega))
        zmin = -max_abs
        zmax = max_abs
    
    fig = go.Figure(data=go.Heatmap(
        z=omega,
        x=x,
        y=y,
        colorscale=cmap,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title="ω", thickness=20)
    ))
    
    # Layout settings
    if for_export:
        fig.update_layout(
            title={'text': title, 'font': {'size': 16}},
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial', 'size': 14},
            margin={'l': 60, 'r': 80, 't': 60, 'b': 60}  # Extra right margin for colorbar
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def create_ke_plot(t_history, ke_history, 
                   title="Kinetic Energy History",
                   for_export=False):
    """
    Create line plot of kinetic energy vs time.
    
    Parameters
    ----------
    t_history : list
        Time values
    ke_history : list
        Kinetic energy values
    title : str
        Plot title
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_history,
        y=ke_history,
        mode='lines',
        name='Kinetic Energy',
        line=dict(color='blue', width=2)
    ))
    # Layout settings
    if for_export:
        fig.update_layout(
            title={'text': title, 'font': {'size': 16}},
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial', 'size': 14},
            margin={'l': 60, 'r': 80, 't': 60, 'b': 60}  # Extra right margin for colorbar
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Kinetic Energy",
            width=800,
            height=400,
            yaxis_type="log" if np.max(ke_history) > 10 else "linear"
        )
    
    return fig

"""
plotting.py - Pathline visualization (dual-purpose)
"""

def plot_pathlines(trajectories, 
                   title="Particle Pathlines",
                   for_export=False,
                   cmap='Viridis'):
    """
    Plot particle pathlines colored by speed.
    
    Parameters
    ----------
    trajectories : list of dict
        Each dict has 'path' (nx2 array) and 'speed' (n array)
    title : str
        Plot title
    for_export : bool
        If True, use publication-quality settings
    cmap : str
        Plotly colorscale name
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    for i, traj_data in enumerate(trajectories):
        traj = traj_data['path']
        speed = traj_data['speed']
        
        # ✅ Convert numpy array to list (Plotly requirement)
        speed_list = speed.tolist()
        
        # ✅ Use markers+lines for color-by-variable support
        fig.add_trace(go.Scatter(
            x=traj[:, 0].tolist(), 
            y=traj[:, 1].tolist(), 
            mode='lines+markers',
            marker={
                'color': speed_list,
                'colorscale': cmap,
                'size': 4,
                'showscale': (i == 0),  # Only show colorbar once
                'colorbar': {
                    'title': 'Speed',
                    'thickness': 15,
                    'x': 1.0 + i * 0.05  # Slight offset if multiple colorbars
                } if i == 0 else None
            },
            line={
                'width': 2,
                'color': 'gray'  # Line color (markers show speed)
            },
            name=f'Pathline {i+1}',
            hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>Speed: %{marker.color:.3f}<extra></extra>'
        ))
    
    # Layout settings
    if for_export:
        fig.update_layout(
            title={'text': title, 'font': {'size': 16}},
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial', 'size': 14},
            margin={'l': 60, 'r': 80, 't': 60, 'b': 60}  # Extra right margin for colorbar
        )
    else:
        fig.update_layout(
            title={'text': title, 'font': {'size': 14}},
            xaxis_title="x",
            yaxis_title="y",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    # Equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    
    return fig

def create_pressure_plot(p, x, y, 
                         title="Pressure Field",
                         for_export=False):
    """
    Create pressure contour plot for display and export.
    """
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    # Clean pressure data
    p_clean = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Symmetric color scale
    p_max = np.max(np.abs(p_clean))
    if p_max < 1e-6:
        p_max = 1.0
    
    # Pressure contours
    fig.add_trace(go.Contour(
        z=p_clean,
        x=x,
        y=y,
        colorscale='RdBu_r',  # Red=high pressure, Blue=low pressure
        contours={
            'coloring': 'heatmap',
            'showlines': False
        },
        colorbar={
            'title': 'p',
            'thickness': 20
        },
        zmin=-p_max,
        zmax=p_max
    ))
    
    # Layout
    if for_export:
        fig.update_layout(
            title={'text': title, 'font': {'size': 16}},
            xaxis_title="x",
            yaxis_title="y",
            width=800,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial', 'size': 14},
            margin={'l': 60, 'r': 80, 't': 60, 'b': 60}
        )
    else:
        fig.update_layout(
            title={'text': title, 'font': {'size': 14}},
            xaxis_title="x",
            yaxis_title="y",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    
    return fig

def create_cavity_results_zip(results, trajs=None, Re=None):
    """
    Bundle all cavity flow results into a ZIP file.
    
    Parameters
    ----------
    results : dict
        Simulation results (x, y, u, v, psi, omega, t_hist, ke_hist, params)
    trajs : list of dict, optional
        Particle trajectories from track_particles_with_speed()
    Re : float, optional
        Reynolds number (from results['params']['Re'] if not provided)
    
    Returns
    -------
    zip_buffer : BytesIO
        ZIP file buffer for download
    """
    import io, zipfile, plotly.io as pio
    from modules.plotting import (
        create_velocity_plot,
        create_streamline_plot,
        create_vorticity_plot,
        create_ke_plot,
        create_pressure_plot,
        plot_pathlines
    )
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Extract data
        x, y, u, v, psi, omega = results['x'], results['y'], results['u'], results['v'], results['psi'], results['omega']
        t_hist, ke_hist = results['t_hist'], results['ke_hist']
        params = results.get('params', {})
        Re = Re if Re is not None else params.get('Re', 0)
        
        # Grid spacing (for pressure recovery)
        nx, ny = len(x), len(y)
        dx = x[1] - x[0] if nx > 1 else 0.01
        dy = y[1] - y[0] if ny > 1 else 0.01
        
        # === CREATE ALL FIGURES ===
        figs = {}
        
        # 1. Velocity
        figs['velocity'] = create_velocity_plot(
            u, x, y, 
            title=f"Velocity (Re={Re})",
            for_export=True
        )
        
        # 2. Streamlines
        figs['streamlines'] = create_streamline_plot(
            psi, x, y, 
            title=f"Streamlines (Re={Re})",
            for_export=True
        )
        
        # 3. Vorticity
        figs['vorticity'] = create_vorticity_plot(
            omega, x, y, 
            title=f"Vorticity (Re={Re})",
            for_export=True
        )
        
        # 4. Kinetic Energy History
        figs['kinetic_energy'] = create_ke_plot(
            t_hist, ke_hist, 
            title=f"KE History (Re={Re})",
            for_export=True
        )
        
        # 5. Pressure (NEW!)
        try:
            p = recover_pressure(u, v, Re, dx, dy)
            figs['pressure'] = create_pressure_plot(
                p, x, y, 
                title=f"Pressure (Re={Re})",
                for_export=True
            )
        except Exception as e:
            st.warning(f"⚠️ Pressure recovery failed: {e}")
        
        # 6. Pathlines (NEW!)
        if trajs is not None and len(trajs) > 0:
            figs['pathlines'] = plot_pathlines(
                trajs, 
                title=f"Particle Pathlines (Re={Re})",
                for_export=True
            )
        
        # === SAVE ALL FIGURES TO ZIP ===
        for name, fig in figs.items():
            try:
                img = pio.to_image(fig, format="png", width=800, height=600, scale=2)
                zip_file.writestr(f"{name}_Re{int(Re)}.png", img)
            except Exception as e:
                st.warning(f"⚠️ Failed to export {name}: {e}")
        
        # === SAVE RAW DATA ===
        import numpy as np
        data_buffer = io.BytesIO()
        np.savez(data_buffer, **results)
        zip_file.writestr("simulation_data.npz", data_buffer.getvalue())
        
        # === SAVE TRAJECTORIES (if available) ===
        if trajs is not None:
            traj_buffer = io.BytesIO()
            np.savez(traj_buffer, trajectories=trajs)
            zip_file.writestr("trajectories.npz", traj_buffer.getvalue())
        
        # === SAVE METADATA ===
        import json
        metadata = {
            'Re': Re,
            'grid': (nx, ny),
            'dt': params.get('dt', 0),
            't_end': params.get('t_end', 0),
            'timestamp': results.get('timestamp', 'unknown')
        }
        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer


def track_particles(u, v, x, y, seed_points, dt, n_steps):
    """
    Lagrangian particle tracking through Eulerian velocity field.
    
    Parameters
    ----------
    u, v : ndarray (ny, nx)
        Velocity field
    x, y : ndarray
        Grid coordinates
    seed_points : list of (x0, y0)
        Starting positions for particles
    dt : float
        Time step for integration
    n_steps : int
        Number of steps to track
    
    Returns
    -------
    trajectories : list of ndarray
        Each element is (n_steps, 2) array of (x, y) positions
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Create interpolators for velocity field
    u_interp = RegularGridInterpolator((y, x), u, method='linear', bounds_error=False)
    v_interp = RegularGridInterpolator((y, x), v, method='linear', bounds_error=False)
    
    trajectories = []
    
    for x0, y0 in seed_points:
        traj = np.zeros((n_steps, 2))
        x_curr, y_curr = x0, y0
        
        for i in range(n_steps):
            traj[i] = [x_curr, y_curr]
            
            # Interpolate velocity at current position
            u_curr = u_interp([y_curr, x_curr])[0]
            v_curr = v_interp([y_curr, x_curr])[0]
            
            # Euler integration (can upgrade to RK4)
            x_curr += u_curr * dt
            y_curr += v_curr * dt
            
            # Check bounds
            if x_curr < 0 or x_curr > 1 or y_curr < 0 or y_curr > 1:
                traj = traj[:i+1]  # Trim trajectory
                break
        
        trajectories.append(traj)
    
    return trajectories


# === EXPORT TO ZIP ===

"""
plotting.py - Airfoil streamline visualization (dual-purpose)
"""

import plotly.graph_objects as go
import numpy as np

def create_streamlines_panel_plot(x_airfoil, y_airfoil, alpha, 
                             V_inf=1.0, 
                             n_streamlines=15,
                             title=None,
                             for_export=False,
                             cmap='Blues',
                             show_airfoil=True,
                             show_speed_bg=True):
    """
    Plot streamlines around airfoil using potential flow approximation.
    Dual-purpose: works for Streamlit display and PNG export.
    
    Parameters
    ----------
    x_airfoil, y_airfoil : ndarray
        Airfoil coordinates
    alpha : float
        Angle of attack (degrees)
    V_inf : float
        Freestream velocity magnitude
    n_streamlines : int
        Number of streamlines to display
    title : str or None
        Plot title (auto-generated if None)
    for_export : bool
        If True, use publication-quality settings
    cmap : str
        Plotly colorscale name for speed background
    show_airfoil : bool
        Show airfoil outline
    show_speed_bg : bool
        Show velocity magnitude background
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Reusable for st.plotly_chart() or pio.to_image()
    """
    if title is None:
        title = f"Streamlines (α={alpha}°)"
    
    fig = go.Figure()
    
    # === Airfoil Outline ===
    if show_airfoil:
        fig.add_trace(go.Scatter(
            x=x_airfoil, 
            y=y_airfoil,
            mode='lines',
            line=dict(color='black', width=2 if not for_export else 3),
            name='Airfoil',
            hoverinfo='skip'
        ))
    
    # === Velocity Field Grid ===
    x_grid = np.linspace(-0.5, 1.5, 100)
    y_grid = np.linspace(-0.5, 0.5, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Freestream + vortex at quarter-chord (simplified potential flow)
    x_vortex = 0.25
    y_vortex = 0.0
    gamma = 2 * np.pi * V_inf * np.sin(np.radians(alpha))
    
    # Velocity field (avoid singularity at vortex core)
    R2 = (X - x_vortex)**2 + (Y - y_vortex)**2
    R2 = np.where(R2 < 0.01, 0.01, R2)
    
    u = V_inf * np.cos(np.radians(alpha)) - gamma * (Y - y_vortex) / (2 * np.pi * R2)
    v = V_inf * np.sin(np.radians(alpha)) + gamma * (X - x_vortex) / (2 * np.pi * R2)
    speed = np.sqrt(u**2 + v**2)
    
    # === Speed Background (optional) ===
    if show_speed_bg:
        fig.add_trace(go.Heatmap(
            x=x_grid, 
            y=y_grid, 
            z=speed,
            colorscale=cmap, 
            showscale=False, 
            opacity=0.3 if not for_export else 0.4,
            hoverinfo='skip',
            name='Speed'
        ))
    
    # === Streamlines via Contour of Streamfunction ===
    # Streamfunction ψ for freestream + vortex:
    # ψ = V_inf * (y*cos(α) - x*sin(α)) + (Γ/2π) * ln(r)
    psi = (V_inf * (Y * np.cos(np.radians(alpha)) - X * np.sin(np.radians(alpha))) + 
           (gamma / (2 * np.pi)) * np.log(np.sqrt(R2)))
    
    # Add streamline contours
    fig.add_trace(go.Contour(
        x=x_grid,
        y=y_grid,
        z=psi,
        contours=dict(
            coloring='lines',
            showlines=True,
            start=psi.min(),
            end=psi.max(),
            size=(psi.max() - psi.min()) / n_streamlines
        ),
        line=dict(width=1.5 if not for_export else 2, color='rgba(0,0,100,0.6)'),
        showscale=False,
        hoverinfo='skip',
        name='Streamlines'
    ))
    
    # === Layout Settings ===
    if for_export:
        # Publication quality
        fig.update_layout(
            title={'text': title, 'font': {'size': 16, 'family': 'Arial'}},
            xaxis_title="x/c",
            yaxis_title="y/c",
            width=800,
            height=600,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial', 'size': 14},
            margin={'l': 60, 'r': 60, 't': 60, 'b': 60},
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                range=[-0.2, 1.2]
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                range=[-0.3, 0.3]
            )
        )
    else:
        # Streamlit display (responsive)
        fig.update_layout(
            title={'text': title, 'font': {'size': 14}},
            xaxis_title="x/c",
            yaxis_title="y/c",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin={'l': 50, 'r': 50, 't': 50, 'b': 50},
            xaxis=dict(range=[-0.2, 1.2]),
            yaxis=dict(range=[-0.3, 0.3])
        )
    
    # Equal aspect ratio (critical for airfoil shape!)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def create_cp_distribution_plot(x_cp, y_cp, Cp, 
                           title="Pressure Coefficient",
                           for_export=False):
    """
    Create Cp distribution plot for Streamlit display and PNG export.
    Compatible with Plotly >= 4.0
    """
    fig = go.Figure()
    
    # Clean Cp data (remove NaN/Inf)
    Cp_clean = np.nan_to_num(Cp, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Auto-scale Cp limits symmetrically
    Cp_max = np.max(np.abs(Cp_clean))
    if Cp_max < 1e-6:  # Avoid division by zero
        Cp_max = 1.0
    zmin = -Cp_max
    zmax = Cp_max
    
    # Airfoil surface with Cp coloring
    fig.add_trace(go.Scatter(
        x=x_cp, 
        y=y_cp,
        mode='lines+markers',
        marker={
            'color': Cp_clean.tolist(),  # Convert numpy array to list for compatibility
            'colorscale': 'RdBu_r',
            'colorbar': {
                'title': 'Cp', 
                'thickness': 20
                # Removed 'titleside' for compatibility
            },
            'size': 6,
            'showscale': True
        },
        line={'width': 2, 'color': 'gray'},
        name='Airfoil Surface'
    ))
    
    # Layout settings
    if for_export:
        fig.update_layout(
            title={'text': title, 'font': {'size': 16}},
            xaxis_title="x/c",
            yaxis_title="y/c",
            width=800,
            height=600,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial', 'size': 14},
            margin={'l': 60, 'r': 60, 't': 60, 'b': 60}
        )
    else:
        fig.update_layout(
            title={'text': title, 'font': {'size': 14}},
            xaxis_title="x/c",
            yaxis_title="y/c",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    # Equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[-0.15, 0.15])
    fig.update_xaxes(range=[-0.05, 1.05])
    
    return fig

"""
plotting.py - Lift visualization (dual-purpose)
"""

import plotly.graph_objects as go
import numpy as np

def create_lift_visualization_plot(x_airfoil, y_airfoil, Cp, Cl, alpha,
                              V_inf=1.0,
                              title=None,
                              for_export=False,
                              show_pressure_arrows=True,
                              show_lift_vector=True,
                              show_freestream=True,
                              arrow_scale=0.05,
                              cmap='RdBu_r'):
    """
    Visualize lift generation via pressure distribution and force vector.
    Dual-purpose: works for Streamlit display and PNG export.
    
    Parameters
    ----------
    x_airfoil, y_airfoil : ndarray
        Airfoil coordinates (closed loop)
    Cp : ndarray
        Pressure coefficient at panel centers
    Cl : float
        Lift coefficient
    alpha : float
        Angle of attack (degrees)
    V_inf : float
        Freestream velocity magnitude
    title : str or None
        Plot title (auto-generated if None)
    for_export : bool
        If True, use publication-quality settings
    show_pressure_arrows : bool
        Show normal arrows representing pressure
    show_lift_vector : bool
        Show lift force vector at quarter-chord
    show_freestream : bool
        Show freestream direction indicator
    arrow_scale : float
        Scaling factor for pressure arrows
    cmap : str
        Plotly colorscale for Cp (RdBu_r: red=high pressure, blue=suction)
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Reusable for st.plotly_chart() or pio.to_image()
    """
    if title is None:
        title = f"Lift Generation: Cl = {Cl:.2f} (α={alpha}°)"
    
    fig = go.Figure()
    
    # === Airfoil Surface Colored by Cp ===
    # Clean Cp data
    Cp_clean = np.nan_to_num(Cp, nan=0.0, posinf=0.0, neginf=0.0)
    Cp_max = np.max(np.abs(Cp_clean))
    if Cp_max < 1e-6:
        Cp_max = 1.0
    
    fig.add_trace(go.Scatter(
        x=x_airfoil, 
        y=y_airfoil,
        mode='lines+markers',
        marker={
            'color': Cp_clean.tolist(),  # Convert to list for Plotly compatibility
            'colorscale': cmap,
            'colorbar': {
                'title': 'Cp',
                'thickness': 20 if not for_export else 25,
                'len': 0.5
            },
            'size': 4 if not for_export else 5,
            'showscale': True
        },
        line={'width': 2 if not for_export else 3, 'color': 'black'},
        name='Airfoil Surface',
        hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>Cp: %{marker.color:.3f}<extra></extra>'
    ))
    
    # === Pressure Arrows (Normal to Surface) ===
    if show_pressure_arrows and len(x_airfoil) > 1:
        # Compute panel normals and add representative arrows
        n_arrows = min(25, len(x_airfoil) // 4)  # Avoid clutter
        indices = np.linspace(0, len(x_airfoil)-2, n_arrows, dtype=int)
        
        for idx in indices:
            if idx < len(Cp_clean):
                # Panel geometry
                x0, y0 = x_airfoil[idx], y_airfoil[idx]
                x1, y1 = x_airfoil[idx+1], y_airfoil[idx+1]
                dx, dy = x1 - x0, y1 - y0
                
                # Normal vector (outward, counter-clockwise ordering)
                nx, ny = -dy, dx
                norm = np.sqrt(nx**2 + ny**2)
                if norm > 1e-6:
                    nx, ny = nx/norm, ny/norm
                    
                    # Arrow length proportional to |Cp|, direction by sign
                    arrow_len = arrow_scale * abs(Cp_clean[idx])
                    arrow_dir = -1 if Cp_clean[idx] < 0 else 1  # Suction pulls, pressure pushes
                    
                    # Arrow color by Cp sign
                    arrow_color = 'royalblue' if Cp_clean[idx] < 0 else 'firebrick'
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x0 + arrow_dir * arrow_len * nx],
                        y=[y0, y0 + arrow_dir * arrow_len * ny],
                        mode='lines',
                        line={'color': arrow_color, 'width': 2},
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # === Lift Force Vector ===
    if show_lift_vector:
        # Quarter-chord location (aerodynamic center for thin airfoils)
        x_lift = 0.25
        # Interpolate y at quarter-chord
        y_lift = np.interp(x_lift, x_airfoil, y_airfoil)
        
        # Lift vector: upward, scaled by Cl for visibility
        lift_scale = 0.25 if not for_export else 0.3
        fig.add_trace(go.Scatter(
            x=[x_lift, x_lift],
            y=[y_lift, y_lift + lift_scale * Cl],
            mode='lines+markers',
            line={'color': 'forestgreen', 'width': 3 if not for_export else 4, 'dash': 'dot'},
            marker={'size': 8, 'color': 'forestgreen', 'symbol': 'triangle-up'},
            name=f'Lift (Cl={Cl:.2f})',
            hovertemplate='Lift vector<br>Cl: ' + str(round(Cl, 3)) + '<extra></extra>'
        ))
        
        # Add Cl annotation
        fig.add_annotation(
            x=x_lift + 0.05,
            y=y_lift + lift_scale * Cl * 0.5,
            text=f'Cl = {Cl:.2f}',
            showarrow=False,
            font={'size': 12 if not for_export else 14, 'color': 'forestgreen', 'family': 'Arial'}
        )
    
    # === Freestream Indicator ===
    if show_freestream:
        # Freestream arrow (bottom-left corner)
        fig.add_trace(go.Scatter(
            x=[-0.3, -0.1], 
            y=[-0.2, -0.2],
            mode='lines+markers',
            line={'color': 'gray', 'width': 2},
            marker={'size': 8, 'color': 'gray', 'symbol': 'arrow-right'},
            name=f'Freestream (V∞={V_inf}, α={alpha}°)',
            hoverinfo='skip'
        ))
        
        # Angle of attack arc
        theta = np.linspace(0, np.radians(alpha), 20)
        arc_r = 0.08
        fig.add_trace(go.Scatter(
            x=[-0.2 + arc_r * np.cos(t) for t in theta],
            y=[-0.2 + arc_r * np.sin(t) for t in theta],
            mode='lines',
            line={'color': 'gray', 'width': 1, 'dash': 'dot'},
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # === Layout Settings ===
    if for_export:
        # Publication quality
        fig.update_layout(
            title={'text': title, 'font': {'size': 16, 'family': 'Arial'}},
            xaxis_title="x/c",
            yaxis_title="y/c",
            width=800,
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'Arial', 'size': 14},
            margin={'l': 60, 'r': 80, 't': 60, 'b': 60},  # Extra right for colorbar
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                range=[-0.2, 1.2]
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                range=[-0.3, 0.3]
            )
        )
    else:
        # Streamlit display (responsive)
        fig.update_layout(
            title={'text': title, 'font': {'size': 14}},
            xaxis_title="x/c",
            yaxis_title="y/c",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin={'l': 50, 'r': 70, 't': 50, 'b': 50},  # Extra right for colorbar
            xaxis=dict(range=[-0.2, 1.2]),
            yaxis=dict(range=[-0.3, 0.3])
        )
    
    # Equal aspect ratio (critical for airfoil shape!)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def plot_ghia_comparison(y, u_sim, x, v_sim, Re, ghia_data=None):
    """
    Plot simulation vs. Ghia benchmark.
    
    Parameters
    ----------
    y : ndarray
        y-coordinates for u-velocity profile
    u_sim : ndarray
        Simulated u-velocity along vertical centerline
    x : ndarray
        x-coordinates for v-velocity profile
    v_sim : ndarray
        Simulated v-velocity along horizontal centerline
    Re : int
        Reynolds number
    ghia_data : dict
        Reference data from Ghia et al. (1982)
        Keys: 'y', 'u', 'x', 'v' (NOT 'u_ghia', 'v_ghia')
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(f"u-velocity (x=0.5)", f"v-velocity (y=0.5)"))
    
    # === Plot Simulation ===
    fig.add_trace(go.Scatter(x=u_sim, y=y, mode='lines', name='Simulation',
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=v_sim, y=x, mode='lines', name='Simulation',
                            line=dict(color='blue', width=2), showlegend=False), row=1, col=2)
    
    # === Plot Ghia Data (if available) ===
    if ghia_data is not None:
        # ✅ FIXED: Use correct keys ('u' and 'v', not 'u_ghia' and 'v_ghia')
        fig.add_trace(go.Scatter(x=ghia_data['u'], y=ghia_data['y'], 
                                mode='markers', name='Ghia et al. (1982)',
                                marker=dict(symbol='x', color='red', size=10, line=dict(width=2))), 
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=ghia_data['v'], y=ghia_data['x'], 
                                mode='markers', name='Ghia',
                                marker=dict(symbol='x', color='red', size=10, line=dict(width=2)), 
                                showlegend=False), 
                     row=1, col=2)
    
    fig.update_layout(title=f"Ghia Benchmark Comparison (Re={Re})",
                     height=500, showlegend=True,
                     font=dict(family='Arial', size=12))
    fig.update_xaxes(title_text="u", row=1, col=1)
    fig.update_xaxes(title_text="v", row=1, col=2)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="x", row=1, col=2)
    
    # Equal aspect ratio for both subplots
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    
    return fig

# === EXPORT TO ZIP ===
"""
aero_export.py - Airfoil results export (matches cavity flow pattern)
"""

import io, zipfile, json
import numpy as np
import plotly.io as pio


def create_aero_results_zip(results, x_airfoil, y_airfoil):
    """
    Bundle all airfoil panel method results into a ZIP file.
    Matches the cavity flow export pattern for consistency.
    
    Parameters
    ----------
    results : dict
        Panel method results with keys:
        - 'Cp', 'Cl', 'x_cp', 'y_cp', 'gamma', 'alpha', 'V_inf'
        - 'series' (NACA code), 'timestamp' (optional)
    x_airfoil, y_airfoil : ndarray
        Airfoil coordinates (for streamline/lift plots)
    
    Returns
    -------
    zip_buffer : BytesIO
        ZIP file buffer for download
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # === Extract Parameters ===
        alpha = results.get('alpha', 0)
        Cl = results.get('Cl', 0)
        series = results.get('series', 'XXXX')
        V_inf = results.get('V_inf', 1.0)
        timestamp = results.get('timestamp', 'unknown')
        
        # === Create All Figures ===
        figs = {}
        
        # 1. Cp Distribution
        figs['cp_distribution'] = create_cp_distribution_plot(
            results['x_cp'], 
            results['y_cp'], 
            results['Cp'],
            title=f"Cp (NACA{series}, α={alpha}°, Cl={Cl:.2f})",
            for_export=True
        )
        
        # 2. Streamlines
        figs['streamlines'] = create_streamlines_panel_plot(
            x_airfoil, 
            y_airfoil, 
            alpha,
            V_inf=V_inf,
            n_streamlines=25,
            title=f"Streamlines (NACA{series}, α={alpha}°)",
            for_export=True
        )
        
        # 3. Lift Visualization
        figs['lift_visualization'] = create_lift_visualization_plot(
            x_airfoil, 
            y_airfoil,
            results['Cp'],
            results['Cl'],
            alpha,
            V_inf=V_inf,
            title=f"Lift Generation (NACA{series}, α={alpha}°)",
            for_export=True,
            show_pressure_arrows=True,
            arrow_scale=0.04
        )
        
        # === Save All Figures to ZIP ===
        for name, fig in figs.items():
            try:
                img = pio.to_image(fig, format="png", width=800, height=600, scale=2)
                # ✅ Use alpha instead of Re for filename
                zip_file.writestr(f"{name}_NACA{series}_alpha{int(alpha)}.png", img)
            except Exception as e:
                # In Streamlit context, use st.warning
                try:
                    import streamlit as st
                    st.warning(f"⚠️ Failed to export {name}: {e}")
                except:
                    print(f"⚠️ Failed to export {name}: {e}")
        
        # === Save Raw Data ===
        data_buffer = io.BytesIO()
        np.savez(data_buffer, **results)
        zip_file.writestr("aero_data.npz", data_buffer.getvalue())
        
        # === Save Airfoil Coordinates ===
        coords_buffer = io.BytesIO()
        np.savez(coords_buffer, x=x_airfoil, y=y_airfoil)
        zip_file.writestr("airfoil_coordinates.npz", coords_buffer.getvalue())
        
        # === Save Metadata (matches cavity flow pattern) ===
        metadata = {
            'airfoil': f'NACA{series}',
            'alpha_deg': alpha,
            'Cl': Cl,
            'V_inf': V_inf,
            'method': 'Vortex Panel Method',
            'timestamp': timestamp,
            'n_panels': len(results.get('gamma', []))
        }
        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer