import numpy as np
import plotly.graph_objects as go

def create_aircraft_3d_demo(pitch=0, yaw=0, roll=0):
    """
    Interactive 3D wireframe aircraft showing pitch/yaw/roll.
    Uses simple parametric geometry for educational clarity.
    """
    fig = go.Figure()
    
    # Simple aircraft geometry (fuselage + wing + tail)
    # Fuselage
    fuselage_x = np.linspace(-1, 1, 20)
    fuselage_y = np.zeros(20)
    fuselage_z = 0.1 * np.sin(np.pi * fuselage_x)  # Tapered
    
    # Wing (rectangular for simplicity)
    wing_span = np.linspace(-2, 2, 10)
    wing_chord = np.array([0.3, 0.3])  # Constant chord
    wing_x = np.array([0.0, 0.3])
    wing_y = wing_span
    wing_z = np.zeros_like(wing_span)
    
    # Apply rotations (simplified)
    # Pitch: rotation about y-axis
    # Yaw: rotation about z-axis  
    # Roll: rotation about x-axis
    
    # For demo: just show the effect on wing orientation
    wing_z_rotated = wing_z + 0.2 * np.sin(np.radians(roll)) * wing_y
    
    # Plot fuselage
    fig.add_trace(go.Scatter3d(
        x=fuselage_x, y=fuselage_y, z=fuselage_z,
        mode='lines', line=dict(color='gray', width=4),
        name='Fuselage'
    ))
    
    # Plot wing (both sides)
    for side in [1, -1]:
        fig.add_trace(go.Scatter3d(
            x=wing_x, y=side * wing_y, z=wing_z_rotated,
            mode='lines', line=dict(color='blue', width=3),
            name='Wing' if side == 1 else ''
        ))
    
    # Tail
    tail_x = np.linspace(-0.8, -1.0, 5)
    tail_y = np.linspace(-0.5, 0.5, 5)
    tail_z = np.zeros(5) + 0.15
    fig.add_trace(go.Scatter3d(
        x=tail_x, y=tail_y, z=tail_z,
        mode='lines', line=dict(color='blue', width=2),
        name='Tail'
    ))
    
    # Add rotation indicators
    # Pitch arrow (nose up/down)
    if pitch != 0:
        fig.add_trace(go.Cone(
            x=[0], y=[0], z=[0],
            u=[0], v=[0], w=[pitch * 0.1],
            sizemode="absolute", showscale=False,
            colorscale=[[0, 'red'], [1, 'red']]
        ))
    
    fig.update_layout(
        title=f"Aircraft Attitude: Pitch={pitch}°, Yaw={yaw}°, Roll={roll}°",
        scene=dict(
            xaxis_title='X (Longitudinal)',
            yaxis_title='Y (Lateral)', 
            zaxis_title='Z (Vertical)',
            aspectmode='data'
        ),
        width=800,
        height=600
    )
    
    return fig

