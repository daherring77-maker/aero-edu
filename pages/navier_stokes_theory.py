# navier_stokes_theory.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go


st.title("🌊 Navier-Stokes Equations: The Heart of Aerodynamics")

st.markdown("""
### Why These Equations Matter
The Navier-Stokes equations describe how **fluids move**—air over a wing, water in a pipe, weather systems. 
They are to aerodynamics what Newton's laws are to mechanics.

### The Incompressible Form (2D)
For low-speed aerodynamics (Mach < 0.3), we can assume constant density:

**Momentum Conservation:**
```math
ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + f
```

**Mass Conservation (Incompressibility):**
```math
∇·u = 0
```

Where:
- `u = (u,v)` : velocity vector
- `p` : pressure
- `ρ` : density (constant)
- `μ` : dynamic viscosity
- `Re = ρUL/μ` : Reynolds number (ratio of inertial to viscous forces)
""")

# Interactive Reynolds number demo
st.subheader("🎛️ Reynolds Number Explorer")
Re = st.slider("Reynolds Number", 1, 10000, 100)

# Show flow regimes
if Re < 10:
    regime = "🐌 Creeping Flow (Stokes) — Viscous forces dominate"
    color = "blue"
elif Re < 1000:
    regime = "🌊 Laminar Flow — Smooth, ordered layers"
    color = "green"
elif Re < 5000:
    regime = "🌀 Transitional — Beginning of instability"
    color = "orange"
else:
    regime = "🌪️ Turbulent Flow — Chaotic, mixing"
    color = "red"

st.markdown(f"**At Re = {Re:,.0f}**: *{regime}*")

# Simple visualization of flow regime
fig = go.Figure()
x = np.linspace(0, 1, 100)

if Re < 100:
    # Smooth parabolic profile
    y = 4 * x * (1 - x)
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='Velocity Profile'))
else:
    # Add some "turbulent" noise
    y = 4 * x * (1 - x) + 0.1 * np.random.randn(100) * np.sqrt(Re/1000)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Velocity (fluctuating)', line=dict(width=1)))

fig.update_layout(
    title=f"Qualitative Velocity Profile at Re = {Re:,.0f}",
    xaxis_title="Channel Height",
    yaxis_title="Velocity",
    height=300
)
st.plotly_chart(fig, width='stretch')

st.markdown("""
### Why We Use Streamfunction-Vorticity for Education

| Challenge | Standard (u,v,p) Formulation | ψ-ω Formulation |
|-----------|-----------------------------|-----------------|
| Pressure coupling | Must solve Poisson equation for p | ✅ Eliminated |
| Incompressibility | Must enforce ∇·u = 0 explicitly | ✅ Built-in by definition |
| Number of equations | 3 (2 momentum + 1 continuity) | ✅ 2 scalar equations |
| Boundary conditions | Complex for pressure | Simpler for walls |

**Trade-off**: Only works in 2D. But for learning fundamentals? Perfect.
""")

with st.expander("🔍 Derivation: From (u,v,p) to (ψ,ω)"):
    st.markdown("""
    1. **Define vorticity**: ω = ∂v/∂x - ∂u/∂y (scalar in 2D)
    2. **Define streamfunction**: u = ∂ψ/∂y, v = -∂ψ/∂x
        - This automatically satisfies ∇·u = 0 ✓
    3. **Take curl of momentum equation** → eliminates pressure
    4. **Result**: 
        - Vorticity transport: ∂ω/∂t + u·∇ω = (1/Re)∇²ω
        - Poisson equation: ∇²ψ = -ω
    """)

st.info(r"💡 **Key Insight**: Every term in the vorticity equation has physical meaning:"
"- `∂ω/∂t` : Local change of rotation"
"- `u·∇ω` : Advection of vorticity by flow"
"- `(1/Re)∇²ω` : Diffusion of vorticity (viscous smoothing)")
