import numpy as np
import matplotlib.pyplot as plt

def diagnose_airfoil(x, y):
    # Ensure closed
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    
    n_panels = len(x) - 1
    x_start, y_start = x[:-1], y[:-1]
    x_end, y_end = x[1:], y[1:]
    dx, dy = x_end - x_start, y_end - y_start
    panel_len = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    n_x, n_y = -np.sin(theta), np.cos(theta)
    xc, yc = 0.5 * (x_start + x_end), 0.5 * (y_start + y_end)
    
    # Check for panel length outliers
    print(f"Panel lengths: min={panel_len.min():.6f}, max={panel_len.max():.6f}, mean={panel_len.mean():.6f}")
    if panel_len.min() < 1e-6:
        print("WARNING: Zero-length panel detected!")
    
    # Check for normal vector consistency (should all point outward)
    # For a closed CCW airfoil, centroid-to-panel vector should oppose normal
    centroid_x, centroid_y = np.mean(x), np.mean(y)
    vec_to_centroid_x = centroid_x - xc
    vec_to_centroid_y = centroid_y - yc
    dot_product = vec_to_centroid_x * n_x + vec_to_centroid_y * n_y
    
    inward_count = np.sum(dot_product > 0)  # Normal points toward centroid = inward
    print(f"Normals pointing inward: {inward_count} / {n_panels}")
    if inward_count > n_panels * 0.1:
        print("WARNING: Many normals pointing inward! Geometry may be CW instead of CCW.")
    
    # Plot geometry with normals
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b-', linewidth=2)
    plt.plot(xc, yc, 'ro', markersize=3)
    plt.quiver(xc, yc, n_x, n_y, scale=20, color='green')
    plt.axis('equal')
    plt.title('Airfoil Geometry + Normals')
    plt.grid(True)
    
    # Plot panel ordering (index vs x)
    plt.subplot(1, 2, 2)
    plt.plot(range(n_panels), xc, 'b-o', markersize=3)
    plt.title('Panel Control Point X vs Panel Index')
    plt.xlabel('Panel Index')
    plt.ylabel('X-coordinate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return inward_count, n_panels

