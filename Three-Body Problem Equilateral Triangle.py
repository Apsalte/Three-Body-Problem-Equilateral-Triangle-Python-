import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


G = 1.0  # normalized units
m = 1.0  # equal masses

# Distance from center to each body
R = 1.0  

# Side length of equilateral triangle
L = R * np.sqrt(3)

# Angular velocity for circular orbit
omega = np.sqrt(G * m * 3 / L**3)  # stable rotation

x1, y1 = R, 0
x2, y2 = -R/2, R * np.sqrt(3)/2
x3, y3 = -R/2, -R * np.sqrt(3)/2

# ------------------------------
# Initial velocities (perpendicular to radius vectors)
# ------------------------------
vx1, vy1 = 0, omega * R
vx2, vy2 = -omega * R * np.sqrt(3)/2, -omega * R / 2
vx3, vy3 = omega * R * np.sqrt(3)/2, -omega * R / 2

initial_state = [x1, y1, x2, y2, x3, y3,
                 vx1, vy1, vx2, vy2, vx3, vy3]

t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

def derivatives(t, state):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state

    r1 = np.array([x1, y1])
    r2 = np.array([x2, y2])
    r3 = np.array([x3, y3])

    r12 = r2 - r1
    r13 = r3 - r1
    r23 = r3 - r2

    d12 = np.linalg.norm(r12)
    d13 = np.linalg.norm(r13)
    d23 = np.linalg.norm(r23)

    a1 = G * m * (r12 / d12**3 + r13 / d13**3)
    a2 = G * m * (-r12 / d12**3 + r23 / d23**3)
    a3 = G * m * (-r13 / d13**3 - r23 / d23**3)

    return [vx1, vy1, vx2, vy2, vx3, vy3,
            a1[0], a1[1], a2[0], a2[1], a3[0], a3[1]]

sol = solve_ivp(derivatives, t_span, initial_state, t_eval=t_eval, rtol=1e-9, atol=1e-9)

x1_sol, y1_sol = sol.y[0], sol.y[1]
x2_sol, y2_sol = sol.y[2], sol.y[3]
x3_sol, y3_sol = sol.y[4], sol.y[5]

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Equilateral Triangle Stable Orbit")
ax.grid(True)

trail1, = ax.plot([], [], '-', color='red', lw=1, zorder=1)
trail2, = ax.plot([], [], '-', color='blue', lw=1, zorder=2)
trail3, = ax.plot([], [], '-', color='green', lw=1, zorder=3)

body1, = ax.plot([], [], 'o', color='red', markersize=8, zorder=4)
body2, = ax.plot([], [], 'o', color='blue', markersize=8, zorder=5)
body3, = ax.plot([], [], 'o', color='green', markersize=8, zorder=6)


def animate(i):
    trail1.set_data(x1_sol[:i], y1_sol[:i])
    trail2.set_data(x2_sol[:i], y2_sol[:i])
    trail3.set_data(x3_sol[:i], y3_sol[:i])

    body1.set_data([x1_sol[i]], [y1_sol[i]])
    body2.set_data([x2_sol[i]], [y2_sol[i]])
    body3.set_data([x3_sol[i]], [y3_sol[i]])

    return trail1, trail2, trail3, body1, body2, body3

ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=True)
plt.show()

