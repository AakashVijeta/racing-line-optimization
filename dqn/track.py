from plot_track import compute_boundaries, make_oval, close_loop
import matplotlib.pyplot as plt
import numpy as np

centerline = make_oval()
left, right = compute_boundaries(centerline, track_width=15)

centerline_closed = close_loop(centerline)
left_closed = close_loop(left)
right_closed = close_loop(right)

plt.plot(centerline_closed[:, 0], centerline_closed[:, 1], "k--", label="centerline")
plt.plot(left_closed[:, 0], left_closed[:, 1], "b-", label="left boundary")
plt.plot(right_closed[:, 0], right_closed[:, 1], "r-", label="right boundary")
plt.axis("equal")
plt.legend()
plt.show()
