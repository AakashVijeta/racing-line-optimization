from track_preprocessing import preprocess_circuit
import numpy as np

centerline = preprocess_circuit("circuits/de-1927.geojson", n_points=400)
import matplotlib.pyplot as plt
plt.plot(centerline[:, 0], centerline[:, 1])
plt.axis("equal")
plt.savefig("nurburgring_check.png")