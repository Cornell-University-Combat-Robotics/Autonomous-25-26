import pandas as pd
import matplotlib.pyplot as plt
import time

plt.ion()

while True:
    data = pd.read_csv("area.csv")

    plt.clf()
    plt.plot(range(len(data)), data["area"])

    plt.xlabel("Frame")
    plt.ylabel("Area")
    plt.title("Contour Area Over Time")

    plt.pause(0.05)

sys.exit()