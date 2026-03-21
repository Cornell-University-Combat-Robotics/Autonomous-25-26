import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "overlap_log.csv"

def log_overlap(is_overlap: bool):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([1 if is_overlap else 0])

def plot_overlap():
    with open(CSV_PATH, "r") as f:
        values = [int(row[0]) for row in csv.reader(f)]

    plt.figure(figsize=(12, 4))
    plt.plot(values, drawstyle="steps-post", color="red", linewidth=1.5)
    plt.yticks([0, 1], ["False", "True"])
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Frame")
    plt.ylabel("Overlap")
    plt.title("Bounding Box Overlap Over Time")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("overlap_graph.png")
    print("Graph saved to overlap_graph.png")
    plt.close()