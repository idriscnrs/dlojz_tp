from pathlib import Path
import csv
import matplotlib.pyplot as plt


def plot_lrfinder(jobid, folder="lrfinder"):
    matches = list(Path(folder).glob(f"*_{jobid}.csv"))
    if not matches:
        raise FileNotFoundError(f"No file matching '*_{jobid}.csv' in '{folder}/'")
    filename = matches[0]

    # Read CSV file
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Extract the two lines
    # Skip the first 10 values and the last 5
    log_lr = [float(val) for val in rows[0][10:-5]]
    loss = [float(val) for val in rows[1][10:-5]]
    lr = [10**l for l in log_lr]

    # Create the figure
    plt.figure()

    # Plot figure
    plt.plot(lr, loss, 'b', label='loss')
    plt.xscale('log')
    plt.xlabel('Learning rate (log scale)')
    plt.ylabel('Loss')
    plt.grid(True)

    base_name = filename.stem
    plt.title(base_name.split("_")[1:7])
    # figure stays open: the inline backend displays it at cell end (no show()/close())
