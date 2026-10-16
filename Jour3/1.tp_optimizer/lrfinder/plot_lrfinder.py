import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str,
                        help='csv filename')
args = parser.parse_args()

# Read CSV file
with open(args.filename, 'r') as f:
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

# Sauvegarder dans un fichier JPG
base_name = Path(args.filename).stem
new_filename = base_name + ".jpg"
plt.title(base_name.split("_")[1:7])
plt.savefig(new_filename, format='jpg')
plt.close()

print(f"Image stored in '{new_filename}'")

