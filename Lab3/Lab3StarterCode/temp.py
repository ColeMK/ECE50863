from matplotlib import pyplot as plt
import numpy as np
from student.student2 import bitrate_map

if __name__ == "__main__":
    x_arr = x = np.linspace(0, 30, 100)
    y = [bitrate_map(x, [2,4,6,8], 6, 10, 25) for x in x_arr]

    # Create a line plot
    plt.plot(x, y, label='sin(x)')

    # Adding a title
    plt.title("Simple Line Plot")

    # Adding labels
    plt.xlabel("buffer")
    plt.ylabel("chunk size")

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.show()

