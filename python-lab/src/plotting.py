import matplotlib.pyplot as plt

def simple_plot(x, y, title="Plot"):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("signal")
    plt.grid(True)
    plt.show()
