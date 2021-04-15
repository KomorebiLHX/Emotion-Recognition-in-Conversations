import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.style.use('fivethirtyeight')
    data_list = [56.23, 56.71, 58.41]
    labels = ['1', '2', '3']
    width = 0.35
    ind = np.arange(3)
    fig, ax = plt.subplots()
    p1 = ax.bar(ind, data_list, width)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.set_ylim(50, 60)
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Number of layers')
    ax.set_title('Comparison of different number of layers')
    for x, y in enumerate(data_list):
        plt.text(x - 0.1, y+0.5, "%s" % y)
    plt.show()

