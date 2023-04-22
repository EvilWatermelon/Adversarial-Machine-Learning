import numpy as np
import matplotlib.pyplot as plt
import pandas as pdb

def risk_matrix():
    fig = plt.figure()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('Consequence')
    plt.ylabel('Likelihood')
    plt.title('Risk Matrix')

    nrows=3
    ncols=3
    axes = [fig.add_subplot(nrows, ncols, r * ncols + c + 1) for r in range(0, nrows) for c in range(0, ncols) ]

    # remove the x and y ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0,5)
        ax.set_ylim(0,5)

    #Add background colors
    #This has been done manually for more fine-grained control
    #Run the loop below to identify the indice of the axes

    #Identify the index of the axes
    #for i in range(len(axes)):
    #    axes[i].text(0,0, i)

    green = [3, 6, 7] #Green boxes
    yellow = [0, 4, 8] #yellow boxes
    red = [1, 2, 5] #red boxes

    for _ in green:
        axes[_].set_facecolor('lightgreen')

    for _ in yellow:
        axes[_].set_facecolor('lightyellow')

    for _ in red:
        axes[_].set_facecolor('red')

    #axes[2].text(1.2, 1.2, '4.62')
    #axes[2].plot(2, 2, color='black', marker='o')

    #axes[6].text(1.2, 1.2, '21 steps')
    #axes[6].plot(2, 2, color='black', marker='o')

    #axes[6].text(1.2, 1.2, '1825.73MB RAM')
    #axes[6].plot(2, 2, color='black', marker='o')
    #axes[3].text(1.2, 1.2, '9% CPU')
    #axes[3].plot(2, 2, color='black', marker='o')
    #axes[7].text(1.2, 1.2, '8137.81MB GPU')
    #axes[7].plot(2, 2, color='black', marker='o')

    #plt.show()
    #plt.savefig('risk_matrix.png')

#risk_matrix()

# Visualizing The Dataset
def dataset_visualization(class_num, train_number):
    plt.figure(figsize=(21,10))
    plt.bar(class_num, train_number)
    plt.xticks(class_num, rotation='vertical')
    plt.show()
