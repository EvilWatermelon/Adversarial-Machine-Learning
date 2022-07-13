import numpy as np
import matplotlib.pyplot as plt
import pandas as pdb

fig = plt.figure()
plt.subplots_adjust(wspace=0, hspace=0)
plt.xticks([])
plt.yticks([])
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.xlabel('Consequence')
plt.ylabel('Likelihood')
plt.title('Risk Matrix')

#This example is for a 5 * 5 matrix
nrows=5
ncols=5
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

green = [10, 15, 16, 20 , 21] #Green boxes
yellow = [0, 5, 6, 11, 17, 22, 23] #yellow boxes
orange = [1 , 2, 7, 12, 13, 18, 19, 24] # orange boxes
red = [3, 4, 8, 9, 14] #red boxes

for _ in green:
    axes[_].set_facecolor('green')

for _ in yellow:
    axes[_].set_facecolor('yellow')

for _ in orange:
    axes[_].set_facecolor('orange')

for _ in red:
    axes[_].set_facecolor('red')


#Add labels to the Green boxes
axes[10].text(0.1,0.8, '4')
axes[15].text(0.1,0.8, '2')
axes[20].text(0.1,0.8, '1')
axes[16].text(0.1,0.8, '5')
axes[21].text(0.1,0.8, '3')


#Add labels to the Yellow boxes
axes[0].text(0.1,0.8, '11')
axes[5].text(0.1,0.8, '7')
axes[6].text(0.1,0.8, '12')
axes[11].text(0.1,0.8, '8')
axes[17].text(0.1,0.8, '9')
axes[22].text(0.1,0.8, '6')
axes[23].text(0.1,0.8, '10')

#Add lables to the Orange boxes
axes[1].text(0.1,0.8, '16')
axes[2].text(0.1,0.8, '20')
axes[7].text(0.1,0.8, '17')
axes[12].text(0.1,0.8, '13')
axes[13].text(0.1,0.8, '18')
axes[18].text(0.1,0.8, '14')
axes[19].text(0.1,0.8, '19')
axes[24].text(0.1,0.8, '15')

#Add lables to the Red Boxes
axes[3].text(0.1,0.8, '23')
axes[8].text(0.1,0.8, '21')
axes[4].text(0.1,0.8, '25')
axes[9].text(0.1,0.8, '24')
axes[14].text(0.1,0.8, '22')

#Plot some data

for _ in range(len(axes)):
        axes[_].plot(np.random.uniform(2, 4, 5), np.random.uniform(2,4,5), '.')
#plt.show()
#plt.savefig('risk_matrix.png')

# Visualizing The Dataset
def dataset_visualization(class_num, train_number):
    plt.figure(figsize=(21,10))
    plt.bar(class_num, train_number)
    plt.xticks(class_num, rotation='vertical')
    plt.show()
