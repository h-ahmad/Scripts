# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:29:39 2022

@author: hussain
"""

import matplotlib.pyplot as plt
import numpy as np
 
# Creating dataset
#np.random.seed(10)

data_cnn_node1 = [46.76, 55.59, 58.28, 59.86, 59.92]
data_resnet_node1 = [55.18, 61.25, 65.74, 64.85, 65.34]

data_cnn_node2 = [46.10, 53.11, 56.29, 56.96, 56.03]
data_resnet_node2 = [52.12, 59.87, 64.27, 65.08, 65.55]

data_cnn_node3 = [48.04, 53.97, 59.93, 58.78, 62.07]
data_resnet_node3 = [49.78, 58.97, 65.48, 65.33, 65.42]

data_cnn_node4 = [48.30, 54.33, 60.27, 59.48, 60.54]
data_resnet_node4 = [52.21, 59.92, 64.22, 64.84, 64.84]

data_cnn_swarm = [59.22, 66.85, 67.26, 67.73, 68.93]
data_resnet_swarm = [66.48,  71.70, 73.17, 73.15, 73.08]

data = [data_cnn_node1, data_resnet_node1, data_cnn_node2, data_resnet_node2, data_cnn_node3, data_resnet_node3, 
        data_cnn_node4, data_resnet_node4, data_cnn_swarm, data_resnet_swarm]
fig = plt.figure(figsize =(10, 7))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["N1_CNN2", "N1_Res", "N2_CNN2", "N2_Res", "N3_CNN2", "N3_Res", "N4_CNN2", "N4_Res", "SWARM_CNN2", "SWARM_Res"], rotation=10)
plt.savefig("boxplot.pdf", format="pdf", bbox_inches="tight")
plt.show()
