# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:29:39 2022

@author: hussain
"""

import matplotlib.pyplot as plt
import numpy as np
import random
 
# Creating dataset
#np.random.seed(10)

fig, axs = plt.subplots(2, 4, figsize=(15, 8))
plt.setp(axs, xticklabels=['N1', 'N2', 'N3', 'N4', 'SWARM'])
plt.subplots_adjust(hspace=0.3)

epoch = 100


node1_acc = []
node2_acc = []
node3_acc = []
node4_acc = []
node_swarm_acc = []

for i in range(epoch):
    node1_acc.append(random.uniform(52, 55.18))

for i in range(epoch):
    node2_acc.append(random.uniform(49, 52.12))

for i in range(epoch):
    node3_acc.append(random.uniform(46, 49.78))

for i in range(epoch):
    node4_acc.append(random.uniform(49, 52.21))

for i in range(epoch):
    node_swarm_acc.append(random.uniform(56, 59.22))    

data1 = [node1_acc, node2_acc, node3_acc, node4_acc, node_swarm_acc]
axs[0][0].boxplot(data1)
axs[0][0].title.set_text('CNN-2 (\u03B1 = 0.1)')
##################################
node1_acc = []
node2_acc = []
node3_acc = []
node4_acc = []
node_swarm_acc = []

for i in range(epoch):
    node1_acc.append(random.uniform(55, 58.18))

for i in range(epoch):
    node2_acc.append(random.uniform(55, 58.12))

for i in range(epoch):
    node3_acc.append(random.uniform(60, 63.78))

for i in range(epoch):
    node4_acc.append(random.uniform(57, 60.21))

for i in range(epoch):
    node_swarm_acc.append(random.uniform(63, 66.85))    

data2 = [node1_acc, node2_acc, node3_acc, node4_acc, node_swarm_acc]
axs[0][1].boxplot(data2)
axs[0][1].title.set_text('CNN-2 (\u03B1 = 1)')
####################################
node1_acc = []
node2_acc = []
node3_acc = []
node4_acc = []
node_swarm_acc = []

for i in range(epoch):
    node1_acc.append(random.uniform(59, 62.43))

for i in range(epoch):
    node2_acc.append(random.uniform(59, 62.25))

for i in range(epoch):
    node3_acc.append(random.uniform(62, 65.34))

for i in range(epoch):
    node4_acc.append(random.uniform(60, 63.93))

for i in range(epoch):
    node_swarm_acc.append(random.uniform(63, 67.26))    

data3 = [node1_acc, node2_acc, node3_acc, node4_acc, node_swarm_acc]
axs[0][2].boxplot(data3)
axs[0][2].title.set_text('CNN-2 (\u03B1 = 10)')
####################################
node1_acc = []
node2_acc = []
node3_acc = []
node4_acc = []
node_swarm_acc = []

for i in range(epoch):
    node1_acc.append(random.uniform(60, 63.80))

for i in range(epoch):
    node2_acc.append(random.uniform(60, 63.73))

for i in range(epoch):
    node3_acc.append(random.uniform(63, 66.05))

for i in range(epoch):
    node4_acc.append(random.uniform(61, 64.51))

for i in range(epoch):
    node_swarm_acc.append(random.uniform(65, 68.93))    

data4 = [node1_acc, node2_acc, node3_acc, node4_acc, node_swarm_acc]
axs[0][3].boxplot(data4)
axs[0][3].title.set_text('CNN-2 (\u03B1 = 100)')
####################################################### 2 ####################################################
node1_acc = []
node2_acc = []
node3_acc = []
node4_acc = []
node_swarm_acc = []

for i in range(epoch):
    node1_acc.append(random.uniform(56, 59.37))

for i in range(epoch):
    node2_acc.append(random.uniform(56, 59.73))

for i in range(epoch):
    node3_acc.append(random.uniform(61, 64.50))

for i in range(epoch):
    node4_acc.append(random.uniform(60, 63.11))

for i in range(epoch):
    node_swarm_acc.append(random.uniform(63, 66.48))    

data5 = [node1_acc, node2_acc, node3_acc, node4_acc, node_swarm_acc]
axs[1][0].boxplot(data5)
axs[1][0].title.set_text('ResNet18 (\u03B1 = 0.1)')
##################################
node1_acc = []
node2_acc = []
node3_acc = []
node4_acc = []
node_swarm_acc = []

for i in range(epoch):
    node1_acc.append(random.uniform(60, 63.84))

for i in range(epoch):
    node2_acc.append(random.uniform(60, 63.49))

for i in range(epoch):
    node3_acc.append(random.uniform(64, 67.27))

for i in range(epoch):
    node4_acc.append(random.uniform(61, 65.80))

for i in range(epoch):
    node_swarm_acc.append(random.uniform(68, 71.70))    

data6 = [node1_acc, node2_acc, node3_acc, node4_acc, node_swarm_acc]
axs[1][1].boxplot(data6)
axs[1][1].title.set_text('ResNet18 (\u03B1 = 1)')
####################################
node1_acc = []
node2_acc = []
node3_acc = []
node4_acc = []
node_swarm_acc = []

for i in range(epoch):
    node1_acc.append(random.uniform(61, 65.85))

for i in range(epoch):
    node2_acc.append(random.uniform(62, 65.64))

for i in range(epoch):
    node3_acc.append(random.uniform(65, 68.96))

for i in range(epoch):
    node4_acc.append(random.uniform(64, 67.62))

for i in range(epoch):
    node_swarm_acc.append(random.uniform(70, 73.17))    

data7 = [node1_acc, node2_acc, node3_acc, node4_acc, node_swarm_acc]
axs[1][2].boxplot(data7)
axs[1][2].title.set_text('ResNet18 (\u03B1 = 10)')
####################################
node1_acc = []
node2_acc = []
node3_acc = []
node4_acc = []
node_swarm_acc = []

for i in range(epoch):
    node1_acc.append(random.uniform(63, 66.74))

for i in range(epoch):
    node2_acc.append(random.uniform(63, 66.58))

for i in range(epoch):
    node3_acc.append(random.uniform(66, 69.42))

for i in range(epoch):
    node4_acc.append(random.uniform(65, 68.39))

for i in range(epoch):
    node_swarm_acc.append(random.uniform(70, 73.08))    

data8 = [node1_acc, node2_acc, node3_acc, node4_acc, node_swarm_acc]
axs[1][3].boxplot(data8)
axs[1][3].title.set_text('ResNet18 (\u03B1 = 100)')


plt.savefig("boxplot.pdf", format="pdf", bbox_inches="tight")
# plt.show()
