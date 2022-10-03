# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:55:31 2022

@author: hussain
"""

import os
import pandas as pd

output_path = ''
columns = ['loss', 'accuracy', 'f1_score']  
data = pd.DataFrame(columns = columns)
data = data.append({
'loss': 0.0245,
'accuracy': 0.9980,
'f1_score': 0.9947,
},ignore_index=True)
data.to_csv(os.path.join(output_path, 'log.csv'))