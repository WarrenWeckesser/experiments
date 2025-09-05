
import numpy as np
from scipy.stats import levene

dt = np.dtype([('program', 'S1'), ('loss', float)])
data = np.genfromtxt('data.csv', dtype=dt, skip_header=1, delimiter=',')

program = data['program']
loss = data['loss']
maska = program == b'A'
maskb = program == b'B'
maskc = program == b'C'

groupa = loss[maska]
groupb = loss[maskb]
groupc = loss[maskc]

result = levene(groupa, groupb, groupc, center='trimmed', proportiontocut=0.2)
print(result)
