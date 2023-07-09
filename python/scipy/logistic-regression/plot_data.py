import matplotlib.pyplot as plt
import numpy as np

admit, gre, gpa, rank = np.loadtxt('binary.csv', delimiter=',', skiprows=1,
                                   unpack=True)
admit_mask = admit == 1

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(gre[admit_mask], gpa[admit_mask], rank[admit_mask], marker='o')
ax.scatter(gre[~admit_mask], gpa[~admit_mask], rank[~admit_mask], marker='^')

ax.set_xlabel('GRE')
ax.set_ylabel('GPA')
ax.set_zlabel('Rank')

plt.show()
