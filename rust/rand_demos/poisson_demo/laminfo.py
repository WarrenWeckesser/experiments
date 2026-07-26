import numpy as np

lam = 1e14

sp = np.spacing(lam)
std = np.sqrt(lam)
ratio = std/sp

print(f'{lam = }')
print(f'spacing(lam) = {sp}')
print(f'sqrt(lam) = {std}')
print(f'{ratio = }')
