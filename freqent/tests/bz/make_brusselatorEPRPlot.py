import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import freqent.freqent as fe

folder = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/190128'

alpha = np.asarray([float(x.split('alpha')[-1].split('_')[0]) for x in glob(os.path.join(folder, '*/'))])

epr = np.zeros(len(alpha))
epr_spectral = np.zeros(len(alpha))
for fInd, f in enumerate(glob(os.path.join(folder, '*/'))):
    with open(os.path.join(f, 'data.pickle'), 'rb') as d:
        data = pickle.load(d)
        epr[fInd] = data['epr']
        epr_spectral[fInd] = data['epr_spectral']

fig, ax = plt.subplots()
ax.loglog(alpha, epr, '.', label='True EPR')
ax.loglog(alpha, epr_spectral, '.', label='Spectral EPR')
ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{\Sigma}$')
plt.legend()

plt.show()
