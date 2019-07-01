import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py
import os
from datetime import datetime
import sys

if sys.platform == 'darwin':
  parentDir = '/Users/Danny/Dropbox/LLM_Danny/freqent/actin/'
if sys.platform == 'linux':
  parentDir = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/'

plt.close('all')
epr = []
exptype = []

with h5py.File(os.path.join(parentDir, datetime.today().strftime('%y%m%d') + '_epr.hdf5')) as f:
    files = list(f['epr'])
    epr = [f['epr'][file][()] for file in files]
    exptype = [f['epr'][file].attrs['experiment type'] for file in files]

df = pd.DataFrame({'epr': epr,
                   'type': exptype,
                   'exp': files})

fig, ax = plt.subplots(figsize=(6, 6))
sns.set(style='whitegrid')
sns.boxplot(y='epr', x='type', data=df, ax=ax, fliersize=0, palette='Set1', whis='range', linewidth=2.5)
sns.swarmplot(y='epr', x='type', data=df, color='0.2', linewidth=0, size=10)

# ax.set(ylim=[0, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel='', ylabel=r'$\langle ds/dt \rangle$')
ax.set(ylim=[-0.0001, 0.004], yticks=[0, 0.001, 0.002, 0.003, 0.004], xlabel='', ylabel=r'$\langle ds/dt \rangle$')
# ax.yaxis.grid(True)
sns.despine(trim=True, bottom=True)
fig.savefig(os.path.join(parentDir, datetime.today().strftime('%y%m%d') + '_actinEPRPlot.pdf'), format='pdf')

plt.show()
