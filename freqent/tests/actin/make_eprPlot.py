import matplotlib.pyplot as plt
import matplotlib as mpl
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
mpl.rcParams['pdf.fonttype'] = 42
epr = []
exptype = []

with h5py.File(os.path.join(parentDir, datetime.today().strftime('%y%m%d') + '_epr.hdf5')) as f:
    files = list(f.keys())[:-1]
    epr = [f[e + '/epr'][()] for e in files]
    exptype = [f[e].attrs['experiment type'] for e in files]

df = pd.DataFrame({'epr': epr,
                   'type': exptype,
                   'exp': files})

fig, ax = plt.subplots(figsize=(6, 6))
sns.set(style='whitegrid')
sns.stripplot(y='epr', x='type', data=df, color='k',
             size=10, alpha=0.8)
sns.pointplot(y='epr', x='type',
              data=df, ax=ax,
              palette='Set1', ci='sd',
              join=False, capsize=0.1)

# ax.set(ylim=[0, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel='', ylabel=r'$\langle ds/dt \rangle$')
ax.set(ylim=[-0.0001, 0.005], yticks=[0, 0.001, 0.002, 0.003, 0.004, 0.005], xlabel='', ylabel=r'$\langle ds/dt \rangle$')
# ax.yaxis.grid(True)
sns.despine(trim=True, bottom=True)
plt.tight_layout()
fig.savefig(os.path.join(parentDir, datetime.today().strftime('%y%m%d') + '_actinEPRPlot.pdf'), format='pdf')

plt.show()
