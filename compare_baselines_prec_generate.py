import subprocess
import os.path
import pandas as pd 

selected_attrs = pd.read_csv('selected_attrs.csv')

n_vals = [50, 100, 200, 400, 800, 1600]
m_vals = [1000]
embedding_vals = ['adj']
k_vals = [200]
sphere_vals = [True]
N = 500


for sphere in sphere_vals:
    for k in k_vals:
        for embedding in embedding_vals:
            for m in m_vals:
                for n in n_vals:
                    for (attr_type, attr, freq) in zip(selected_attrs['attributeType'], selected_attrs['attribute'], selected_attrs['freq']):
                        if (2 * n <= freq):
                            subprocess.check_call(['python3', 'compare_baselines_prec.py', '-a', attr, '-t', attr_type, '-n', str(n), '-m', str(m), '-e', embedding, '-k', str(k), '-s' if sphere else '', '-v', '-N', str(N)])