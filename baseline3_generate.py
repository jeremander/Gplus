import subprocess
import pandas as pd 

selected_attrs = pd.read_csv('selected_attrs.csv')

n_vals = [50, 100, 200, 400, 800, 1600]
embedding_vals = ['adj']
k_vals = [50, 100, 200]
sphere_vals = [False, True]

num_samples = 50
jobs = 1

for sphere in sphere_vals:
    for k in k_vals:
        for embedding in embedding_vals:
            for n in n_vals:
                for (attr, attr_type) in zip(selected_attrs['attribute'], selected_attrs['attributeType']):
                    if (selected_attrs[(selected_attrs['attribute'] == attr) & (selected_attrs['attributeType'] == attr_type)]['freq'].iloc[0] >= 2 * n):
                        safe_attr = '_'.join(attr.split())
                        subcmd = "python3 baseline3.py -a '%s' -t '%s' -n %d -e %s -k %d %s -S %d -j %d" % (attr, attr_type, n, embedding, k, '-s' if sphere else '', num_samples, jobs)
                        cmd = 'qsub -q all.q -l num_proc=%d,mem_free=4G,h_rt=24:00:00 -b Y -V -cwd -j yes -o . -N baseline3_%s_%s_n%d_%s_k%d%s "%s"' % (jobs, safe_attr, attr_type, n, embedding, k, '_normalize' if sphere else '', subcmd)
                        print(cmd)
                        subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
                        #subprocess.check_call(['python3', 'baseline3.py', '-a', attr, '-t', attr_type, '-n', str(n), '-e', embedding, '-k', str(k), '-S', str(num_samples), '-j', str(jobs)] + (['-s'] if sphere else []))