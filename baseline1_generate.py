import subprocess
import pandas as pd 

selected_attrs = pd.read_csv('selected_attrs.csv')

#n_vals = [50, 100, 200, 400, 800, 1600]
n_vals = [1600, 800, 400, 200, 100, 50]
max_count_features_vals = [500, 1000]
num_estimators = 1000
num_samples = 50
jobs = 8

for max_count_features in max_count_features_vals:
    for n in n_vals:
        for (attr, attr_type) in zip(selected_attrs['attribute'], selected_attrs['attributeType']):
            if (selected_attrs[(selected_attrs['attribute'] == attr) & (selected_attrs['attributeType'] == attr_type)]['freq'].iloc[0] >= 2 * n):
                safe_attr = '_'.join(attr.split())
                subcmd = "python3 baseline1.py -a '%s' -t '%s' -n %d -m %d -e %d -S %d -j %d" % (attr, attr_type, n, max_count_features, num_estimators, num_samples, jobs)
                cmd = 'qsub -q intel.q -l num_proc=%d,mem_free=16G,h_rt=72:00:00 -b Y -V -cwd -j yes -o . -N %s_%s_n%d_m%d "%s"' % (jobs, safe_attr, attr_type, n, max_count_features, subcmd)
                #print(cmd)
                #subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
                subprocess.check_call(['python3', 'baseline1.py', '-a', attr, '-t', attr_type, '-n', str(n), '-m', str(max_count_features), '-e', str(num_estimators), '-S', str(num_samples), '-j', str(jobs)])