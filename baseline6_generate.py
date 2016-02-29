import subprocess
import os.path
import pandas as pd 

selected_attrs = pd.read_csv('selected_attrs.csv')

n_vals = [50, 100, 200, 400, 800, 1600]
sim_vals = ['NPMI1s']
delta_vals = [0.0]

num_samples = 50
jobs = 1
memory = 32  # number of GB
folder = 'gplus0_lcc/baseline6/'

plot = True        # convert csv to png (but only if csv exists)
write = True        # write csv's that do not exist
overwrite = False   # rewrite csv's that already exist
grid = False         # issue the generation commands in parallel on the grid

assert (not (overwrite and (not write)))
assert (not (plot and grid))

for delta in delta_vals:
    for sim in sim_vals:
        for n in n_vals:
            for (attr, attr_type) in zip(selected_attrs['attribute'], selected_attrs['attributeType']):
                if (selected_attrs[(selected_attrs['attribute'] == attr) & (selected_attrs['attributeType'] == attr_type)]['freq'].iloc[0] >= 2 * n):
                    csv_filename = folder + '%s_%s_n%d_%s_delta%s_precision.csv' % (attr_type, attr, n, sim, str(delta))
                    plot_filename = folder + '%s_%s_n%d_%s_delta%s_precision.png' % (attr_type, attr, n, sim, str(delta))
                    if overwrite:
                        subprocess.check_call(['rm', '-f', csv_filename, plot_filename])
                    if (write or (plot and os.path.isfile(csv_filename) and (not os.path.isfile(plot_filename)))):
                        if grid:
                            safe_attr = '_'.join(attr.split())
                            subcmd = "python3 baseline6.py -a '%s' -t '%s' -n %d -s %s -d %s -S %d" % (attr, attr_type, n, sim, str(delta), num_samples)
                            cmd = 'qsub -q all.q -l num_proc=%d,mem_free=%dG,h_rt=24:00:00 -b Y -V -cwd -j yes -o . -N baseline6_%s_%s_n%d_%s_delta%s "%s"' % (jobs, memory, safe_attr, attr_type, n, sim, str(delta), subcmd)
                            print(cmd)
                            subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
                        else:
                            subprocess.check_call(['python3', 'baseline6.py', '-a', attr, '-t', attr_type, '-n', str(n), '-s', sim, '-d', str(delta), '-S', str(num_samples), '-v' if plot else ''])