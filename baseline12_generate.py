import subprocess
import os.path
import pandas as pd 

selected_attrs = pd.read_csv('selected_attrs.csv')

n_vals = [50, 100, 200, 400, 800, 1600]
embedding_vals = ['adj']
k_vals = [50, 100, 200]
sphere_vals = [True]
max_count_features_vals = [1000]

num_samples = 50
jobs = 1
memory = 8  # number of GB
max_folder = 'gplus0_lcc/baseline12_max/'
mean_folder = 'gplus0_lcc/baseline12_mean/'

plot = True         # convert csv to png (but only if csv exists)
write = True        # write csv's that do not exist
overwrite = False   # rewrite csv's that already exist
grid = False         # issue the generation commands in parallel on the grid

assert (not (overwrite and (not write)))
assert (not (plot and grid))

for sphere in sphere_vals:
    for k in k_vals:
        for embedding in embedding_vals:
            for max_count_features in max_count_features_vals:
                for n in n_vals:
                    for (attr, attr_type) in zip(selected_attrs['attribute'], selected_attrs['attributeType']):
                        if (selected_attrs[(selected_attrs['attribute'] == attr) & (selected_attrs['attributeType'] == attr_type)]['freq'].iloc[0] >= 2 * n):
                            max_csv_filename = max_folder + '%s_%s_n%d_%s_k%d%s_m%d_precision.csv' % (attr_type, attr, n, embedding, k, '_normalize' if sphere else '', max_count_features)
                            max_plot_filename = max_folder + '%s_%s_n%d_%s_k%d%s_m%d_precision.png' % (attr_type, attr, n, embedding, k, '_normalize' if sphere else '', max_count_features)
                            mean_csv_filename = mean_folder + '%s_%s_n%d_%s_k%d%s_m%d_precision.csv' % (attr_type, attr, n, embedding, k, '_normalize' if sphere else '', max_count_features)
                            mean_plot_filename = mean_folder + '%s_%s_n%d_%s_k%d%s_m%d_precision.png' % (attr_type, attr, n, embedding, k, '_normalize' if sphere else '', max_count_features)
                            if overwrite:
                                subprocess.check_call(['rm', '-f', max_csv_filename, max_plot_filename, mean_csv_filename, mean_plot_filename])
                            if (write or (plot and os.path.isfile(max_csv_filename) and (not os.path.isfile(max_plot_filename)))):
                                if grid:
                                    safe_attr = '_'.join(attr.split())
                                    subcmd = "python3 baseline12.py -a '%s' -t '%s' -n %d -e %s -k %d %s -m %d -S %d -j %d" % (attr, attr_type, n, embedding, k, '-s' if sphere else '', max_count_features, num_samples, jobs)
                                    cmd = 'qsub -q all.q -l num_proc=%d,mem_free=%dG,h_rt=72:00:00 -b Y -V -cwd -j yes -o . -N baseline12_%s_%s_n%d_%s_k%d%s_m%d "%s"' % (jobs, memory, safe_attr, attr_type, n, embedding, k, '_normalize' if sphere else '', max_count_features, subcmd)
                                    print(cmd)
                                    subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
                                else:
                                    subprocess.check_call(['python3', 'baseline12.py', '-a', attr, '-t', attr_type, '-n', str(n), '-e', embedding, '-k', str(k), '-m', str(max_count_features), '-S', str(num_samples), '-v' if plot else '', '-j', str(jobs)] + (['-s'] if sphere else []))