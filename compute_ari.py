# Takes files listing cluster labels for each attribute and pastes them together into one file

import numpy as np
import pandas as pd
import optparse
from sklearn.metrics import adjusted_rand_score
from itertools import combinations


def main():
    p = optparse.OptionParser()
    p.add_option('-p', type = str, help = 'PMI type (PMIs, NPMI1s, or NPMI2s)')
    p.add_option('-e', type = str, help = 'embedding (eig, lap, normlap)')
    p.add_option('-d', type = float, help = 'smoothing parameter')
    p.add_option('-k', type = int, help = 'number of eigenvalues')
    p.add_option('-c', type = int, help = 'number of kmeans clusters')
    opts, args = p.parse_args()

    sim = opts.p
    embedding = opts.e
    delta = opts.d
    k = opts.k
    nclusts = opts.c
    attr_types = ['employer', 'major', 'places_lived', 'school']

    data_folder = 'gplus0_lcc/data/PMI/'
    report_folder = 'gplus0_lcc/reports/PMI/'
    filenames = [data_folder + '%s_%s_%s_delta%s_k%d_c%d_node_labels.csv' % (attr_type, sim, embedding, str(delta), k, nclusts) for attr_type in attr_types]

    df = pd.DataFrame(columns = attr_types)

    for (attr_type, filename) in zip(attr_types, filenames):
        df[attr_type] = pd.read_csv(filename, header = None)[0]

    ari_df = pd.DataFrame(columns = ['type1', 'type2', 'ARI'])
    for (i, (type1, type2)) in enumerate(combinations(attr_types, 2)):
        filtered_df = df[((df[type1] != -1) & (df[type2] != -1))]
        ari_df.loc[i] = [type1, type2, adjusted_rand_score(filtered_df[type1], filtered_df[type2])]

    output_filename = report_folder + '%s_delta%s_k%d_c%d_ARI.csv' % (sim, str(delta), k, nclusts)
    with open(output_filename, 'w') as f:
        f.write(ari_df.to_string(index = False))


if __name__ == "__main__":
    main()