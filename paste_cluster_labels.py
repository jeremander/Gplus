# Takes files listing cluster labels for each attribute and pastes them together into one file

import numpy as np
import pandas as pd
import optparse


def main():
    p = optparse.OptionParser()
    p.add_option('-p', type = str, help = 'PMI type (PMIs, NPMI1s, or NPMI2s)')
    p.add_option('-d', type = float, help = 'smoothing parameter')
    p.add_option('-e', type = str, help = 'embedding (adj, adj+diag, normlap, or regnormlap)')
    p.add_option('-k', type = int, help = 'number of eigenvalues')
    p.add_option('-s', action = 'store_true', default = False, help = 'normalize in sphere')
    p.add_option('-c', type = int, help = 'number of kmeans clusters')
    opts, args = p.parse_args()

    sim = opts.p
    delta = opts.d
    embedding = opts.e
    k = opts.k
    sphere = opts.s
    nclusts = opts.c
    attr_types = ['employer', 'major', 'places_lived', 'school']

    data_folder = 'gplus0_lcc/data/PMI/'
    filenames = [data_folder + '%s_%s_%s_delta%s_k%d%s_c%d_node_labels.csv' % (attr_type, sim, embedding, str(delta), k, '_normalized' if sphere else '', nclusts) for attr_type in attr_types]

    df = pd.DataFrame(columns = attr_types)

    for (attr_type, filename) in zip(attr_types, filenames):
        df[attr_type] = pd.read_csv(filename, header = None)[0]

    output_filename = data_folder + '%s_%s_delta%s_k%d%s_c%d_node_labels.csv' % (sim, embedding, str(delta), k, '_normalized' if sphere else '', nclusts)
    df.to_csv(output_filename, index = False)


if __name__ == "__main__":
    main()