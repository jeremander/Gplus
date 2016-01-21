# reads in the max precisions for each of the baselines on a given parameter setting, then plots them against each other

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import optparse
from rstyle import *
from ggplot import *


#colors = ['#000000', '#FF7E75', '#E5CD2E', '#10CA48', '#10CFD4']
colors = ['#000000', '#e41a1c', '#eecd2e', '#4daf4a', '#377eb8']

def main():
    p = optparse.OptionParser()
    p.add_option('--attr', '-a', type = str, help = 'attribute')
    p.add_option('--attr_type', '-t', type = str, help = 'attribute type')
    p.add_option('--num_train_each', '-n', type = int, help = 'number of training samples of True and False for the attribute (for total of 2n training samples)')
    p.add_option('--max_count_features', '-m', type = int, default = 1000, help = 'max number of count features for baseline1')
    p.add_option('--embedding', '-e', type = str, default = 'adj', help = 'embedding (adj, adj+diag, normlap, regnormlap)')
    p.add_option('-k', type = int, default = 200, help = 'number of eigenvalues')
    p.add_option('--sphere', '-s', action = 'store_true', default = False, help = 'normalize in sphere')
    p.add_option('-v', action = 'store_true', default = False, help = 'save results')
    p.add_option('-N', type = int, default = 500, help = 'top N precisions to display')

    opts, args = p.parse_args()

    attr, attr_type, num_train_each, max_count_features, embedding, k, sphere, save, N = opts.attr, opts.attr_type, opts.num_train_each, opts.max_count_features, opts.embedding, opts.k, opts.sphere, opts.v, opts.N

    max_mean_prec_df = pd.DataFrame(columns = ['rank'] + [('baseline%d' % i) for i in range(1, 5)])

    for i in range(1, 5):
        if (i == 1):
            df = pd.read_csv('gplus0_lcc/baseline1/%s_%s_n%d_m%d_precision.csv' % (attr_type, attr, num_train_each, max_count_features))
        else:
            df = pd.read_csv('gplus0_lcc/baseline%d/%s_%s_n%d_%s_k%d%s_precision.csv' % (i, attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else ''))
        max_mean_prec_df['baseline%d' % i] = df['max_mean_prec'][:N]

    selected_attrs = pd.read_csv('selected_attrs.csv')
    row = selected_attrs[selected_attrs['attribute'] == attr].iloc[0]
    num_true_in_test = row['freq'] - num_train_each
    num_test = row['totalKnown'] - 2 * num_train_each
    guess_rate = num_true_in_test / num_test
    max_mean_prec_df['guess'] = guess_rate

    fig = plt.figure(figsize = (12, 8), facecolor = 'white')
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    plots = []
    axes = []
    plots.append(ax.plot(max_mean_prec_df.index, max_mean_prec_df['guess'], color = colors[0], linewidth = 4, linestyle = 'dashed')[0])
    plots[-1].set_dash_capstyle('projecting')
    axes.append(plt.gca())
    for i in range(1, 5):
        plots.append(ax.plot(max_mean_prec_df.index, max_mean_prec_df['baseline%d' % i], color = colors[i], linewidth = 4)[0])
        axes.append(plt.gca())
    plt.xlabel('rank', fontsize = 14, labelpad = 8)
    plt.ylabel('precision', fontsize = 14, labelpad = 12)
    plt.title("Best nomination precision\n%s: %s" % (attr_type.replace('_', ' '), attr), fontsize = 16, fontweight = 'bold', y = 1.02)
    plt.setp(axes, xticks = np.arange(0, N + 1, 100))#, yticks = np.arange(0, 1.1, 0.25))
    #plt.ylim((0.0, 1.0))
    plt.legend(plots, ['guess', 'content', 'context', 'NPMI', 'NPMI+context'], loc = (1.01, 0.5))
    for a in axes:
        rstyle(a)

    filename = 'gplus0_lcc/compare/prec/%s_%s_n%d_m%d_%s_k%d%s_max_mean_prec.png' % (attr, attr_type, num_train_each, max_count_features, embedding, k, '_normalize' if sphere else '')

    if save:
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()
