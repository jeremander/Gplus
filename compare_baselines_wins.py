# Reads in the max precisions for each of the baselines on a given parameter setting, then computes the number of wins for each baseline, aggregating over all experiments with a given set of parameters. Then plots a stacked bar graph

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import optparse
import itertools
from rstyle import *
from copy import deepcopy

ns = [50, 100, 200, 400, 800, 1600]
embedding = 'adj'
attr_types = ['employer', 'major', 'places_lived', 'school']
ranks = [25, 50, 100, 200, 400, 800]

all_colors = ['#e41a1c', '#eecd2e', '#e75522', '#ea9028', '#4daf4a', '#377eb8', 'purple', 'pink']
all_keys = ['content', 'context', 'max fusion', 'mean fusion', 'NPMI', 'NPMI+context', 'joint NPMI', 'random walk']
all_baselines = ['1', '2', '12_max', '12_mean', '3', '4', '5', '6']

baseline_indices = [0, 1, 3, 4, 5, 7]
colors = [all_colors[i] for i in baseline_indices]
keys = [all_keys[i] for i in baseline_indices]
baselines = [all_baselines[i] for i in baseline_indices]

assert (baselines[0] == '1')


def main():
    p = optparse.OptionParser()
    p.add_option('--max_count_features', '-m', type = int, default = 1000, help = 'max number of count features for baseline1')
    p.add_option('--sim', type = str, default = 'NPMI1s', help = 'similarity operation (PMIs, NPMI1s, prob)')
    p.add_option('--delta', '-d', type = float, default = 0.0, help = 'smoothing parameter')
    p.add_option('-k', type = int, default = 200, help = 'number of eigenvalues')
    p.add_option('--sphere', '-s', action = 'store_true', default = False, help = 'normalize in sphere')
    p.add_option('--thresh', '-t', type = float, default = -1.0, help = 'significance threshold')
    p.add_option('--save', '-v', action = 'store_true', default = False, help = 'save plot')

    opts, args = p.parse_args()

    max_count_features, sim, delta, k, sphere, thresh, save = opts.max_count_features, opts.sim, opts.delta, opts.k, opts.sphere, opts.thresh, opts.save

    selected_attrs = pd.read_csv('selected_attrs.csv')

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex = False, sharey = False, figsize = (12, 8), facecolor = 'white')
    axes = [ax0, ax1, ax2, ax3]

    for (ctr, attr_type) in enumerate(attr_types):
        print(attr_type)
        attrs_for_type = selected_attrs[selected_attrs['attributeType'] == attr_type]

        results_df = pd.DataFrame()
        results_df['baseline'] = [('baseline%s' % b) for b in baselines] * len(ns)
        results_df['n'] = list(itertools.chain(*[[n for b in baselines] for n in ns]))
        for rank in ranks:
            results_df[rank] = np.zeros(len(baselines) * len(ns), dtype = int)

        for (i, n) in enumerate(ns):
            for (attr, freq) in zip(attrs_for_type['attribute'], attrs_for_type['freq']):
                if (2 * n <= freq):
                    max_mean_prec_df = pd.DataFrame(columns = [('baseline%s' % b) for b in baselines])
                    for b in baselines:
                        if (b == '1'):
                            df = pd.read_csv('gplus0_lcc/baseline1/%s_%s_n%d_m%d_precision.csv' % (attr_type, attr, n, max_count_features))
                        elif (b[:2] == '12'):
                            df = pd.read_csv('gplus0_lcc/baseline%s/%s_%s_n%d_%s_k%d%s_m%d_precision.csv' % (b, attr_type, attr, n, embedding, k, '_normalize' if sphere else '', max_count_features))
                        elif (b == '6'):
                            df = pd.read_csv('gplus0_lcc/baseline6/%s_%s_n%d_%s_delta%s_precision.csv' % (attr_type, attr, n, sim, delta))
                        else:
                            df = pd.read_csv('gplus0_lcc/baseline%s/%s_%s_n%d_%s_k%d%s_precision.csv' % (b, attr_type, attr, n, embedding, k, '_normalize' if sphere else ''))
                        max_mean_prec_df['baseline%s' % b] = df['max_mean_prec']

                    for rank in ranks:
                        (best_index, best_prec) = max(enumerate(max_mean_prec_df.loc[rank - 1]), key = lambda pair : pair[1])
                        if (best_prec >= thresh):
                            results_df.ix[len(baselines) * i + best_index, rank] += 1

        results_agg_df = results_df.drop(['n'], axis = 1).groupby('baseline').sum()

        print(results_agg_df)

        ind = np.arange(len(ranks))
        width = 0.5
        plots = []
        plot = axes[ctr].bar(range(len(ranks)), results_agg_df.loc['baseline1'], width = width, color = colors[0])
        plots.append(plot)
        cumsums = deepcopy(np.asarray(results_agg_df.loc['baseline1']))
        for (b, c) in zip(baselines[1:], colors[1:]):
            plot = axes[ctr].bar(range(len(ranks)), results_agg_df.loc['baseline%s' % b], width = width, bottom = cumsums, color = c)
            plots.append(plot)
            cumsums += np.asarray(results_agg_df.loc['baseline%s' % b])
        axes[ctr].set_xlim((-0.5, len(ranks)))
        axes[ctr].set_title(attr_type.replace('_', ' '))

    plt.setp(axes, xticks = ind + width / 2, xticklabels = results_agg_df.columns, yticks = np.arange(0, 35, 5))
    fig.text(0.5, 0.04, 'rank', ha = 'center', fontsize = 14)
    fig.text(0.07, 0.5, 'wins', va = 'center', rotation = 'vertical', fontsize = 14)
    plt.figlegend(plots, keys, 'center')
    plt.suptitle("Relative performance of baselines", fontsize = 16, fontweight = 'bold')
    plt.subplots_adjust(wspace = 0.64, hspace = 0.58)
    for ax in axes:
        rstyle(ax)

    if save:
        filename = 'gplus0_lcc/compare/wins/m%d_%s_k%d_thresh%.3f%s_baseline%s_wins.png' % (max_count_features, embedding, k, thresh, '_normalize' if sphere else '', '_'.join(baselines))
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()

