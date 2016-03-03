# reads in the max precisions for each of the baselines on a given parameter setting, then plots them against each other

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import optparse
from rstyle import *
from ggplot import *


all_colors = ['#e41a1c', '#eecd2e', '#e75522', '#ea9028', '#4daf4a', '#377eb8', 'purple', 'pink']
all_keys = ['content', 'context', 'max fusion', 'mean fusion', 'NPMI', 'NPMI+context', 'joint NPMI', 'random walk']
all_linestyles = ['solid', 'solid', 'dashed', 'solid', 'solid', 'solid', 'solid', 'solid']
all_baselines = ['1', '2', '12_max', '12_mean', '3', '4', '5', '6']

baseline_indices = [0, 1, 3, 4, 5, 6, 7]
colors = [all_colors[i] for i in baseline_indices]
keys = [all_keys[i] for i in baseline_indices]
linestyles = [all_linestyles[i] for i in baseline_indices]
baselines = [all_baselines[i] for i in baseline_indices]


def main():
    p = optparse.OptionParser()
    p.add_option('--attr', '-a', type = str, help = 'attribute')
    p.add_option('--attr_type', '-t', type = str, help = 'attribute type')
    p.add_option('--num_train_each', '-n', type = int, help = 'number of training samples of True and False for the attribute (for total of 2n training samples)')
    p.add_option('--max_count_features', '-m', type = int, default = 1000, help = 'max number of count features for baseline1')
    p.add_option('--embedding', '-e', type = str, default = 'adj', help = 'embedding (adj, adj+diag, normlap, regnormlap)')
    p.add_option('--sim', type = str, default = 'NPMI1s', help = 'similarity operation (PMIs, NPMI1s, prob)')
    p.add_option('--delta', '-d', type = float, default = 0.0, help = 'smoothing parameter')
    p.add_option('-k', type = int, default = 200, help = 'number of eigenvalues')
    p.add_option('--sphere', '-s', action = 'store_true', default = False, help = 'normalize in sphere')
    p.add_option('--steps', type = int, default = 1, help = 'number of random walk steps')
    p.add_option('--rwcontext', action = 'store_true', default = False, help = 'use context only in random walk')
    p.add_option('-v', action = 'store_true', default = False, help = 'save results')
    p.add_option('-N', type = int, default = 500, help = 'top N precisions to display')

    opts, args = p.parse_args()

    attr, attr_type, num_train_each, max_count_features, embedding, sim, delta, k, sphere, steps, rwcontext, save, N = opts.attr, opts.attr_type, opts.num_train_each, opts.max_count_features, opts.embedding, opts.sim, opts.delta, opts.k, opts.sphere, opts.steps, opts.rwcontext, opts.v, opts.N

    max_mean_prec_df = pd.DataFrame(columns = ['rank'] + [('baseline%s' % b) for b in baselines])

    for b in baselines:
        if (b == '1'):
            df = pd.read_csv('gplus0_lcc/baseline1/%s_%s_n%d_m%d_precision.csv' % (attr_type, attr, num_train_each, max_count_features))
            max_mean_prec_df['baseline1'] = df['mean_logreg_prec'][:N]  # logistic regression only
        elif (b[:2] == '12'):
            df = pd.read_csv('gplus0_lcc/baseline%s/%s_%s_n%d_%s_k%d%s_m%d_precision.csv' % (b, attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '', max_count_features))
            max_mean_prec_df['baseline%s' % b] = df['mean_logreg_prec'][:N]  # logistic regression only
        elif (b == '6'):
            df = pd.read_csv('gplus0_lcc/baseline6/%s_%s_n%d_%s_delta%s_precision.csv' % (attr_type, attr, num_train_each, sim, delta))
            filename = 'gplus0_lcc/baseline6/%s_%s_n%d_%s_delta%s_precision.csv' % (attr_type, attr, num_train_each, sim, delta)
            if ('max_mean_prec' not in df.columns):
                cols = [col for col in df.columns if 'mean' in col]
                df['max_mean_prec'] = df[cols].max(axis = 1)
                df.to_csv(filename, index = False)
            max_mean_prec_df['baseline6'] = df[str(('mean', steps, 'context' if rwcontext else 'both'))][:N]
        else:
            df = pd.read_csv('gplus0_lcc/baseline%s/%s_%s_n%d_%s_k%d%s_precision.csv' % (b, attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else ''))
            filename = 'gplus0_lcc/baseline%s/%s_%s_n%d_%s_k%d%s_precision.csv' % (b, attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '')
            if ('max_mean_prec' not in df.columns):
                df['max_mean_prec'] = df[['mean_rfc_prec', 'mean_boost_prec', 'mean_logreg_prec', 'mean_gnb_prec']].max(axis = 1)
                df.to_csv(filename, index = False)
            max_mean_prec_df['baseline%s' % b] = df['mean_logreg_prec'][:N]  # logistic regression only
        #max_mean_prec_df['baseline%s' % b] = df['max_mean_prec'][:N]  # max of all systems

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
    plots.append(ax.plot(max_mean_prec_df.index, max_mean_prec_df['guess'], color = 'black', linewidth = 4, linestyle = 'dashed')[0])
    plots[-1].set_dash_capstyle('projecting')
    axes.append(plt.gca())
    for (b, c, sty) in zip(baselines, colors, linestyles):
        plots.append(ax.plot(max_mean_prec_df.index, max_mean_prec_df['baseline%s' % b], color = c, linestyle = sty, linewidth = 4)[0])
        if (sty == 'dashed'):
            plots[-1].set_dash_capstyle('projecting')
        axes.append(plt.gca())
    plt.xlabel('rank', fontsize = 14, labelpad = 8)
    plt.ylabel('precision', fontsize = 14, labelpad = 12)
    plt.title("Best nomination precision\n%s: %s" % (attr_type.replace('_', ' '), attr), fontsize = 16, fontweight = 'bold', y = 1.02)
    plt.setp(axes, xticks = np.arange(0, N + 1, 100))#, yticks = np.arange(0, 1.1, 0.25))
    #plt.ylim((0.0, 1.0))
    plt.legend(plots, ['guess'] + keys, loc = (1.01, 0.5))
    for a in axes:
        rstyle(a)

    filename = 'gplus0_lcc/compare/prec/%s_%s_n%d_m%d_%s_%s_delta%s_k%d%s_steps%d%s_mean_prec.png' % (attr, attr_type, num_train_each, max_count_features, embedding, sim, str(delta), k, '_normalize' if sphere else '', steps, '_rwcontext' if rwcontext else '')

    if save:
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()

