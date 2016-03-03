# Performs vertex nomination on content + context via random walk approach on joint PMI matrices and/or social graph. 

done_import = False
while (not done_import):
    try:
        import optparse
        import matplotlib.pyplot as plt
        import sys
        import itertools
        from gplus import *
        from collections import defaultdict
        from scipy.sparse.linalg.isolve import bicg
        done_import = True
    except:
        pass

pd.options.display.max_rows = None
pd.options.display.width = 1000

topN_save = 1000    # number of precisions to save
topN_plot = 500     # number of precisions to plot
topN_nominees = 50  # number of nominees to include for top attribute analysis

attr_types = ['employer', 'major', 'places_lived', 'school']
max_steps = 8
steps_to_display = [1, 2, 4, 8]

#style = 'sequence'  # options are 'fusion', 'sequence', and 'mean'
style = 'sequence'

def zeronorm_dist(v1, v2):
    return (np.isclose(v1, 0.0) ^ np.isclose(v2, 0.0)).sum() / len(v1)

def main():
    p = optparse.OptionParser()
    p.add_option('--attr', '-a', type = str, help = 'attribute')
    p.add_option('--attr_type', '-t', type = str, help = 'attribute type')
    p.add_option('--num_train_each', '-n', type = int, help = 'number of training samples of True and False for the attribute (for total of 2n training samples)')
    p.add_option('--sim', '-s', type = str, default = 'NPMI1s', help = 'similarity operation (PMIs, NPMI1s, prob)')
    p.add_option('--delta', '-d', type = float, default = 0.0, help = 'smoothing parameter')
    p.add_option('--num_samples', '-S', type = int, default = 50, help = 'number of Monte Carlo samples')
    p.add_option('-v', action = 'store_true', default = False, help = 'save plot')
    opts, args = p.parse_args()

    attr, attr_type, num_train_each, sim, delta, num_samples, save_plot = opts.attr, opts.attr_type, opts.num_train_each, opts.sim, opts.delta, opts.num_samples, opts.v

    steps = list(range(1, max_steps + 1))
    cmap = plt.cm.gist_ncar
    colors = {step : cmap(int((i + 1) * cmap.N / (len(steps_to_display) + 1.0))) for (i, step) in enumerate(steps_to_display)}
    flags = ['content', 'context', 'both'] if (style == 'fusion') else ['both']
    linestyles = ['dotted', 'dashed', 'solid'] if (style == 'fusion') else ['solid']

    folder = 'gplus0_lcc/baseline6/'
    zeronorm_filename = folder + '%s_%s_n%d_%s_delta%s_zeronorm.csv' % (attr_type, attr, num_train_each, sim, str(delta))
    agg_precision_filename = folder + '%s_%s_n%d_%s_delta%s_precision.csv' % (attr_type, attr, num_train_each, sim, str(delta))
    zeronorm_plot_filename = folder + '%s_%s_n%d_%s_delta%s_zeronorm.png' % (attr_type, attr, num_train_each, sim, str(delta))
    plot_filename = folder + '%s_%s_n%d_%s_delta%s_precision.png' % (attr_type, attr, num_train_each, sim, str(delta))

    print_flush("\nNominating nodes with whose '%s' attribute is '%s' (%d pos/neg seeds)..." % (attr_type, attr, num_train_each))
    print_flush("\nLoading AttributeAnalyzer...")
    a = AttributeAnalyzer(load_data = False)
    other_attr_types = [at for at in a.attr_types if (at != attr_type)]
    sim_names = other_attr_types + ['social_graph']
    sqrt_samples = np.sqrt(num_samples)

    try:
        mean_zeronorm_df = pd.read_csv(zeronorm_filename)
        agg_precision_df = pd.read_csv(agg_precision_filename)
        print_flush("\nLoaded data from '%s'." % agg_precision_filename)
        selected_attrs = pd.read_csv('selected_attrs.csv')
        if (attr in list(selected_attrs['attribute'])):
            row = selected_attrs[selected_attrs['attribute'] == attr].iloc[0]
            num_true_in_test = row['freq'] - num_train_each
            num_test = row['totalKnown'] - 2 * num_train_each
        else:
            ind = a.get_attribute_indicator(attr, attr_type)
            num_true_in_test = len(ind[ind == 1]) - num_train_each
            num_test = ind.count() - 2 * num_train_each

    except OSError:
        print_flush("\nLoading similarity operators...")
        random_walk_ops = []
        for at in attr_types:
            if (at != attr_type):
                a.make_random_walk_operator(at, sim = sim, delta = delta, load = True, save = True)
                random_walk_ops.append(a.random_walk_operators[at])
        g = Gplus()
        g.load_sparse_adjacency_operator()
        random_walk_ops.append((ConstantDiagonalLinearOperator(a.num_vertices, 1.0) + g.sparse_adjacency_operator).to_column_stochastic())

        # get attribute indicator for all the nodes
        attr_indicator = a.get_attribute_indicator(attr, attr_type)

        precision_dfs = {(step, flag) : pd.DataFrame(columns = range(num_samples)) for (step, flag) in itertools.product(steps, flags)}
        zeronorm_dfs = {sim_name : pd.DataFrame(columns = range(num_samples)) for sim_name in sim_names}
        zeronorm_df = pd.DataFrame(columns = range(num_samples))

        for s in range(num_samples):
            print_flush("\nSEED = %d" % s)
            np.random.seed(s)
            (training_true, training_false, test) = a.get_attribute_sample(attr, attr_type, num_train_each)
            test_df = pd.DataFrame()
            test_out = attr_indicator[test]
            test_df['test'] = test_out
            if (style == 'fusion'):
                for (rw, sim_name) in zip(random_walk_ops, sim_names):
                    x_plus, x_minus = np.zeros((a.num_vertices, max_steps + 1)), np.zeros((a.num_vertices, max_steps + 1))
                    p = 1.0 / num_train_each
                    for (i, j) in zip(training_true, training_false):  # initial states in the Markov chain
                        x_plus[i, 0] = p
                        x_minus[j, 0] = p
                    print_flush("Stepping random walk (%s)..." % sim_name)
                    for t in range(max_steps):
                        x_plus[:, t + 1] = rw * x_plus[:, t]
                        x_minus[:, t + 1] = rw * x_minus[:, t]
                    scores = x_plus - x_minus  # scores for each step
                    zeronorm_dfs[sim_name][s] = [zeronorm_dist(scores[:, t + 1], scores[:, t]) for t in range(max_steps)] # zero-norm distances of successive score vectors
                    # turn score vectors into cumulative averages
                    for t in range(1, max_steps + 1):
                        test_df[(t, sim_name)] = scores[test, 1 : t + 1].mean(axis = 1)
                for (step, flag) in itertools.product(steps, flags):
                    cols = [] + (other_attr_types if (flag != 'context') else []) + (['social_graph'] if (flag != 'content') else [])
                    cols = [(step, sim_name) for sim_name in cols]
                    test_df[(step, flag)] = test_df[cols].sum(axis = 1)
            elif (style == 'sequence'):
                x_plus, x_minus = np.zeros((a.num_vertices, max_steps + 1)), np.zeros((a.num_vertices, max_steps + 1))
                p = 1.0 / num_train_each
                for (i, j) in zip(training_true, training_false):  # initial states in the Markov chain
                    x_plus[i, 0] = p
                    x_minus[j, 0] = p
                print_flush("Stepping random walk...")
                for t in range(max_steps):
                    x_plus_cur = x_plus[:, t]
                    x_minus_cur = x_minus[:, t]
                    for (rw, sim_name) in zip(random_walk_ops, sim_names):
                        x_plus_cur = rw * x_plus_cur
                        x_minus_cur = rw * x_minus_cur
                    x_plus[:, t + 1] = x_plus_cur
                    x_minus[:, t + 1] = x_minus_cur
                scores = x_plus - x_minus  # scores for each step
                zeronorm_df[s] = [zeronorm_dist(scores[:, t + 1], scores[:, t]) for t in range(max_steps)]
                for t in range(1, max_steps + 1):
                    test_df[(t, 'both')] = scores[test, 1 : t + 1].mean(axis = 1)
            else:  # style == 'mean'
                x_plus, x_minus = np.zeros((a.num_vertices, max_steps + 1)), np.zeros((a.num_vertices, max_steps + 1))
                p = 1.0 / num_train_each
                for (i, j) in zip(training_true, training_false):  # initial states in the Markov chain
                    x_plus[i, 0] = p
                    x_minus[j, 0] = p
                rw_sum = (1.0 / len(random_walk_ops)) * reduce(lambda x, y : x + y, random_walk_ops)
                print_flush("Stepping random walk...")
                for t in range(max_steps):
                    x_plus[:, t + 1] = rw_sum * x_plus[:, t]
                    x_minus[:, t + 1] = rw_sum * x_minus[:, t]
                scores = x_plus - x_minus  # scores for each step
                zeronorm_df[s] = [zeronorm_dist(scores[:, t + 1], scores[:, t]) for t in range(max_steps)]
                for t in range(1, max_steps + 1):
                    test_df[(t, 'both')] = scores[test, 1 : t + 1].mean(axis = 1)
            # do vertex nomination
            for (step, flag) in itertools.product(steps, flags):
                test_df = test_df.sort_values(by = (step, flag), ascending = False)
                precision_dfs[(step, flag)][s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test) + 1.0)

        # compute means and standard errors over all the samples
        agg_precision_df = pd.DataFrame(columns = list(itertools.product(['mean', 'stderr'], steps, flags)))
        for (step, flag) in itertools.product(steps, flags):
            agg_precision_df[('mean', step, flag)] = precision_dfs[(step, flag)].mean(axis = 1)
            agg_precision_df[('stderr', step, flag)] = precision_dfs[(step, flag)].std(axis = 1) / sqrt_samples

        # compute mean zero-norm distances of successive score vectors
        if (style == 'fusion'):
            mean_zeronorm_df = pd.DataFrame(columns = sim_names)
            for sim_name in sim_names:
                mean_zeronorm_df[sim_name] = zeronorm_dfs[sim_name].mean(axis = 1)
        else:
            mean_zeronorm_df = pd.DataFrame(columns = ['both'])
            mean_zeronorm_df['both'] = zeronorm_df.mean(axis = 1)
        mean_zeronorm_df.to_csv(zeronorm_filename, index = False)

        # save the aggregate data frames
        N_save = min(len(test_out), topN_save)
        agg_precision_df = agg_precision_df[:N_save]

        agg_precision_df.to_csv(agg_precision_filename, index = False)

        num_true_in_test = test_out.sum()
        num_test = len(test_out)


    if save_plot:
        # plot the nomination precision 
        plt.clf()
        N_plot = min(len(agg_precision_df), topN_plot)
        plots = []

        agg_precision_df.columns = [str(col) for col in agg_precision_df.columns]

        for step in steps_to_display:
            for (j, flag) in enumerate(flags):
                plt.fill_between(agg_precision_df.index, agg_precision_df[str(('mean', step, flag))] - 2 * agg_precision_df[str(('stderr', step, flag))], agg_precision_df[str(('mean', step, flag))] + 2 * agg_precision_df[str(('stderr', step, flag))], color = colors[step], alpha = 0.1)
                plot, = plt.plot(agg_precision_df.index, agg_precision_df[str(('mean', step, flag))], color = colors[step], linewidth = 2, linestyle = linestyles[j], label = ','.join([str(step), flag]))
                if ((step == 1) or (flag == 'both')):
                    plots.append(plot)

        guess_rate = num_true_in_test / num_test
        guess, = plt.plot([guess_rate for i in range(N_plot)], linestyle = 'dashed', linewidth = 2, color = 'black', label = 'Guess')
        plt.xlabel('rank')
        plt.ylabel('precision')
        plt.xlim((0.0, N_plot))
        plt.ylim((0.0, 1.0))
        plt.title('Vertex Nomination Precision')
        plt.legend(handles = plots + [guess])
        plt.savefig(plot_filename)

        # plot successive zero-norm distances for each random walk state
        plt.clf()
        plots = []
        if (style == 'fusion'):
            sim_name_colors = {'employer' : 'blue', 'major' : 'red', 'places_lived' : 'orange', 'school' : 'purple', 'social_graph' : 'green'}
            for sim_name in sim_names:
                plot, = plt.plot(mean_zeronorm_df.index + 1, mean_zeronorm_df[sim_name], color = sim_name_colors[sim_name], linewidth = 2, label = sim_name)
                plots.append(plot)
        else:
            plot, = plt.plot(mean_zeronorm_df.index + 1, mean_zeronorm_df['both'], linewidth = 2)
            plots.append(plot)

        plt.xlabel('steps')
        plt.ylabel('score change (zero-norm)')
        plt.xlim((1, max_steps))
        plt.title('Convergence of Random Walks')
        plt.legend(handles = plots)
        plt.savefig(zeronorm_plot_filename)


    print("\nDone!")

if __name__ == "__main__":
    main()

