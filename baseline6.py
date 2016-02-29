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
#sim = 'NPMI1s'  # use NPMI as attribute similarity measure
#max_steps = 32
#steps_to_display = [1, 2, 4, 8, 16, 32]

def main():
    p = optparse.OptionParser()
    p.add_option('--attr', '-a', type = str, help = 'attribute')
    p.add_option('--attr_type', '-t', type = str, help = 'attribute type')
    p.add_option('--num_train_each', '-n', type = int, help = 'number of training samples of True and False for the attribute (for total of 2n training samples)')
    #p.add_option('--mean', '-m', type = str, default = 'arith', help = 'type of mean for averaging probabilities (arith, geom)')
    #p.add_option('--use_graph', '-g', action = 'store_true', default = False, help = 'use the social graph')
    p.add_option('--sim', '-s', type = str, default = 'NPMI1s', help = 'similarity operation (PMIs, NPMI1s, prob)')
    p.add_option('--delta', '-d', type = float, default = 0.0, help = 'smoothing parameter')
    p.add_option('--num_samples', '-S', type = int, default = 50, help = 'number of Monte Carlo samples')
    p.add_option('-v', action = 'store_true', default = False, help = 'save plot')
    opts, args = p.parse_args()

    attr, attr_type, num_train_each, sim, delta, num_samples, save_plot = opts.attr, opts.attr_type, opts.num_train_each, opts.sim, opts.delta, opts.num_samples, opts.v

    #means = ['arith', 'geom']
    means = ['arith']  # arith seems better
    steps = ['one', 'inf']
    flags = [1, 2, 3]  # 1 content, 2 context, 3 both
    colors = ['red', 'green', 'orange', 'blue']
    linestyles = ['dotted', 'dashed', 'solid']

    folder = 'gplus0_lcc/baseline6/'
    agg_precision_filename = folder + '%s_%s_n%d_%s_delta%s_precision.csv' % (attr_type, attr, num_train_each, sim, str(delta))
    plot_filename = folder + '%s_%s_n%d_%s_delta%s_precision.png' % (attr_type, attr, num_train_each, sim, str(delta))

    print_flush("\nNominating nodes with whose '%s' attribute is '%s' (%d pos/neg seeds)..." % (attr_type, attr, num_train_each))
    print_flush("\nLoading AttributeAnalyzer...")
    a = AttributeAnalyzer(load_data = False)
    other_attr_types = [at for at in a.attr_types if (at != attr_type)]
    sim_names = other_attr_types + ['social_graph']
    sqrt_samples = np.sqrt(num_samples)

    try:
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
        included_attr_types = [at for at in attr_types if (at != attr_type)]
        random_walk_ops = []
        for at in attr_types:
            if (at != attr_type):
                a.make_random_walk_operator(at, sim = sim, delta = delta, load = True, save = True)
                random_walk_ops.append(a.random_walk_operators[at])
        g = Gplus()
        g.load_sparse_adjacency_operator()
        random_walk_ops.append((ConstantDiagonalLinearOperator(a.num_vertices, 1.0) + g.sparse_adjacency_operator).to_column_stochastic())
        solve_ops = [ConstantDiagonalLinearOperator(a.num_vertices, 1.0) - 0.5 * rw_op for rw_op in random_walk_ops]

        # get attribute indicator for all the nodes
        attr_indicator = a.get_attribute_indicator(attr, attr_type)

        precision_dfs = {(mean, step, flag) : pd.DataFrame(columns = range(num_samples)) for (mean, step, flag) in itertools.product(means, steps, flags) if ((mean == 'arith') or (flag & 1))}

        for s in range(num_samples):
            print_flush("\nSEED = %d" % s)
            np.random.seed(s)
            (training_true, training_false, test) = a.get_attribute_sample(attr, attr_type, num_train_each)
            test_df = pd.DataFrame(columns = ['test'] + list(itertools.product(steps, sim_names)))
            test_out = attr_indicator[test]
            test_df['test'] = test_out
            x0_plus, x0_minus = np.zeros(a.num_vertices), np.zeros(a.num_vertices)
            p = 1.0 / num_train_each
            for (i, j) in zip(training_true, training_false):  # initial states in the Markov chain
                x0_plus[i] = p
                x0_minus[j] = p
            for (rw, op, sim_name) in zip(random_walk_ops, solve_ops, sim_names):
                print_flush("Solving for step nomination probabilities (%s)..." % sim_name)
                x1_plus = normalize(rw * x0_plus)
                x1_minus = normalize(rw * x0_minus)
                probs = x1_plus / (x1_plus + x1_minus)  # normalize probabilities
                #probs = x1_plus - x1_minus
                test_df[('one', sim_name)] = [probs[i] for i in test]
                sol_plus = normalize(timeit(bicg)(op, x0_plus)[0])
                sol_minus = normalize(timeit(bicg)(op, x0_minus)[0])
                probs = sol_plus / (sol_plus + sol_minus)  # normalize probabilities
                #probs = sol_plus - sol_minus
                test_df[('inf', sim_name)] = [probs[i] for i in test]
            
            for (mean, step, flag) in itertools.product(means, steps, flags):
                cols = [] + (other_attr_types if (flag & 1) else []) + (['social_graph'] if (flag & 2) else [])
                cols = [(step, sim_name) for sim_name in cols]
                if (mean == 'arith'):
                    test_df[(mean, step, flag)] = test_df[cols].sum(axis = 1)
                else:
                    if (flags == 2):
                        continue
                    test_df[(mean, step, flag)] = np.log(test_df[cols]).sum(axis = 1)

            # do vertex nomination
            for (mean, step, flag) in itertools.product(means, steps, flags):
                if ((mean == 'arith') or (flag & 1)):
                    test_df = test_df.sort_values(by = (mean, step, flag), ascending = False)
                    precision_dfs[(mean, step, flag)][s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test) + 1.0)

        # compute means and standard errors over all the samples
        agg_precision_df = pd.DataFrame(columns = list(itertools.product(['mean', 'stderr'], means, steps, flags)))
        for (mean, step, flag) in itertools.product(means, steps, flags):
            if ((mean == 'arith') or (flag & 1)):
                agg_precision_df[('mean', mean, step, flag)] = precision_dfs[(mean, step, flag)].mean(axis = 1)
                agg_precision_df[('stderr', mean, step, flag)] = precision_dfs[(mean, step, flag)].std(axis = 1) / sqrt_samples

        # save the aggregate data frames
        N_save = min(len(test_out), topN_save)
        agg_precision_df = agg_precision_df[:N_save]

        agg_precision_df.to_csv(agg_precision_filename, index = False)

        num_true_in_test = test_out.sum()
        num_test = len(test_out)


    # plot the nomination precision 
    if save_plot:
        N_plot = min(len(agg_precision_df), topN_plot)
        plots = []

        for (i, (mean, step)) in enumerate(itertools.product(means, steps)):
            for flag in flags:
                if ((mean == 'arith') or (flag & 1)):
                    plt.fill_between(agg_precision_df.index, agg_precision_df[str(('mean', mean, step, flag))] - 2 * agg_precision_df[str(('stderr', mean, step, flag))], agg_precision_df[str(('mean', mean, step, flag))] + 2 * agg_precision_df[str(('stderr', mean, step, flag))], color = colors[i], alpha = 0.1)
                    plot, = plt.plot(agg_precision_df.index, agg_precision_df[str(('mean', mean, step, flag))], color = colors[i], linewidth = 2, linestyle = linestyles[flag - 1], label = ','.join([mean, step, str(flag)]))
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

    print("\nDone!")

if __name__ == "__main__":
    main()

