# Performs vertex nomination on content + context (based on joint PMI embeddings) using random forests, AdaBoost, logistic regression, and Gaussian Naive Bayes. Obscures one attribute type and does nomination on the desired attribute of that type. Averages precision-by-rank over a number of samples.

done_import = False
while (not done_import):
    try:
        import optparse
        import matplotlib.pyplot as plt
        import sys
        from gplus import *
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression
        from collections import defaultdict
        done_import = True
    except:
        pass

pd.options.display.max_rows = None
pd.options.display.width = 1000

topN_save = 1000    # number of precisions to save
topN_plot = 500     # number of precisions to plot
topN_nominees = 50  # number of nominees to include for top attribute analysis
classifiers = ['rfc', 'boost', 'logreg', 'gnb']

num_rf_trees = 100  # number of trees in random forest
num_boost_trees = 100  # number of trees in AdaBoost

sim = 'NPMI1s'  # use NPMI as attribute similarity measure
delta = 0.0
tau = 1.0  # try adding identity to off-diagonal blocks

def main():
    p = optparse.OptionParser()
    p.add_option('--attr', '-a', type = str, help = 'attribute')
    p.add_option('--attr_type', '-t', type = str, help = 'attribute type')
    p.add_option('--num_train_each', '-n', type = int, help = 'number of training samples of True and False for the attribute (for total of 2n training samples)')
    p.add_option('--embedding', '-e', type = str, help = 'embedding (adj, normlap, regnormlap)')
    p.add_option('-k', type = int, help = 'number of eigenvalues')
    p.add_option('--sphere', '-s', action = 'store_true', default = False, help = 'normalize in sphere')
    p.add_option('--num_samples', '-S', type = int, default = 50, help = 'number of Monte Carlo samples')
    p.add_option('-v', action = 'store_true', default = False, help = 'save plot')
    p.add_option('--jobs', '-j', type = int, default = -1, help = 'number of jobs')
    opts, args = p.parse_args()

    attr, attr_type, num_train_each, embedding, k, sphere, num_samples, save_plot, jobs = opts.attr, opts.attr_type, opts.num_train_each, opts.embedding, opts.k, opts.sphere, opts.num_samples, opts.v, opts.jobs

    folder = 'gplus0_lcc/baseline5/'
    agg_precision_filename = folder + '%s_%s_n%d_%s_k%d%s_precision.csv' % (attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '')
    plot_filename = folder + '%s_%s_n%d_%s_k%d%s_precision.png' % (attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '')
    top_attrs_filename = folder + '%s_%s_n%d_%s_k%d%s_top_attrs.txt' % (attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '')

    print("\nNominating nodes with whose '%s' attribute is '%s' (%d pos/neg seeds)..." % (attr_type, attr, num_train_each))
    print("\nLoading AttributeAnalyzer...")
    a = AttributeAnalyzer(load_data = False)
    sqrt_samples = np.sqrt(num_samples)

    try:
        agg_precision_df = pd.read_csv(agg_precision_filename)
        print("\nLoaded data from '%s'." % agg_precision_filename)
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
        print("\nLoading attribute data...")
        timeit(a.load_data)()
        a.make_joint_attr_embedding_matrix(attr_type, sim = sim, embedding = embedding, delta = delta, tau = tau, k = k, sphere = 2 if sphere else 0)

        # get attribute indicator for all the nodes
        attr_indicator = a.get_attribute_indicator(attr, attr_type)

        # prepare the classifiers
        rfc = RandomForestClassifier(n_estimators = num_rf_trees, n_jobs = jobs)
        boost = AdaBoostClassifier(n_estimators = num_boost_trees)
        logreg = LogisticRegression(n_jobs = jobs)
        gnb = GaussianNB()
        rfc_precision_df = pd.DataFrame(columns = range(num_samples))
        boost_precision_df = pd.DataFrame(columns = range(num_samples))
        logreg_precision_df = pd.DataFrame(columns = range(num_samples))
        gnb_precision_df = pd.DataFrame(columns = range(num_samples))

        # maintain top nominee attributes dictionary
        top_attrs = defaultdict(float)

        for s in range(num_samples):
            print("\nSEED = %d" % s)
            np.random.seed(s)
            print("\nObtaining feature vectors for random training and test sets...")
            ((train_in, train_out), (test_in, test_out)) = timeit(a.get_joint_PMI_training_and_test)(attr, attr_type, num_train_each)

            # train and predict
            print("\nTraining %d random forest trees..." % num_rf_trees)
            timeit(rfc.fit)(train_in, train_out)
            print("\nPredicting probabilities...")
            probs_rfc = timeit(rfc.predict_proba)(test_in)[:, 1]

            print("\nTraining %d AdaBoost trees..." % num_boost_trees)
            timeit(boost.fit)(train_in, train_out)
            print("\nPredicting probabilities...")
            probs_boost = timeit(boost.predict_proba)(test_in)[:, 1]

            print("\nTraining logistic regression...")
            timeit(logreg.fit)(train_in, train_out)
            print("\nPredicting probabilities...")
            probs_logreg = timeit(logreg.predict_proba)(test_in)[:, 1]

            print("\nTraining Naive Bayes...")
            timeit(gnb.fit)(train_in, train_out)
            print("\nPredicting probabilities...")
            probs_gnb = timeit(gnb.predict_proba)(test_in)[:, 1]            

            test_df = pd.DataFrame(columns = ['test', 'probs_rfc', 'probs_boost', 'probs_logreg', 'probs_gnb'])
            test_df['test'] = test_out
            test_df['probs_rfc'] = probs_rfc
            test_df['probs_boost'] = probs_boost
            test_df['probs_logreg'] = probs_logreg
            test_df['probs_gnb'] = probs_gnb

            # do vertex nomination
            test_df = test_df.sort_values(by = 'probs_rfc', ascending = False)
            rfc_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            test_df = test_df.sort_values(by = 'probs_boost', ascending = False)
            boost_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            test_df = test_df.sort_values(by = 'probs_logreg', ascending = False)
            logreg_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            test_df = test_df.sort_values(by = 'probs_gnb', ascending = False)
            gnb_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)

            # determine top attributes
            best_i, best_prec = -1, -1.0
            for (i, prec_series) in enumerate([rfc_precision_df[s], boost_precision_df[s], logreg_precision_df[s], gnb_precision_df[s]]):
                if (prec_series[topN_nominees - 1] > best_prec):
                    best_i, best_prec = i, prec_series[topN_nominees - 1]
            test_df = test_df.sort_values(by = 'probs_%s' % classifiers[i], ascending = False)
            for node in test_df.index[:topN_nominees]:
                attrs = a.attrs_by_node_by_type[attr_type][node]
                for at in attrs:
                    top_attrs[at] += 1.0 / len(attrs)  # divide the vote equally among all attributes

            sys.stdout.flush()  # flush the output buffer

        # compute means and standard errors over all the samples
        agg_precision_df = pd.DataFrame(columns = ['mean_rfc_prec', 'stderr_rfc_prec', 'mean_boost_prec', 'stderr_boost_prec', 'mean_logreg_prec', 'stderr_logreg_prec', 'mean_gnb_prec', 'stderr_gnb_prec', 'max_mean_prec'])
        agg_precision_df['mean_rfc_prec'] = rfc_precision_df.mean(axis = 1)
        agg_precision_df['stderr_rfc_prec'] = rfc_precision_df.std(axis = 1) / sqrt_samples
        agg_precision_df['mean_boost_prec'] = boost_precision_df.mean(axis = 1)
        agg_precision_df['stderr_boost_prec'] = boost_precision_df.std(axis = 1) / sqrt_samples
        agg_precision_df['mean_logreg_prec'] = logreg_precision_df.mean(axis = 1)
        agg_precision_df['stderr_logreg_prec'] = logreg_precision_df.std(axis = 1) / sqrt_samples
        agg_precision_df['mean_gnb_prec'] = gnb_precision_df.mean(axis = 1)
        agg_precision_df['stderr_gnb_prec'] = gnb_precision_df.std(axis = 1) / sqrt_samples
        agg_precision_df['max_mean_prec'] = agg_precision_df[['mean_rfc_prec', 'mean_boost_prec', 'mean_logreg_prec', 'mean_gnb_prec']].max(axis = 1)

        # save the aggregate data frames
        N_save = min(len(test_out), topN_save)
        agg_precision_df = agg_precision_df[:N_save]

        agg_precision_df.to_csv(agg_precision_filename, index = False)

        top_attrs_df = pd.DataFrame(list(top_attrs.items()), columns = ['attribute', 'voteProportion'])
        top_attrs_df = top_attrs_df.set_index('attribute')
        top_attrs_df['voteProportion'] /= top_attrs_df['voteProportion'].sum()
        top_attrs_df = top_attrs_df.sort_values(by = 'voteProportion', ascending = False)
        open(top_attrs_filename, 'w').write(str(top_attrs_df))

        num_true_in_test = test_out.sum()
        num_test = len(test_out)

    # plot the nomination precision 
    if save_plot:
        N_plot = min(len(agg_precision_df), topN_plot)
        plt.fill_between(agg_precision_df.index, agg_precision_df['mean_rfc_prec'] - 2 * agg_precision_df['stderr_rfc_prec'], agg_precision_df['mean_rfc_prec'] + 2 * agg_precision_df['stderr_rfc_prec'], color = 'green', alpha = 0.25)
        rfc_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_rfc_prec'], color = 'green', linewidth = 2, label = 'Random Forest')
        plt.fill_between(agg_precision_df.index, agg_precision_df['mean_boost_prec'] - 2 * agg_precision_df['stderr_boost_prec'], agg_precision_df['mean_boost_prec'] + 2 * agg_precision_df['stderr_boost_prec'], color = 'blue', alpha = 0.25)
        boost_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_boost_prec'], color = 'blue', linewidth = 2, label = 'AdaBoost')
        plt.fill_between(agg_precision_df.index, agg_precision_df['mean_logreg_prec'] - 2 * agg_precision_df['stderr_logreg_prec'], agg_precision_df['mean_logreg_prec'] + 2 * agg_precision_df['stderr_logreg_prec'], color = 'red', alpha = 0.25)
        logreg_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_logreg_prec'], color = 'red', linewidth = 2, label = 'Logistic Regression')
        plt.fill_between(agg_precision_df.index, agg_precision_df['mean_gnb_prec'] - 2 * agg_precision_df['stderr_gnb_prec'], agg_precision_df['mean_gnb_prec'] + 2 * agg_precision_df['stderr_gnb_prec'], color = 'orange', alpha = 0.25)
        gnb_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_gnb_prec'], color = 'orange', linewidth = 2, label = 'Naive Bayes')

        guess_rate = num_true_in_test / num_test
        guess, = plt.plot([guess_rate for i in range(N_plot)], linestyle = 'dashed', linewidth = 2, color = 'black', label = 'Guess')
        plt.xlabel('rank')
        plt.ylabel('precision')
        plt.xlim((0.0, N_plot))
        plt.ylim((0.0, 1.0))
        plt.title('Vertex Nomination Precision')
        plt.legend(handles = [rfc_plot, boost_plot, logreg_plot, gnb_plot, guess])
        plt.savefig(plot_filename)

    print("\nDone!")

if __name__ == "__main__":
    main()


