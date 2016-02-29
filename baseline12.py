# Performs vertex nomination on content and context using random forests, AdaBoost, logistic regression, and Multinomial Naive Bayes. Obscures one attribute type and does nomination on the desired attribute of that type. Takes both mean and max of the two probabilities for each node, then averages precision-by-rank over a number of samples.

done_import = False
while (not done_import):
    try:
        import optparse
        import matplotlib.pyplot as plt
        import sys
        from gplus import *
        from sklearn.naive_bayes import MultinomialNB, GaussianNB
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
classifiers = ['rfc', 'boost', 'logreg', 'nb']

num_rf_trees = 100  # number of trees in random forest
num_boost_trees = 100  # number of trees in AdaBoost

def main():
    p = optparse.OptionParser()
    p.add_option('--attr', '-a', type = str, help = 'attribute')
    p.add_option('--attr_type', '-t', type = str, help = 'attribute type')
    p.add_option('--num_train_each', '-n', type = int, help = 'number of training samples of True and False for the attribute (for total of 2n training samples)')
    p.add_option('--embedding', '-e', type = str, help = 'embedding (adj, adj+diag, normlap, regnormlap)')
    p.add_option('-k', type = int, help = 'number of eigenvalues')
    p.add_option('--sphere', '-s', action = 'store_true', default = False, help = 'normalize in sphere')
    p.add_option('--max_count_features', '-m', type = int, default = 1000, help = 'max number of count features')
    p.add_option('--num_samples', '-S', type = int, default = 50, help = 'number of Monte Carlo samples')
    p.add_option('-v', action = 'store_true', default = False, help = 'save plot')
    p.add_option('--jobs', '-j', type = int, default = -1, help = 'number of jobs')
    opts, args = p.parse_args()

    attr, attr_type, num_train_each, embedding, k, sphere, max_count_features, num_samples, save_plot, jobs = opts.attr, opts.attr_type, opts.num_train_each, opts.embedding, opts.k, opts.sphere, opts.max_count_features, opts.num_samples, opts.v, opts.jobs

    max_folder = 'gplus0_lcc/baseline12_max/'
    mean_folder = 'gplus0_lcc/baseline12_mean/'
    max_agg_precision_filename = max_folder + '%s_%s_n%d_%s_k%d%s_m%d_precision.csv' % (attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '', max_count_features)
    mean_agg_precision_filename = mean_folder + '%s_%s_n%d_%s_k%d%s_m%d_precision.csv' % (attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '', max_count_features)
    max_plot_filename = max_folder + '%s_%s_n%d_%s_k%d%s_m%d_precision.png' % (attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '', max_count_features)
    mean_plot_filename = mean_folder + '%s_%s_n%d_%s_k%d%s_m%d_precision.png' % (attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '', max_count_features)

    print("\nNominating nodes with whose '%s' attribute is '%s' (%d pos/neg seeds)..." % (attr_type, attr, num_train_each))
    print("\nLoading AttributeAnalyzer...")
    a = AttributeAnalyzer(load_data = False)
    sqrt_samples = np.sqrt(num_samples)

    try:
        max_agg_precision_df = pd.read_csv(max_agg_precision_filename)
        print("\nLoaded data from '%s'." % max_agg_precision_filename)
        mean_agg_precision_df = pd.read_csv(mean_agg_precision_filename)
        print("\nLoaded data from '%s'." % mean_agg_precision_filename)
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
        a.load_attrs_by_node_by_type()
        print("\nMaking count vectorizers...")
        a.make_count_vectorizers(max_count_features, load = True, save = True)
        a.make_complete_feature_matrix(max_count_features, load = True, save = True)

        g = Gplus()
        print("\nLoading graph embedding...")
        g.make_graph_embedding_matrix(embedding = embedding, k = k, tol = None, plot = False, load = True, save = False)

        if sphere:
            print("\nNormalizing feature vectors...")
            timeit(normalize_mat_rows)(g.graph_embedding_matrix)

        # map node indices to feature matrix rows
        nodes_to_rows = dict((node, row) for (row, node) in enumerate(g.attributed_nodes))
        # get attribute indicator for all the nodes
        attr_indicator = a.get_attribute_indicator(attr, attr_type)

        # prepare the classifiers
        rfc = RandomForestClassifier(n_estimators = num_rf_trees, n_jobs = jobs)
        boost = AdaBoostClassifier(n_estimators = num_boost_trees)
        logreg = LogisticRegression(n_jobs = jobs)
        mnb = MultinomialNB()
        gnb = GaussianNB()
        max_rfc_precision_df = pd.DataFrame(columns = range(num_samples))
        max_boost_precision_df = pd.DataFrame(columns = range(num_samples))
        max_logreg_precision_df = pd.DataFrame(columns = range(num_samples))
        max_nb_precision_df = pd.DataFrame(columns = range(num_samples))
        mean_rfc_precision_df = pd.DataFrame(columns = range(num_samples))
        mean_boost_precision_df = pd.DataFrame(columns = range(num_samples))
        mean_logreg_precision_df = pd.DataFrame(columns = range(num_samples))
        mean_nb_precision_df = pd.DataFrame(columns = range(num_samples))


        for s in range(num_samples):
            print("\nSEED = %d" % s)

            print("\nNominating by content...")
            np.random.seed(s)
            print("\nObtaining feature vectors for random training and test sets...")
            ((train_in, train_out), (test_in, test_out)) = timeit(a.get_training_and_test)(attr, attr_type, num_train_each)

            # train and predict via content
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
            timeit(mnb.fit)(train_in, train_out)
            print("\nPredicting probabilities...")
            probs_mnb = timeit(mnb.predict_proba)(test_in)[:, 1]            

            content_df = pd.DataFrame(columns = ['test', 'probs_rfc', 'probs_boost', 'probs_logreg', 'probs_nb'])
            content_df['test'] = test_out
            content_df['probs_rfc'] = probs_rfc
            content_df['probs_boost'] = probs_boost
            content_df['probs_logreg'] = probs_logreg
            content_df['probs_nb'] = probs_mnb

            print("\nNominating by context...")
            np.random.seed(s)  # make sure to re-seed!
            print("\nObtaining feature vectors for random training and test sets...")
            (training_true, training_false, test) = a.get_attribute_sample(attr, attr_type, num_train_each)
            training = sorted(training_true + training_false)
            train_rows, test_rows = [nodes_to_rows[i] for i in training], [nodes_to_rows[i] for i in test]
            train_in, test_in = g.graph_embedding_matrix[train_rows], g.graph_embedding_matrix[test_rows]
            train_out, test_out = attr_indicator[training], attr_indicator[test]

            # train and predict via context
            print("\nTraining %d random forest trees.." % num_rf_trees)
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

            context_df = pd.DataFrame(columns = ['test', 'probs_rfc', 'probs_boost', 'probs_logreg', 'probs_nb'])
            context_df['test'] = test_out
            context_df['probs_rfc'] = probs_rfc
            context_df['probs_boost'] = probs_boost
            context_df['probs_logreg'] = probs_logreg
            context_df['probs_nb'] = probs_gnb

            max_df = pd.DataFrame(test_out, columns = ['test'])
            mean_df = pd.DataFrame(test_out, columns = ['test'])
            for col in ['probs_rfc', 'probs_boost', 'probs_logreg', 'probs_nb']:
                max_df[col] = np.vectorize(max)(content_df[col], context_df[col])
                mean_df[col] = np.vectorize(lambda x, y : np.mean([x, y]))(content_df[col], context_df[col])

            # do vertex nomination
            max_df = max_df.sort_values(by = 'probs_rfc', ascending = False)
            max_rfc_precision_df[s] = np.asarray(max_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            max_df = max_df.sort_values(by = 'probs_boost', ascending = False)
            max_boost_precision_df[s] = np.asarray(max_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            max_df = max_df.sort_values(by = 'probs_logreg', ascending = False)
            max_logreg_precision_df[s] = np.asarray(max_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            max_df = max_df.sort_values(by = 'probs_nb', ascending = False)
            max_nb_precision_df[s] = np.asarray(max_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            mean_df = mean_df.sort_values(by = 'probs_rfc', ascending = False)
            mean_rfc_precision_df[s] = np.asarray(mean_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            mean_df = mean_df.sort_values(by = 'probs_boost', ascending = False)
            mean_boost_precision_df[s] = np.asarray(mean_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            mean_df = mean_df.sort_values(by = 'probs_logreg', ascending = False)
            mean_logreg_precision_df[s] = np.asarray(mean_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            mean_df = mean_df.sort_values(by = 'probs_nb', ascending = False)
            mean_nb_precision_df[s] = np.asarray(mean_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)

            sys.stdout.flush()  # flush the output buffer

        # compute means and standard errors over all the samples
        max_agg_precision_df = pd.DataFrame(columns = ['mean_rfc_prec', 'stderr_rfc_prec', 'mean_boost_prec', 'stderr_boost_prec', 'mean_logreg_prec', 'stderr_logreg_prec', 'mean_nb_prec', 'stderr_nb_prec', 'max_mean_prec'])
        max_agg_precision_df['mean_rfc_prec'] = max_rfc_precision_df.mean(axis = 1)
        max_agg_precision_df['stderr_rfc_prec'] = max_rfc_precision_df.std(axis = 1) / sqrt_samples
        max_agg_precision_df['mean_boost_prec'] = max_boost_precision_df.mean(axis = 1)
        max_agg_precision_df['stderr_boost_prec'] = max_boost_precision_df.std(axis = 1) / sqrt_samples
        max_agg_precision_df['mean_logreg_prec'] = max_logreg_precision_df.mean(axis = 1)
        max_agg_precision_df['stderr_logreg_prec'] = max_logreg_precision_df.std(axis = 1) / sqrt_samples
        max_agg_precision_df['mean_nb_prec'] = max_nb_precision_df.mean(axis = 1)
        max_agg_precision_df['stderr_nb_prec'] = max_nb_precision_df.std(axis = 1) / sqrt_samples
        max_agg_precision_df['max_mean_prec'] = max_agg_precision_df[['mean_rfc_prec', 'mean_boost_prec', 'mean_logreg_prec', 'mean_nb_prec']].max(axis = 1)
        mean_agg_precision_df = pd.DataFrame(columns = ['mean_rfc_prec', 'stderr_rfc_prec', 'mean_boost_prec', 'stderr_boost_prec', 'mean_logreg_prec', 'stderr_logreg_prec', 'mean_nb_prec', 'stderr_nb_prec', 'max_mean_prec'])
        mean_agg_precision_df['mean_rfc_prec'] = mean_rfc_precision_df.mean(axis = 1)
        mean_agg_precision_df['stderr_rfc_prec'] = mean_rfc_precision_df.std(axis = 1) / sqrt_samples
        mean_agg_precision_df['mean_boost_prec'] = mean_boost_precision_df.mean(axis = 1)
        mean_agg_precision_df['stderr_boost_prec'] = mean_boost_precision_df.std(axis = 1) / sqrt_samples
        mean_agg_precision_df['mean_logreg_prec'] = mean_logreg_precision_df.mean(axis = 1)
        mean_agg_precision_df['stderr_logreg_prec'] = mean_logreg_precision_df.std(axis = 1) / sqrt_samples
        mean_agg_precision_df['mean_nb_prec'] = mean_nb_precision_df.mean(axis = 1)
        mean_agg_precision_df['stderr_nb_prec'] = mean_nb_precision_df.std(axis = 1) / sqrt_samples
        mean_agg_precision_df['max_mean_prec'] = mean_agg_precision_df[['mean_rfc_prec', 'mean_boost_prec', 'mean_logreg_prec', 'mean_nb_prec']].max(axis = 1)

        # save the aggregate data frames
        N_save = min(len(test_out), topN_save)
        max_agg_precision_df = max_agg_precision_df[:N_save]
        mean_agg_precision_df = mean_agg_precision_df[:N_save]

        max_agg_precision_df.to_csv(max_agg_precision_filename, index = False)
        mean_agg_precision_df.to_csv(mean_agg_precision_filename, index = False)

        num_true_in_test = test_out.sum()
        num_test = len(test_out)

    # plot the nomination precision 
    if save_plot:
        for (agg_precision_df, plot_filename) in zip([max_agg_precision_df, mean_agg_precision_df], [max_plot_filename, mean_plot_filename]):
            plt.clf()
            N_plot = min(len(agg_precision_df), topN_plot)
            plt.fill_between(agg_precision_df.index, agg_precision_df['mean_rfc_prec'] - 2 * agg_precision_df['stderr_rfc_prec'], agg_precision_df['mean_rfc_prec'] + 2 * agg_precision_df['stderr_rfc_prec'], color = 'green', alpha = 0.25)
            rfc_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_rfc_prec'], color = 'green', linewidth = 2, label = 'Random Forest')
            plt.fill_between(agg_precision_df.index, agg_precision_df['mean_boost_prec'] - 2 * agg_precision_df['stderr_boost_prec'], agg_precision_df['mean_boost_prec'] + 2 * agg_precision_df['stderr_boost_prec'], color = 'blue', alpha = 0.25)
            boost_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_boost_prec'], color = 'blue', linewidth = 2, label = 'AdaBoost')
            plt.fill_between(agg_precision_df.index, agg_precision_df['mean_logreg_prec'] - 2 * agg_precision_df['stderr_logreg_prec'], agg_precision_df['mean_logreg_prec'] + 2 * agg_precision_df['stderr_logreg_prec'], color = 'red', alpha = 0.25)
            logreg_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_logreg_prec'], color = 'red', linewidth = 2, label = 'Logistic Regression')
            plt.fill_between(agg_precision_df.index, agg_precision_df['mean_nb_prec'] - 2 * agg_precision_df['stderr_nb_prec'], agg_precision_df['mean_nb_prec'] + 2 * agg_precision_df['stderr_nb_prec'], color = 'orange', alpha = 0.25)
            nb_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_nb_prec'], color = 'orange', linewidth = 2, label = 'Naive Bayes')

            guess_rate = num_true_in_test / num_test
            guess, = plt.plot([guess_rate for i in range(N_plot)], linestyle = 'dashed', linewidth = 2, color = 'black', label = 'Guess')
            plt.xlabel('rank')
            plt.ylabel('precision')
            plt.xlim((0.0, N_plot))
            plt.ylim((0.0, 1.0))
            plt.title('Vertex Nomination Precision')
            plt.legend(handles = [rfc_plot, boost_plot, logreg_plot, nb_plot, guess])
            plt.savefig(plot_filename)

    print("\nDone!")

if __name__ == "__main__":
    main()