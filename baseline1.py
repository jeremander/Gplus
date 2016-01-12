# Performs vertex nomination on content (node attributes) using random forests, AdaBoost, logistic regression, and Multinomial Naive Bayes. Obscures one attribute type and does nomination on the desired attribute of that type. Averages precision-by-rank over a number of samples.

done_import = False
while (not done_import):
    try:
        import optparse
        import matplotlib.pyplot as plt
        import sys
        from gplus import *
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression
        done_import = True
    except:
        pass


topN_save = 1000
topN_plot = 500

num_rf_trees = 100  # number of trees in random forest
num_boost_trees = 100  # number of trees in AdaBoost

def main():
    p = optparse.OptionParser()
    p.add_option('--attr', '-a', type = str, help = 'attribute')
    p.add_option('--attr_type', '-t', type = str, help = 'attribute type')
    p.add_option('--num_train_each', '-n', type = int, help = 'number of training samples of True and False for the attribute (for total of 2n training samples)')
    p.add_option('--max_count_features', '-m', type = int, default = 1000, help = 'max number of count features')
    p.add_option('--num_samples', '-S', type = int, default = 50, help = 'number of Monte Carlo samples')
    p.add_option('-v', action = 'store_true', default = False, help = 'save plot')
    p.add_option('--jobs', '-j', type = int, default = -1, help = 'number of jobs')
    opts, args = p.parse_args()

    attr, attr_type, num_train_each, max_count_features, num_samples, save_plot, jobs = opts.attr, opts.attr_type, opts.num_train_each, opts.max_count_features, opts.num_samples, opts.v, opts.jobs

    folder = 'gplus0_lcc/baseline1/'
    agg_precision_filename = folder + '%s_%s_n%d_m%d_precision.csv' % (attr_type, attr, num_train_each, max_count_features)
    agg_feature_importances_filename = folder + '%s_%s_n%d_m%d_feature_importances.csv' % (attr_type, attr, num_train_each, max_count_features)
    agg_word_importances_filename = folder + '%s_%s_n%d_m%d_word_importances.csv' % (attr_type, attr, num_train_each, max_count_features)
    plot_filename = folder + '%s_%s_n%d_m%d_precision.png' % (attr_type, attr, num_train_each, max_count_features)

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
        a.load_attrs_by_node_by_type()
        print("\nMaking count vectorizers...")
        a.make_count_vectorizers(max_count_features, load = True, save = True)
        a.make_complete_feature_matrix(max_count_features, load = True, save = True)

        # prepare the classifiers
        rfc = RandomForestClassifier(n_estimators = num_rf_trees, n_jobs = jobs)
        boost = AdaBoostClassifier(n_estimators = num_boost_trees)
        logreg = LogisticRegression(n_jobs = jobs)
        mnb = MultinomialNB()
        rfc_precision_df = pd.DataFrame(columns = range(num_samples))
        boost_precision_df = pd.DataFrame(columns = range(num_samples))
        logreg_precision_df = pd.DataFrame(columns = range(num_samples))
        mnb_precision_df = pd.DataFrame(columns = range(num_samples))

        # get the feature list
        attr_type_col = []
        word_char_col = []
        features = []
        for at in a.attr_types:
            if (at != attr_type):
                pairs = sorted(list(a.word_cvs[at].vocabulary_.items()), key = lambda pair : pair[1])
                attr_type_col += [at for pair in pairs]
                word_char_col += ['word' for pair in pairs]
                features += [pair[0] for pair in pairs]
                pairs = sorted(list(a.char_cvs[at].vocabulary_.items()), key = lambda pair : pair[1])
                attr_type_col += [at for pair in pairs]
                word_char_col += ['char' for pair in pairs]
                features += [pair[0] for pair in pairs]
        feature_importances_df = pd.DataFrame(columns = ['attributeType', 'word/char'] + list(range(num_samples)), index = features)
        feature_importances_df['attributeType'] = attr_type_col
        feature_importances_df['word/char'] = word_char_col

        for s in range(num_samples):
            print("\nSEED = %d" % s)
            np.random.seed(s)
            print("\nObtaining feature vectors for random training and test sets...")
            ((train_in, train_out), (test_in, test_out)) = timeit(a.get_training_and_test)(attr, attr_type, num_train_each)

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
            timeit(mnb.fit)(train_in, train_out)
            print("\nPredicting probabilities...")
            probs_mnb = timeit(mnb.predict_proba)(test_in)[:, 1]            

            test_df = pd.DataFrame(columns = ['test', 'probs_rfc', 'probs_boost', 'probs_logreg', 'probs_mnb'])
            test_df['test'] = test_out
            test_df['probs_rfc'] = probs_rfc
            test_df['probs_boost'] = probs_boost
            test_df['probs_logreg'] = probs_logreg
            test_df['probs_mnb'] = probs_mnb

            # do vertex nomination
            test_df = test_df.sort_values(by = 'probs_rfc', ascending = False)
            rfc_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            test_df = test_df.sort_values(by = 'probs_boost', ascending = False)
            boost_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            test_df = test_df.sort_values(by = 'probs_logreg', ascending = False)
            logreg_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            test_df = test_df.sort_values(by = 'probs_mnb', ascending = False)
            mnb_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)

            # collate the feature importance scores from the random forest
            feature_importances_df[s] = rfc.feature_importances_

            sys.stdout.flush()  # flush the output buffer

        # compute means and standard errors over all the samples
        agg_precision_df = pd.DataFrame(columns = ['mean_rfc_prec', 'stderr_rfc_prec', 'mean_boost_prec', 'stderr_boost_prec', 'mean_logreg_prec', 'stderr_logreg_prec', 'mean_mnb_prec', 'stderr_mnb_prec'])
        agg_precision_df['mean_rfc_prec'] = rfc_precision_df.mean(axis = 1)
        agg_precision_df['stderr_rfc_prec'] = rfc_precision_df.std(axis = 1) / sqrt_samples
        agg_precision_df['mean_boost_prec'] = boost_precision_df.mean(axis = 1)
        agg_precision_df['stderr_boost_prec'] = boost_precision_df.std(axis = 1) / sqrt_samples
        agg_precision_df['mean_logreg_prec'] = logreg_precision_df.mean(axis = 1)
        agg_precision_df['stderr_logreg_prec'] = logreg_precision_df.std(axis = 1) / sqrt_samples
        agg_precision_df['mean_mnb_prec'] = mnb_precision_df.mean(axis = 1)
        agg_precision_df['stderr_mnb_prec'] = mnb_precision_df.std(axis = 1) / sqrt_samples
        agg_feature_importances_df = pd.DataFrame(columns = ['attributeType', 'word/char', 'mean_feature_importances', 'std_feature_importances'])
        agg_feature_importances_df['attributeType'] = feature_importances_df['attributeType']
        agg_feature_importances_df['word/char'] = feature_importances_df['word/char']
        agg_feature_importances_df['mean_feature_importances'] = feature_importances_df[list(range(num_samples))].mean(axis = 1)
        agg_feature_importances_df['std_feature_importances'] = feature_importances_df[list(range(num_samples))].std(axis = 1)
        agg_feature_importances_df = agg_feature_importances_df.sort_values(by = 'mean_feature_importances', ascending = False)
        agg_word_importances_df = agg_feature_importances_df[agg_feature_importances_df['word/char'] == 'word'][['attributeType', 'mean_feature_importances', 'std_feature_importances']]

        # save the aggregate data frames
        N_save = min(len(test_out), topN_save)
        agg_precision_df = agg_precision_df[:N_save]

        agg_precision_df.to_csv(agg_precision_filename, index = False)
        agg_feature_importances_df.to_csv(agg_feature_importances_filename, index = True, sep = '\t')
        agg_word_importances_df.to_csv(agg_word_importances_filename, index = True, sep = '\t')

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
        plt.fill_between(agg_precision_df.index, agg_precision_df['mean_mnb_prec'] - 2 * agg_precision_df['stderr_mnb_prec'], agg_precision_df['mean_mnb_prec'] + 2 * agg_precision_df['stderr_mnb_prec'], color = 'orange', alpha = 0.25)
        mnb_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_mnb_prec'], color = 'orange', linewidth = 2, label = 'Naive Bayes')

        guess_rate = num_true_in_test / num_test
        guess, = plt.plot([guess_rate for i in range(N_plot)], linestyle = 'dashed', linewidth = 2, color = 'black', label = 'Guess')
        plt.xlabel('rank')
        plt.ylabel('precision')
        plt.xlim((0.0, N_plot))
        plt.ylim((0.0, 1.0))
        plt.title('Vertex Nomination Precision')
        plt.legend(handles = [rfc_plot, boost_plot, logreg_plot, mnb_plot, guess])
        plt.savefig(plot_filename)

    print("\nDone!")

if __name__ == "__main__":
    main()