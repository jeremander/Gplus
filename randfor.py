# Performs vertex nomination using random forests. Obscures one attribute type and does binary classification on the desired attribute of that type. 

import optparse
import matplotlib.pyplot as plt
from gplus import *
from sklearn.ensemble import RandomForestClassifier


def main():
    p = optparse.OptionParser()
    p.add_option('--attr', '-a', type = str, help = 'attribute')
    p.add_option('--attr_type', '-t', type = str, help = 'attribute type')
    p.add_option('--num_train_each', '-n', type = int, help = 'number of training samples of True and False for the attribute (for total of 2n training samples)')
    p.add_option('--max_count_features', '-m', type = int, default = 500, help = 'max number of count features')
    p.add_option('--num_estimators', '-e', type = int, default = 1000, help = 'number of trees in forest')
    p.add_option('--seed', '-s', type = int, default = None, help = 'random seed')
    p.add_option('--jobs', '-j', type = int, default = -1, help = 'number of jobs')
    opts, args = p.parse_args()

    attr, attr_type, num_train_each, max_count_features, num_estimators, seed, jobs = opts.attr, opts.attr_type, opts.num_train_each, opts.max_count_features, opts.num_estimators, opts.seed, opts.jobs

    if (seed is None):
        seed = np.random.randint(2**32)

    folder = 'gplus0_lcc/randfor/'

    print("\nLoading AttributeAnalyzer...")
    a = AttributeAnalyzer()
    a.load_attrs_by_node_by_type()
    print("\nMaking count vectorizers...")
    a.make_count_vectorizers(max_count_features, load = True, save = True)
    a.make_complete_feature_matrix(max_count_features, load = True, save = True)
    print("\nObtaining feature vectors for random training and test sets.")
    np.random.seed(seed)  # seed the subset sampling
    ((train_in, train_out), (test_in, test_out)) = a.get_training_and_test(attr, attr_type, num_train_each)

    @timeit
    def save_indices():
        index_filename = folder + '%s_%s_n%d_s%d_indices.csv' % (attr_type, attr, num_train_each, seed)
        print("\nSaving training/test indices to %s..." % index_filename)
        with open(index_filename, 'w') as f:
            f.write(','.join(map(str, train_out[train_out == 1].index)))
            f.write('\n')
            f.write(','.join(map(str, train_out[train_out == 0].index)))
            f.write('\n')
            f.write(','.join(map(str, test_out.index)))
            f.write('\n')

    save_indices()

    # train and predict
    rfc = RandomForestClassifier(n_estimators = num_estimators, n_jobs = jobs)
    print("\nTraining %d random forests..." % num_estimators)
    timeit(rfc.fit)(train_in, train_out)
    print("\nPredicting probabilities...")
    probs = timeit(rfc.predict_proba)(test_in)[:, 1]

    test_df = pd.DataFrame(columns = ['test', 'probs', 'precision'])
    test_df['test'] = test_out
    test_df['probs'] = probs
    test_df = test_df.sort_values(by = 'probs', ascending = False)

    # do vertex nomination
    test_df['precision'] = test_df['test'].cumsum() / np.arange(1.0, len(test_out) + 1.0)
    test_filename = folder + '%s_%s_n%d_s%d_m%d_e%d_test.csv' % (attr_type, attr, num_train_each, seed, max_count_features, num_estimators)
    test_df.to_csv(test_filename, index = True)

    # plot the nomination precision
    num_ranks = min(len(test_out), 1000)
    plt.plot(test_df['precision'][:num_ranks], linewidth = 2, color = 'green')
    guess_rate = test_df['precision'].iloc[-1]
    plt.plot([guess_rate for i in range(num_ranks)], linestyle = 'dashed', linewidth = 2, color = 'red')
    plt.xlabel('rank')
    plt.ylabel('precision')
    plt.ylim((0.0, 1.0))
    plt.title('Vertex Nomination Precision')
    plot_filename = folder + '%s_%s_n%d_s%d_m%d_e%d_precision.png' % (attr_type, attr, num_train_each, seed, max_count_features, num_estimators)
    plt.savefig(plot_filename)


    # get the feature importances
    features = []
    for at in a.attr_types:
        if (at != attr_type):
            pairs = sorted(list(a.word_cvs[at].vocabulary_.items()), key = lambda pair : pair[1])
            features += ['*' + at + ',word* ' + pair[0] for pair in pairs]
            pairs = sorted(list(a.char_cvs[at].vocabulary_.items()), key = lambda pair : pair[1])
            features += ['*' + at + ',char* ' + pair[0] for pair in pairs]
    feature_importances_series = pd.Series(rfc.feature_importances_, index = features).sort_values(ascending = False)
    feature_importances_filename = folder + '%s_%s_n%d_s%d_m%d_e%d_feature_importances.csv' % (attr_type, attr, num_train_each, seed, max_count_features, num_estimators)
    feature_importances_series.to_csv(feature_importances_filename, index = True)

    print("\nDone!")

if __name__ == "__main__":
    main()