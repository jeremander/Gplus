# Performs vertex nomination on context (social graph) using random forests, logistic regression, and semi-supervised k-means. Obscures one attribute type and does nomination on the desired attribute of that type. Averages precision-by-rank over a number of samples.

import optparse
import matplotlib.pyplot as plt
from gplus import *
from ssvn import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from skrvm import RVC

topN_save = 1000
topN_plot = 500

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
    p.add_option('--num_samples', '-S', type = int, default = 50, help = 'number of Monte Carlo samples')
    p.add_option('-v', action = 'store_true', default = False, help = 'save plot')
    p.add_option('--jobs', '-j', type = int, default = -1, help = 'number of jobs')
    opts, args = p.parse_args()

    attr, attr_type, num_train_each, embedding, k, sphere, num_samples, save_plot, jobs = opts.attr, opts.attr_type, opts.num_train_each, opts.embedding, opts.k, opts.sphere, opts.num_samples, opts.v, opts.jobs

    folder = 'gplus0_lcc/baseline2/'
    agg_precision_filename = folder + '%s_%s_n%d_%s_k%d%s_precision.csv' % (attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '')
    plot_filename = folder + '%s_%s_n%d_%s_k%d%s_precision.png' % (attr_type, attr, num_train_each, embedding, k, '_normalize' if sphere else '')

    print("\nNominating nodes with whose '%s' attribute is '%s' (%d pos/neg seeds)..." % (attr_type, attr, num_train_each))
    print("\nLoading AttributeAnalyzer...")
    a = AttributeAnalyzer(load_data = False)
    a.load_attrs_by_node_by_type()
    sqrt_samples = np.sqrt(num_samples)

    g = Gplus()
    print("\nLoading graph embedding...")
    g.make_graph_embedding_matrix(embedding = embedding, k = k, tol = None, plot = False, load = True, save = False)

    try:
        agg_precision_df = pd.read_csv(agg_precision_filename)
        print("\nLoaded data from '%s'." % agg_precision_filename)
        ind = a.get_attribute_indicator(attr, attr_type)
        num_true_in_test = len(ind[ind == 1]) - num_train_each
        num_test = ind.count() - 2 * num_train_each

    except OSError:

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
        rfc_precision_df = pd.DataFrame(columns = range(num_samples))
        boost_precision_df = pd.DataFrame(columns = range(num_samples))
        logreg_precision_df = pd.DataFrame(columns = range(num_samples))

        for s in range(num_samples):
            print("\nSEED = %d" % s)
            np.random.seed(s)
            print("\nObtaining feature vectors for random training and test sets...")
            (training_true, training_false, test) = a.get_attribute_sample(attr, attr_type, num_train_each)
            training = sorted(training_true + training_false)
            train_rows, test_rows = [nodes_to_rows[i] for i in training], [nodes_to_rows[i] for i in test]
            train_in, test_in = g.graph_embedding_matrix[train_rows], g.graph_embedding_matrix[test_rows]
            train_out, test_out = attr_indicator[training], attr_indicator[test]

            # train and predict
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

            # M = np.vstack([train_in, test_in])
            # output = pd.concat([train_out, test_out])
            # output.iloc[2 * num_train_each : ] = -1  
            # observe = list(output)
            # print("Performing semi-supervised kMeans...")
            # nom = timeit(nominate_from_embedding)(M, observe, k)

            test_df = pd.DataFrame(columns = ['test', 'probs_rfc', 'probs_boost', 'probs_logreg'])
            test_df['test'] = test_out
            test_df['probs_rfc'] = probs_rfc
            test_df['probs_boost'] = probs_boost
            test_df['probs_logreg'] = probs_logreg

            # do vertex nomination
            test_df = test_df.sort_values(by = 'probs_rfc', ascending = False)
            rfc_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            test_df = test_df.sort_values(by = 'probs_boost', ascending = False)
            boost_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)
            test_df = test_df.sort_values(by = 'probs_logreg', ascending = False)
            logreg_precision_df[s] = np.asarray(test_df['test']).cumsum() / np.arange(1.0, len(test_out) + 1.0)

        # compute means and standard deviations over all the samples
        agg_precision_df = pd.DataFrame(columns = ['mean_rfc_prec', 'stderr_rfc_prec', 'mean_logreg_prec', 'stderr_logreg_prec'])
        agg_precision_df['mean_rfc_prec'] = rfc_precision_df.mean(axis = 1)
        agg_precision_df['stderr_rfc_prec'] = rfc_precision_df.std(axis = 1) / sqrt_samples
        agg_precision_df['mean_logreg_prec'] = logreg_precision_df.mean(axis = 1)
        agg_precision_df['stderr_logreg_prec'] = logreg_precision_df.std(axis = 1) / sqrt_samples

        # save the aggregate data frames
        N_save = min(len(test_out), topN_save)
        agg_precision_df = agg_precision_df[:N_save]

        agg_precision_df.to_csv(agg_precision_filename, index = False)

        num_true_in_test = test_out.sum()
        num_test = len(test_out)

    # plot the nomination precision 
    if save_plot:
        N_plot = min(len(agg_precision_df), topN_plot)
        plt.fill_between(agg_precision_df.index, agg_precision_df['mean_rfc_prec'] - 2 * agg_precision_df['stderr_rfc_prec'], agg_precision_df['mean_rfc_prec'] + 2 * agg_precision_df['stderr_rfc_prec'], color = 'green', alpha = 0.25)
        rfc_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_rfc_prec'], color = 'green', linewidth = 2, label = 'Random Forest')
        plt.fill_between(agg_precision_df.index, agg_precision_df['mean_logreg_prec'] - 2 * agg_precision_df['stderr_logreg_prec'], agg_precision_df['mean_logreg_prec'] + 2 * agg_precision_df['stderr_logreg_prec'], color = 'red', alpha = 0.25)
        logreg_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_logreg_prec'], color = 'red', linewidth = 2, label = 'Logistic Regression')
        # plt.fill_between(agg_precision_df.index, agg_precision_df['mean_rvc_prec'] - 2 * agg_precision_df['stderr_rvc_prec'], agg_precision_df['mean_rvc_prec'] + 2 * agg_precision_df['stderr_rvc_prec'], color = 'orange', alpha = 0.25)
        # rvc_plot, = plt.plot(agg_precision_df.index, agg_precision_df['mean_rvc_prec'], color = 'orange', linewidth = 2, label = 'Relevance Vector Machine')

        guess_rate = num_true_in_test / num_test
        guess, = plt.plot([guess_rate for i in range(N_plot)], linestyle = 'dashed', linewidth = 2, color = 'black', label = 'Guess')
        plt.xlabel('rank')
        plt.ylabel('precision')
        plt.xlim((0.0, N_plot))
        plt.ylim((0.0, 1.0))
        plt.title('Vertex Nomination Precision')
        plt.legend(handles = [rfc_plot, logreg_plot, guess])
        plt.savefig(plot_filename)

    print("\nDone!")

if __name__ == "__main__":
    main()