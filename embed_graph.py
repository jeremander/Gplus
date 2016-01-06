# Embeds the gplus0_lcc graph into feature space

from gplus import *


def main():
    p = optparse.OptionParser()
    p.add_option('-e', type = str, help = 'embedding (adj, adj+diag, normlap, regnormlap)')
    p.add_option('-k', type = int, help = 'number of eigenvalues')
    p.add_option('-t', type = float, default = None, help = 'tolerance for eigsh')
    p.add_option('-v', action = 'store_true', default = False, help = 'save scree plot')
    opts, args = p.parse_args()

    embedding = opts.e
    k = opts.k
    tol = opts.t
    save_plot = opts.v

    g = Gplus()
    g.make_graph_embedding_matrix(embedding = embedding, k = k, tol = tol, plot = save_plot, load = True, save = True)

    print("\nDone!")
    

if __name__ == "__main__":
    main()