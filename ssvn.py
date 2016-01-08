# Calls Matlab from within Python to execute Jordan Yoder's semi-supervised kMeans code for vertex nomination

import tempfile
import subprocess
import os
import numpy as np
import igraph as ig

matlab_path = '/Applications/MATLAB_R2015b.app/bin/matlab'
ssvn_dir = os.getcwd() + '/ssVN_MATLAB/'

def nominate_from_adjacency(A, observe, k, d = None):
    """Python interface to Jordan's ssVN code. Parameters are as follows:
    A: adjacency matrix (n x n)
    observe: observe[i] is -1 if vertex i is ambiguous (class membership unknown); 
             observe[i] is 1,2,...,K according as vertex i is known to be in class 1,2,...,K, respectively;
             observe[i] is 0 if vertex i is NOT in class 1 but unknown which of the other K-1 it is
    k: number of classes
    d: (optional) dimension for embedding
    """
    observe = [val if val in range(k + 1) else -1 for val in observe ]
    try:
        r = np.random.randint(1 << 32)
        tmp_script = os.getcwd() + ('/tmp%d.m' % r)
        tmp_infile = os.getcwd() + ('/tmp%d.in' % r)
        tmp_outfile = os.getcwd() + ('/tmp%d.out' % r)
        A.transpose().astype(np.uint8).tofile(tmp_infile)  # saves as binary in column-order
        s = "cd('%s');\n" % ssvn_dir
        s += "A = fread(fopen('%s', 'rb'), [%d, %d], 'uint8');\n" % (tmp_infile, A.shape[0], A.shape[1])  # MATLAB reads the file in row-order
        s += "observe = %s;\n" % str(observe)
        s += "k = %d;\n" % k
        if (d is None):
            s += "[order] = nominateJ(A, observe, k);\n"
        else:
            s += "d = %d;\n" % d
            s += "[order] = nominateJ(A, observe, k, d);\n"
        s += ("dlmwrite('%s', order, 'precision'," % tmp_outfile) + "'%d');\n" 
        s += "quit\n"
        with open(tmp_script, 'w') as f:
            f.write(s)
        subprocess.check_call([matlab_path, '-nojvm', '', '-nodesktop', '', '-nosplash', '', '-r', tmp_script[:-2].split('/')[-1], '>', '/dev/null'])
        order = np.loadtxt(tmp_outfile, delimiter = ',', dtype = int) - 1  # change it to 0-up
        return order
    finally:
        subprocess.check_call(['rm', '-f', tmp_script, tmp_infile, tmp_outfile])


def nominate_from_embedding(M, observe, k):
    """Python interface to Jordan's ssVN code. Parameters are as in nominate_from_adjacency, except an (n x d) embedding matrix M is provided instead of the adjacency matrix."""
    observe = [val if val in range(k + 1) else -1 for val in observe ]
    try:
        r = np.random.randint(1 << 32)
        tmp_script = os.getcwd() + ('/tmp%d.m' % r)
        tmp_infile = os.getcwd() + ('/tmp%d.in' % r)
        tmp_outfile = os.getcwd() + ('/tmp%d.out' % r)
        M.transpose().astype(float).tofile(tmp_infile)
        s = "cd('%s');\n" % ssvn_dir
        s += "A = 0;\n"
        s += "observe = %s;\n" % str(observe)
        s += "k = %d;\n" % k
        s += "d = %d;\n" % M.shape[1]
        s += "M = fread(fopen('%s', 'rb'), [%d, %d], 'double');\n" % (tmp_infile, M.shape[0], M.shape[1])
        s += "embedFun = @(A, d) M;\n"
        s += "[order] = nominateJ(A, observe, k, d, embedFun);\n"
        s += ("dlmwrite('%s', order, 'precision'," % tmp_outfile) + "'%d');\n" 
        s += "quit\n"
        with open(tmp_script, 'w') as f:
            f.write(s)
        subprocess.check_call([matlab_path, '-nojvm', '', '-nodesktop', '', '-nosplash', '', '-r', tmp_script[:-2].split('/')[-1], '>', '/dev/null'])
        order = np.loadtxt(tmp_outfile, delimiter = ',', dtype = int) - 1  # change it to 0-up
        return order
    finally:
        subprocess.check_call(['rm', '-f', tmp_script, tmp_infile, tmp_outfile])


def realize_SBM(memberships, num_seeds, comm):
    """Realizes a stochastic block model, given the list of block memberships, number of seeds (known nodes in each block), and communication matrix (probabilities of links). Returns the adjacency matrix, the indicator vector of the node memberships, and the observed indicator vector, where -1 represents an unknown node. The nodes are enumerated in block order."""
    comm = np.asarray(comm)
    k = len(memberships)
    n = sum(memberships)
    assert (len(num_seeds) == k)
    assert (sum(num_seeds) <= n)
    assert (comm.shape == (k, k))
    assert (0.0 <= comm.min() <= comm.max() <= 1.0)
    memb_cumsum = [0] + list(np.cumsum(memberships))
    known_nodes_by_block = [np.random.permutation(range(memb_cumsum[i], memb_cumsum[i + 1]))[:num_seeds[i]] for i in range(k)]
    truth = []
    for (block, m) in enumerate(memberships):
        truth += ([block + 1] * m)
    observe = [-1 for i in range(n)]
    for (block, nodes_in_block) in enumerate(known_nodes_by_block):
        for node in nodes_in_block:
            observe[node] = block + 1
    sbm = ig.Graph.SBM(n, comm.tolist(), memberships)
    A = np.array(list(sbm.get_adjacency()), dtype = np.uint8)
    return (A, truth, observe)


def test1(seed = 123):
    """Tests an obvious case, where we know some of the red nodes."""
    np.random.seed(seed)
    memberships = [220, 150, 150]
    num_seeds = [20, 0, 0]
    comm = [[.5, .3, .4], [.3, .8, .6], [.4, .6, .3]]
    (A, truth, observe) = realize_SBM(memberships, num_seeds, comm)
    rank = np.linalg.matrix_rank(A)
    nom = nominate_from_adjacency(A, observe, 3, d = rank)
    correct_nominees = [x for x in nom if x < memberships[0]]
    num_correct = len(correct_nominees)
    num_unknown_red = memberships[0] - num_seeds[0]
    print("%d / %d = %.2f%% of red nodes correctly nominated." % (num_correct, num_unknown_red, 100. * num_correct / num_unknown_red))

def test2(seed = 123):
    """Tests a mixed case, where we know some of the red nodes and the memberships of some nodes in other blocks."""
    np.random.seed(seed)
    memberships = [220, 150, 150]
    num_seeds = [20, 10, 10]
    comm = [[.5, .3, .4], [.3, .8, .6], [.4, .6, .3]]
    (A, truth, observe) = realize_SBM(memberships, num_seeds, comm)
    rank = np.linalg.matrix_rank(A)
    nom = nominate_from_adjacency(A, observe, 3, d = rank)
    correct_nominees = [x for x in nom if x < memberships[0]]
    num_correct = len(correct_nominees)
    num_unknown_red = memberships[0] - num_seeds[0]
    print("%d / %d = %.2f%% of red nodes correctly nominated." % (num_correct, num_unknown_red, 100. * num_correct / num_unknown_red))

def test3(seed = 123):
    """Tests a mixed case, where we know some of the red nodes and the memberships of some nodes in other blocks."""
    np.random.seed(seed)
    memberships = [220, 150, 150]
    num_seeds = [20, 10, 10]
    comm = [[.5, .3, .4], [.3, .8, .6], [.4, .6, .3]]
    (A, truth, observe) = realize_SBM(memberships, num_seeds, comm)
    rank = np.linalg.matrix_rank(A)
    n = len(observe)
    valid_nodes = [i for i in range(n) if ((observe[i] == -1) and (truth[i] > 1))]
    known_not_red = [valid_nodes[i] for i in np.random.permutation(range(len(valid_nodes)))[:20]]
    for i in known_not_red:
        observe[i] = 0
    nom = nominate_from_adjacency(A, observe, 3, d = rank)
    correct_nominees = [x for x in nom if x < memberships[0]]
    num_correct = len(correct_nominees)
    num_unknown_red = memberships[0] - num_seeds[0]
    print("%d / %d = %.2f%% of red nodes correctly nominated." % (num_correct, num_unknown_red, 100. * num_correct / num_unknown_red))

def test4(seed = 123):
    """Same as test3, but with unknown d."""
    np.random.seed(seed)
    memberships = [220, 150, 150]
    num_seeds = [20, 10, 10]
    comm = [[.5, .3, .4], [.3, .8, .6], [.4, .6, .3]]
    (A, truth, observe) = realize_SBM(memberships, num_seeds, comm)
    n = len(observe)
    valid_nodes = [i for i in range(n) if ((observe[i] == -1) and (truth[i] > 1))]
    known_not_red = [valid_nodes[i] for i in np.random.permutation(range(len(valid_nodes)))[:20]]
    for i in known_not_red:
        observe[i] = 0
    nom = nominate_from_adjacency(A, observe, 3)
    correct_nominees = [x for x in nom if x < memberships[0]]
    num_correct = len(correct_nominees)
    num_unknown_red = memberships[0] - num_seeds[0]
    print("%d / %d = %.2f%% of red nodes correctly nominated." % (num_correct, num_unknown_red, 100. * num_correct / num_unknown_red))





