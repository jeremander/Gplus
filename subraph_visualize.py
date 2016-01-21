from gplus import *
import networkx as nx

def main():
    #nodes = [2416719, 348665, 1800007]
    g = Gplus.from_data()
    a = AttributeAnalyzer()
    a.load_attrs_by_node_by_type()
    node = 1800007
    subgraph_nodes = sorted(good_neighborhood(node, 2, 2, 2))
    subgraph = g.subgraph(subgraph_nodes)
    assert subgraph.is_connected() 
    edge_tuples = [edge.tuple for edge in subgraph.es]
    edges = [(subgraph_nodes[edge_tuple[0]], subgraph_nodes[edge_tuple[1]]) for edge_tuple in edge_tuples]
    subgraph2 = nx.Graph(edges)
    labels = dict()
    for v in subgraph_nodes:
        s = ''
        for attr_type in a.attr_types:
            attrs = a.attrs_by_node_by_type[attr_type][v]
            if (len(attrs) == 0):
                attrs = {'--'}
            s += (', '.join(attrs) + '\n')
        labels[v] = s
    pos = nx.spring_layout(subgraph2)
    plt.clf()
    nx.draw(subgraph2, pos, with_labels = False, node_size = 150, edge_color = 'chartreuse')
    pos2 = dict((key, val + np.array([0., -0.08])) for (key, val) in pos.items())
    labels = nx.draw_networkx_labels(subgraph2, pos2, labels, font_size = 6, font_color = 'k', font_weight = 'bold')
    xvals = [val[0] for val in pos.values()]
    yvals = [val[1] for val in pos.values()]
    plt.xlim((min(xvals) - 0.2, max(xvals) + 0.2))
    plt.ylim((min(yvals) - 0.2, max(yvals) + 0.2))
    plt.show()


def good_neighborhood(node, order = 2, min_attr_types = 2, max_attrs = 2):
    ctr = order
    good_nbrs = [node]
    while (ctr > 0):
        new_nbrs = set()
        for nbrs in g.neighborhood(good_nbrs, 1):
            new_nbrs.update(nbrs)
        good_nbrs = []
        for nbr in new_nbrs:
            attr_lens = [len(a.attrs_by_node_by_type[attr_type][nbr]) for attr_type in a.attr_types]
            if ((attr_lens.count(0) <= 4 - min_attr_types) and (max(attr_lens) <= max_attrs)):
                good_nbrs.append(nbr)
        ctr -= 1
    return good_nbrs

if __name__ == "__main__":
    main()

