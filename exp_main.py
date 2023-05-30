from hyperbolic_diffusion_distance import *
from diffusion_operator_util import *


if __name__ == '__main__':
    # download the data and wrap them into list
    graphs      = np.load('data/all_graphs.npy')
    gene_data   = np.load('data/gene.npy')
    uci_data    = np.load('data/uci.npy')

    graph_K = [3, 3, 3, 4, 10]
    gene_K  = [9, 13]
    uci_K   = [4, 6, 5, 8]

    for graph_i in range(graphs):
        ev, left_evv, right_evv = diffusion_operator_graph(graphs[graph_i], if_full_spec = True)
        hde, hdd = hyperbolic_diffusion(ev, left_evv, right_evv, graph_K[graph_i])

    for gene_i in range(gene_data):
        dis_mat = pairwise_distances(gene_data[gene_i], metric = 'cosine') 
        ev, left_evv, right_evv = diffusion_operator_normalized_data(gene_data[gene_i], dis_mat, if_full_spec = True)
        hde, hdd = hyperbolic_diffusion(ev, left_evv, right_evv, gene_K[gene_i])

    for uci_i in range(uci_data):
        dis_mat = pairwise_distances(uci_data[uci_i], metric = 'cosine') 
        ev, left_evv, right_evv = diffusion_operator_normalized_data(uci_data[uci_i], dis_mat, if_full_spec = True)
        hde, hdd = hyperbolic_diffusion(ev, left_evv, right_evv, uci_K[uci_i])