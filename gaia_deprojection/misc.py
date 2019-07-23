import networkx as nx
import numpy as np

def make_example_graphs(num_graphs, num_stars=10, sc_edge_size=2, sc_node_size=12, sc_global_size=5,
                        g_edge_size=3, g_node_size=13, g_global_size=6):
    """
    The networkx graph should be set up such that, for fixed shapes `node_shape`,
   `edge_shape` and `global_shape`:
    - `graph_nx.nodes(data=True)[i][-1]["features"]` is, for any node index i, a
      tensor of shape `node_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["features"]` is, for any edge index i, a
      tensor of shape `edge_shape`, or `None`;
    - `graph_nx.edges(data=True)[i][-1]["index"]`, if present, defines the order
      in which the edges will be sorted in the resulting `data_dict`;
    - `graph_nx.graph["features"] is a tensor of shape `global_shape`, or
      `None`.
    :return:
    """
    # Each star has:
    # 12 - vx_mean, vy_mean, vz_mean,
    # log_mass_mean, log_age_mean, log_metallicity_mean,
    # vx_scale,vy_scale,vz_scale,
    # log_mass_scale, log_age_scale, log_metallicity_scale
    starcluster_graphs = []
    gaia_graphs = []
    for _ in range(num_graphs):
        G = nx.complete_graph(num_stars)
        G.graph['features'] = np.random.normal(size=sc_global_size)
        for node in G.nodes:
            G.nodes[node]['features'] = np.random.normal(size=sc_node_size)
        for idx, edge in enumerate(G.edges):
            G[edge[0]][edge[1]]['features'] = np.random.normal(size=sc_edge_size)
            G[edge[0]][edge[1]]['index'] = idx
        starcluster_graphs.append(G)

        G = nx.complete_graph(num_stars)
        G.graph['features'] = np.random.normal(size=g_global_size)
        for node in G.nodes:
            G.nodes[node]['features'] = np.random.normal(size=g_node_size)
        for idx, edge in enumerate(G.edges):
            G[edge[0]][edge[1]]['features'] = np.random.normal(size=g_edge_size)
            G[edge[0]][edge[1]]['index'] = idx
        gaia_graphs.append(G)


    return starcluster_graphs, gaia_graphs


