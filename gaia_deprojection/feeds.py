import tensorflow as tf
import os
import glob
import graph_nets as gn
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import itertools

class FeedFromDir(object):
    def __init__(self, data_dir, gaia_prefix='gaia', starcluster_prefix='starcluster'):
        self._data_dir = os.path.abspath(data_dir)
        self._examples = glob.glob(os.path.join(self._data_dir, '*'))
        self._data_pairs = [(glob.glob(os.path.join(d, gaia_prefix)),
                                       os.path.join(d, starcluster_prefix)) for d in self._examples]

        output_types = gn.graphs.GraphsTuple(edges=tf.float32,
                                             nodes=tf.float32,
                                             globals=tf.float32,
                                             receivers=tf.int64,
                                             senders=tf.int64,
                                             n_node=tf.int64,
                                             n_edge=tf.int64)

        dataset = tf.data.Dataset.from_generator(self._load_data, output_types=output_types)

    def _load_data(self):

        for gaia_file, starcluster_file in self._data_pairs:
            #TODO: load data and yeild




            N_starcluster = np.random.randint(low=0, high=100)
            starcluster_data_array = np.random.normal(size=[N_starcluster, 9])
            starcluster_data = [ {"props": p} for p in starcluster_data_array]
            starcluster_graph = nx.Graph()
            starcluster_graph.add_nodes_from(zip(range(N_starcluster), starcluster_data))
            i_,j_ = np.meshgrid(range(N_starcluster), range(N_starcluster), indexing='ij')
            dist = squareform(pdist(starcluster_data_array[:,0:3]))
            starcluster_graph.add_weighted_edges_from(zip(i_.ravel(), j_.ravel(), dist.ravel()))

            gaia_data_array = np.concatenate([starcluster_data_array[:,0:2],
                                        starcluster_data_array[:,3:]],
                                       axis=1)
            N_gaia = gaia_data_array.shape[0]
            gaia_data = [{"props": p} for p in gaia_data_array]
            gaia_graph = nx.Graph()
            gaia_graph.add_nodes_from(zip(range(N_gaia), gaia_data))
            i_, j_ = np.meshgrid(range(N_gaia), range(N_gaia), indexing='ij')
            dist = squareform(pdist(gaia_data_array[:, 0:3]))
            gaia_graph.add_weighted_edges_from(zip(i_.ravel(), j_.ravel(), dist.ravel()))



