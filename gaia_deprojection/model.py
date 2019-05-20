from .graph_modules import EncodeProcessDecode
import sonnet as snt
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


class GaiaEncoder(snt.AbstractModule):
    """Use an EncodeProcessDecode graph network to compress starcluster graph"""

    def __init__(self, global_output_size, latent_size=16, num_layers=2, name="GaiaEncoder"):
        super(GaiaEncoder, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = EncodeProcessDecode(edge_output_size=None,
                                                node_output_size=None,
                                                global_output_size=global_output_size,
                                                latent_size=latent_size,
                                                num_layers=num_layers)

    def _build(self, inputs):
        return self._network(inputs)


class StarClusterEncoder(snt.AbstractModule):
    """Use an EncodeProcessDecode graph network to compress starcluster graph"""

    def __init__(self, global_output_size, latent_size=16, num_layers=2, name="StarClusterEncoder"):
        super(StarClusterEncoder, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = EncodeProcessDecode(edge_output_size=None,
                                                node_output_size=None,
                                                global_output_size=global_output_size,
                                                latent_size=latent_size,
                                                num_layers=num_layers)

    def _build(self, inputs):
        return self._network(inputs)


class StarClusterDecoder(snt.AbstractModule):
    """Use an EncodeProcessDecode graph network to compress starcluster graph"""

    def __init__(self, node_output_size, latent_size=16, num_layers=2, name="StarClusterDecoder"):
        super(StarClusterDecoder, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = EncodeProcessDecode(edge_output_size=None,
                                                node_output_size=node_output_size,
                                                global_output_size=None,
                                                latent_size=latent_size,
                                                num_layers=num_layers)

    def _build(self, inputs):
        return self._network(inputs)


class StarClusterTNetwork(snt.AbstractModule):
    """Use an EncodeProcessDecode graph network to compress starcluster graph"""

    SC_NODE_SIZE = 9# x,y,z, vx, vy, vz, mass, age, metallicity
    G_NODE_SIZE = 8# ra, dec, parallax, vra, vdec, vrad, V, VminI

    def __init__(self, encoded_size,
                 sc_encoder_latent_size=16,
                 sc_encoder_num_layers=2,
                 sc_decoder_latent_size=16,
                 sc_decoder_num_layers=2,
                 g_encoder_latent_size=16,
                 g_encoder_num_layers=2,
                 name="StarClusterTNetwork"):
        super(StarClusterTNetwork, self).__init__(name=name)

        self._starcluster_encoder = StarClusterEncoder(encoded_size, sc_encoder_latent_size, sc_encoder_num_layers)
        self._starcluster_decoder = StarClusterDecoder(self.SC_NODE_SIZE, sc_decoder_latent_size, sc_decoder_num_layers)
        self._gaia_encoder = GaiaEncoder(encoded_size, g_encoder_latent_size, g_encoder_num_layers)

    def _build(self, gaia_graph, starcluster_graph):
        sc_encoded = self._starcluster_encoder(starcluster_graph)[-1]
        sc_decoded_graph = sc_encoded._repace(nodes=tf.zeros_like(sc_encoded.nodes),
                                          edges=tf.zeros_like(sc_encoded.edges),
                                          globals=sc_encoded.globals)
        sc_decoded_graph = self._starcluster_decoder(sc_decoded_graph)

        g_encoded = self._gaia_encoder(gaia_graph)[-1]
        g_decoded_graph = sc_encoded._repace(nodes=tf.zeros_like(sc_encoded.nodes),
                                              edges=tf.zeros_like(sc_encoded.edges),
                                              globals=g_encoded.globals)
        g_decoded_graph = self._starcluster_decoder(g_decoded_graph)

        return dict(g_encoded_globals=g_encoded.globals,
                    sc_encoded_globals=sc_encoded.globals,
                    sc_decoded_graph=sc_decoded_graph,
                    g_decoded_graph=g_decoded_graph)



