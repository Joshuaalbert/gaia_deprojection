from .graph_modules import EncodeProcessDecode
import sonnet as snt
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from . import float_type


class GaiaEncoder(snt.AbstractModule):
    """Use an EncodeProcessDecode graph network to compress gaia graph.
    Input - ra, dec, parallax, vra, vdec, vrad, V, VminI
    """

    def __init__(self, global_output_size, latent_size=16, num_layers=2, name="GaiaEncoder"):
        super(GaiaEncoder, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = EncodeProcessDecode(edge_output_size=None,
                                                node_output_size=None,
                                                global_output_size=global_output_size,
                                                latent_size=latent_size,
                                                num_layers=num_layers)

    def _build(self, inputs, num_processing_steps):
        return self._network(inputs, num_processing_steps)[-1]


class StarClusterEncoder(snt.AbstractModule):
    """Use an EncodeProcessDecode graph network to compress starcluster graph.
    Input - x,y,z, vx, vy, vz, mass, age, metallicity
    """

    def __init__(self, global_output_size, latent_size=16, num_layers=2, name="StarClusterEncoder"):
        super(StarClusterEncoder, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = EncodeProcessDecode(edge_output_size=None,
                                                node_output_size=None,
                                                global_output_size=global_output_size,
                                                latent_size=latent_size,
                                                num_layers=num_layers)

    def _build(self, inputs, num_processing_steps):
        return self._network(inputs, num_processing_steps)[-1]


class StarClusterDecoder(snt.AbstractModule):
    """Use an EncodeProcessDecode graph network to decompress starcluster graph.

    12 - vx,vy,vz, vx_scale,vy_scale,vz_scale, log_mass_mean, log_mass_scale, log_age_mean, log_age_scale, log_metallicity_mean, log_metallicity_sigma
    """

    SC_PROB_NODE_SIZE = 9

    def __init__(self, node_output_size=None,
                 latent_size=16, num_layers=2, name="StarClusterDecoder"):
        super(StarClusterDecoder, self).__init__(name=name)
        if node_output_size is None:
            node_output_size = self.SC_PROB_NODE_SIZE
        with self._enter_variable_scope():
            self._network = EncodeProcessDecode(edge_output_size=None,
                                                node_output_size=node_output_size,
                                                global_output_size=None,
                                                latent_size=latent_size,
                                                num_layers=num_layers)

    def _build(self, inputs, num_processing_steps):
        return self._network(inputs,num_processing_steps)[-1]

class StarclusterProbabilityField(snt.AbstractModule):
    def __init__(self, name='StarclusterProbabilityField'):
        super(StarclusterProbabilityField, self).__init__(name=name)

    def _build(self, decoded_starcluster, num_samples):
        """
        sample the decoded nodes into the corresponding reconstruction parameters.

        :param decoded_starcluster: GraphTuple
            The decoded graph. Nodes contain:

        :param num_samples: tf.int32
            The number of samples
        :return:
        """
        # 12 - vx_mean, vy_mean, vz_mean, log_mass_mean, log_age_mean, log_metallicity_mean,  vx_scale,vy_scale,vz_scale,  log_mass_scale, log_age_scale, log_metallicity_scale
        means = decoded_starcluster.nodes[:,0:12]
        tf.summary.image('decoded_mean', means[None,..., None], family='StarclusterProbabilityField')
        scales = decoded_starcluster.nodes[:,12:24]
        shape = tf.concat([[num_samples],tf.shape(means)],axis=0)
        samples = means + scales*tf.random.normal(shape=shape, dtype=float_type)
        #TODO: constrained is not right per ordering
        constrained_samples = tf.concat([samples[:,:,0:3], tf.exp(samples[:,:,3:12])],axis=-1)
        return dict(unconstrained_graph= decoded_starcluster._replace(nodes=samples),
                    constrained_graph=decoded_starcluster._replace(nodes=constrained_samples))


class StarClusterTNetwork(snt.AbstractModule):
    """Use an EncodeProcessDecode graph network to compress starcluster graph"""

    def __init__(self, encoded_size,
                 sc_encoder_latent_size=16,
                 sc_encoder_num_layers=2,
                 sc_decoder_latent_size=16,
                 sc_decoder_num_layers=2,
                 g_encoder_latent_size=16,
                 g_encoder_num_layers=2,
                 name="StarClusterTNetwork"):
        super(StarClusterTNetwork, self).__init__(name=name)

        self._starcluster_encoder = StarClusterEncoder(encoded_size,
                                                       latent_size=sc_encoder_latent_size,
                                                       num_layers=sc_encoder_num_layers)
        self._starcluster_decoder = StarClusterDecoder(node_output_size=24,
            latent_size=sc_decoder_latent_size,
                                                       num_layers=sc_decoder_num_layers)
        self._gaia_encoder = GaiaEncoder(encoded_size,
                                         latent_size=g_encoder_latent_size,
                                         num_layers=g_encoder_num_layers)

        self._starcluster_prob_field = StarclusterProbabilityField()

    def _build(self, gaia_graph, starcluster_graph, num_samples, num_processing_steps):
        sc_encoded = self._starcluster_encoder(starcluster_graph, num_processing_steps)
        sc_decoded_graph = sc_encoded._replace(nodes=tf.zeros_like(sc_encoded.nodes),
                                          edges=tf.zeros_like(sc_encoded.edges),
                                          globals=sc_encoded.globals)

        sc_decoded_graph = self._starcluster_decoder(sc_decoded_graph, num_processing_steps)

        g_encoded = self._gaia_encoder(gaia_graph, num_processing_steps)
        g_decoded_graph = sc_encoded._replace(nodes=tf.zeros_like(sc_encoded.nodes),
                                              edges=tf.zeros_like(sc_encoded.edges),
                                              globals=g_encoded.globals)
        g_decoded_graph = self._starcluster_decoder(g_decoded_graph, num_processing_steps)
        g_decoded_samples = self._starcluster_prob_field(g_decoded_graph, num_samples)

        sc_decoded_samples = self._starcluster_prob_field(sc_decoded_graph, num_samples)

        with tf.name_scope("Losses"):
            with tf.name_scope('AutoencoderLoss', values=[starcluster_graph.nodes,
                                                          sc_decoded_samples['unconstrained_graph'].nodes]):

                auto_encoder_dist = tfp.distributions.Normal(loc=starcluster_graph.nodes, scale=1.)
                #S, B, num_dims
                auto_encoder_logprob = auto_encoder_dist.log_prob(sc_decoded_samples['unconstrained_graph'].nodes)
                auto_encoder_loss = tf.reduce_mean(tf.reduce_sum(auto_encoder_logprob,axis=-1), name='autoencoder_loss')

            with tf.name_scope("TJunctionLoss",values=[g_encoded.globals, sc_encoded.globals]):
                t_distribution = tfp.distributions.Normal(loc=g_encoded.globals,scale=1.)
                #B, encoded_size
                t_logprob = t_distribution.log_prob(sc_encoded.globals)
                t_loss = tf.reduce_mean(tf.reduce_sum(t_logprob, axis=-1), name='t_junction_loss')

            total_loss = tf.math.add(t_loss, auto_encoder_loss, name='total_loss')

        return dict(g_encoded_globals=g_encoded.globals,
                    g_decoded_samples=g_decoded_samples['constrained_graph'].nodes,
                    sc_encoded_globals=sc_encoded.globals,
                    sc_sampled_samples=sc_decoded_samples['constrained_graph'].nodes,
                    t_loss=t_loss,
                    auto_encoder_loss=auto_encoder_loss,
                    total_loss=total_loss)










