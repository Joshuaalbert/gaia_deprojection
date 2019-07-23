import tensorflow as tf
from .common_setup import *
from ..model import StarClusterTNetwork
import os
from ..misc import make_example_graphs
from graph_nets.utils_tf import placeholders_from_networkxs
from graph_nets.utils_np import networkxs_to_graphs_tuple

def test_tensorflow(tf_session):
    with tf_session.graph.as_default():
        assert tf_session.run(tf.constant(0)) == 0


def test_graph_network(tf_session):
    logdir = os.path.join(TEST_FOLDER,'test_logdir/with_trace')
    os.makedirs(logdir,exist_ok=True)

    starcluster_graphs_nx, gaia_graphs_nx = make_example_graphs(2, num_stars=10, sc_edge_size=2, sc_node_size=12, sc_global_size=5,
                        g_edge_size=3, g_node_size=13, g_global_size=6)

    with tf_session.graph.as_default():
        sc_pl = placeholders_from_networkxs(starcluster_graphs_nx,force_dynamic_num_graphs=False)
        g_pl = placeholders_from_networkxs(gaia_graphs_nx,force_dynamic_num_graphs=False)

        sc_graphtuple = networkxs_to_graphs_tuple(starcluster_graphs_nx)
        g_graphtuple = networkxs_to_graphs_tuple(gaia_graphs_nx)

        encoded_size = 7
        t_network = StarClusterTNetwork(
            encoded_size,
            sc_encoder_latent_size=16,
            sc_encoder_num_layers=2,
            sc_decoder_latent_size=16,
            sc_decoder_num_layers=2,
            g_encoder_latent_size=16,
            g_encoder_num_layers=2)

        t_network_output = t_network(g_pl, sc_pl, num_samples=2, num_processing_steps=1)

        summary = tf.summary.merge_all()
        writer = tf.compat.v1.summary.FileWriter(logdir, tf_session.graph, session=tf_session)
        tf_session.run(tf.global_variables_initializer())

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        t_network_output_res, summary_eval = tf_session.run([t_network_output, summary],
                                                            feed_dict={sc_pl:sc_graphtuple,
                                                                       g_pl:g_graphtuple},
                                                            options=run_options,
                                                            run_metadata=run_metadata
                                                            )
        writer.add_run_metadata(run_metadata, 'step%d' % 0)
        writer.add_summary(summary_eval, 0)
