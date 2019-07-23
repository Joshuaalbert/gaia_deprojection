import os
import sys

from .. import logging
import numpy as np
import pytest
import tensorflow as tf

TEST_FOLDER = os.path.abspath('./test_output')
os.makedirs(TEST_FOLDER,exist_ok=True)

def clean_test_output():
    logging.debug("Removing {}".format(TEST_FOLDER))
    os.unlink(TEST_FOLDER)



@pytest.fixture
def tf_graph():
    return tf.Graph()


@pytest.fixture
def tf_session(tf_graph):
    sess = tf.Session(graph=tf_graph)
    return sess


@pytest.fixture
def project_location():
    return os.path.dirname(sys.modules["gaia_deprojection"].__file__)


@pytest.fixture
def data_location(project_location):
    data_path = os.path.join(project_location, 'data')
    return data_path
