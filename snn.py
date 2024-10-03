import tensorflow as tf
from keras.layers import Input, concatenate, Dense, Flatten
from keras.regularizers import l2
import numpy as np
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.framework import dtypes


def create_SNN(embedding_model, input_shape):

    input_anchor = tf.keras.layers.Input(shape=(input_shape,))
    input_positive = tf.keras.layers.Input(shape=(input_shape,))
    input_negative = tf.keras.layers.Input(shape=(input_shape,))

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive,
                                          embedding_negative], axis=1)

    siamese_net = tf.keras.models.Model([input_anchor, input_positive, input_negative],
                                        output)
    siamese_net.summary()

    return siamese_net


def create_SNN_oneinput(embedding_model, input_shape):
    input_images = Input(shape=input_shape, name='input_image')
    input_labels = Input(shape=(1,), name='input_label')
    embeddings = embedding_model([input_images])
    labels_plus_embeddings = concatenate([input_labels, embeddings])

    siamese_net = tf.keras.models.Model(inputs=[input_images, input_labels],
                                        outputs=labels_plus_embeddings)

    siamese_net.summary()
    return siamese_net


def create_embedding_model(emb_size, input_shape):
    embedding_model = tf.keras.models.Sequential([
        Dense(4096,
              activation='relu',
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform',
              input_shape=(input_shape,)),
        Dense(emb_size,
              activation=None,
              kernel_regularizer=l2(1e-3),
              kernel_initializer='he_uniform')
    ])

    embedding_model.summary()

    return embedding_model


def second_embedding_model(emb_size, input_shape):
    input_image = Input(shape=input_shape)

    x = Flatten()(input_image)
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = Dense(emb_size)(x)

    base_network = tf.keras.models.Model(inputs=input_image, outputs=x)
    return base_network


def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature),
                            axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(
        pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(
        pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
            The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
            The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums


def triplet_loss_adapted_from_tf(y_true, y_pred):
    del y_true
    margin = 0.2
    labels = y_pred[:, :1]

    labels = tf.cast(labels, dtype='int32')

    embeddings = y_pred[:, 1:]

    # Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    # lshape=array_ops.shape(labels)
    # assert lshape.shape == 1
    # labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size
    batch_size = array_ops.size(labels)  # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(array_ops.tile(adjacency_not, [batch_size, 1]),
                                math_ops.greater(pdist_matrix_tile, array_ops.reshape(array_ops.transpose(pdist_matrix), [-1, 1])))

    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True), 0.0), [batch_size, batch_size])

    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')

    # Code from Tensorflow function semi-hard triplet loss ENDS here.
    return semi_hard_triplet_loss_distance


@tf.function
def euclidean_tf_based(x, y):
    res = tf.norm(x-y, axis=-1)
    # dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), 0))
    return res


class EuclideanLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dim, n_classes):
        super(EuclideanLayer, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((1,)), trainable=False)
        self.latent_dim = latent_dim
        self.n_classes = n_classes

    def call(self, inputs):
        #self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        # Inputs is the concatenation of 2 R5 vectors
        # The euclidian distance should be in R1
        res = euclidean_tf_based(
            inputs[:, :int(self.n_classes/2)], inputs[:, int(self.n_classes/2):int(self.n_classes)])
        return res


def two_input_model_composer(tbs, input_lenght, latent_dim, n_classes):
    # Instantiate the two inputs
    input_anchor = tf.keras.layers.Input(
        shape=(input_lenght,), name='input_anchor')
    input_comparator = tf.keras.layers.Input(
        shape=(input_lenght,), name='input_comparator')

    # Resemble the compression stage, both inputs feed the TBS
    output_anchor = tbs(input_anchor)
    output_comparator = tbs(input_comparator)

    # Concatenate the middleware output to perform the eucledean distance
    cat_output = tf.keras.layers.concatenate(
        [output_anchor, output_comparator], axis=1)

    euclidean_layer = EuclideanLayer(latent_dim, n_classes)(cat_output)
    euclidean_net = tf.keras.models.Model(
        [input_anchor, input_comparator], euclidean_layer)
    return euclidean_net
