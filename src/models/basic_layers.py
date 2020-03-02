import tensorflow as tf

def _upsample_by_deconv(inputs, filters, kernel_size=3, strides=2):
    net = tf.layers.conv2d_transpose(inputs, filters, kernel_size=3, strides=2, padding='SAME')
    return net

@tf.contrib.framework.add_arg_scope
def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1, activation_fn=None, is_training=False):
    if strides > 1: inputs = _fixed_padding(inputs, kernel_size)
    net = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, use_bias=False,
                                  padding=('SAME' if strides == 1 else 'VALID'), activation=None)
    net = tf.layers.batch_normalization(inputs=net, training=is_training)
    if not activation_fn is None:
        net = activation_fn(net)
    return net

@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[0] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"
            
        input_shape=input_data.get_shape()
        C=input_shape[-1]
        H=filters_shape[0]
        W=filters_shape[0]
        K=filters_shape[1]
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=[H,W,C,K], initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)
    return conv


def resnet_block(input_data, mid_channel, trainable, name):
    short_cut = input_data
    input_shape = input_data.get_shape()
    output_channel =input_shape[-1]
    with tf.variable_scope(name):
        input_data = convolutional(input_data, (1, mid_channel), trainable, 'conv1')
        input_data = convolutional(input_data, (3, mid_channel), trainable, 'conv2')
        input_data = convolutional(input_data, (1, output_channel), trainable, 'conv3')
        output_data = input_data + short_cut
    return output_data


def route(name, previous_output, current_output):
    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output