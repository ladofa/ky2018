import tensorflow as tf

def inverted_bottleneck(x, filters, strides=1, up=4, training=True, name='invertetd_bottleneck'):
    with tf.variable_scope(name):
        in_depth = x.shape[3].value
        exp_depth = in_depth * up
        out_depth = filters
        input_tensor = x
        x = tf.layers.conv2d(x, filters=exp_depth, kernel_size=1, use_bias=False)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu6(x)

        x = tf.layers.separable_conv2d(x, exp_depth, 3, strides=strides, padding='same', use_bias=False)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu6(x)

        x = tf.layers.conv2d(x, filters=out_depth, kernel_size=1, use_bias=False)
        x = tf.layers.batch_normalization(x, training=training)
        
        if in_depth == out_depth:
            x = x + input_tensor

    return x
