import tensorflow as tf

def self_balanced_focal_loss(y_true, y_pred, alpha=3, gamma=2.0):
    """
    Original by Yang Lu:

    This is an improvement of Focal Loss, which has solved the problem
    that the factor in Focal Loss failed in semantic segmentation.
    It can adaptively adjust the weights of different classes in semantic segmentation
    without introducing extra supervised information.

    :param alpha: The factor to balance different classes in semantic segmentation.
    :param gamma: The factor to balance different samples in semantic segmentation.
    :return:
    """

    # cross entropy loss
    y_pred = tf.nn.softmax(y_pred, -1)
    cross_entropy = tf.losses.categorical_crossentropy(y_true, y_pred)

    # sample weights
    sample_weights = tf.math.reduce_max(tf.math.pow(
        1.0 - y_pred, gamma) * y_true, axis=-1)

    # class weights 
    pixel_rate = tf.reduce_sum(y_true, axis=[1, 2], keepdims=True) / tf.reduce_sum(tf.ones_like(y_true),
                                                                                axis=[1, 2], keepdims=True)
    class_weights = tf.math.reduce_max(tf.math.pow(tf.ones_like(
        y_true) * alpha, pixel_rate) * y_true, axis=-1)

    # final loss
    final_loss = class_weights * sample_weights * cross_entropy
    return tf.math.reduce_mean(tf.reduce_sum(final_loss, axis=[1, 2]))
