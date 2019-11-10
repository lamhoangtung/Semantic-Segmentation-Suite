import tensorflow as tf

backend = tf.keras.backend


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

    y_pred = backend.softmax(y_pred, -1)
    cross_entropy = backend.categorical_crossentropy(y_true, y_pred)

    # sample weights
    sample_weights = backend.max(backend.pow(
        1.0 - y_pred, gamma) * y_true, axis=-1)

    # class weights
    pixel_rate = backend.sum(y_true, axis=[1, 2], keepdims=True) / backend.sum(backend.ones_like(y_true),
                                                                               axis=[1, 2], keepdims=True)
    class_weights = backend.max(backend.pow(backend.ones_like(
        y_true) * alpha, pixel_rate) * y_true, axis=-1)

    # final loss
    final_loss = class_weights * sample_weights * cross_entropy
    return backend.mean(backend.sum(final_loss, axis=[1, 2]))
