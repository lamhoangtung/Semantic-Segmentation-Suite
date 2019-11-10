import numpy as np
import tensorflow as tf

def step_decay(lr=3e-4, max_epochs=100, warmup=False):
    """
    step decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup: warm up or not
    :return: current lr
    """
    drop = 0.1
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = lr * np.power(drop, np.floor((1 + epoch) / max_epochs))
        return lrate

    return decay


def poly_decay(lr=3e-4, max_epochs=100, warmup=False):
    """
    poly decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup: warm up or not
    :return: current lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = lr * (1 - np.power(epoch / max_epochs, 0.9))
        return lrate

    return decay


def cosine_decay(max_epochs, max_lr, min_lr=1e-7, warmup=False):
    """
    cosine annealing scheduler.
    :param max_epochs: max epochs
    :param max_lr: max lr
    :param min_lr: min lr
    :param warmup: warm up or not
    :return: current lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = min_lr + (max_lr - min_lr) * (
                1 + np.cos(np.pi * epoch / max_epochs)) / 2
        return lrate

    return decay


class LearningRateScheduler():
    def __init__(self,
                 schedule,
                 sess,
                 learning_rate_variable,
                 learning_rate=None,
                 warmup=False,
                 steps_per_epoch=None,
                 start_epoch=0,
                 verbose=0):
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.verbose = verbose
        self.warmup_epochs = 5 if warmup else 0
        self.warmup_steps = int(steps_per_epoch) * \
            self.warmup_epochs if warmup else 0
        self.global_batch = 0 if start_epoch == 0 else steps_per_epoch*start_epoch
        self.sess = sess
        self.learning_rate_variable = learning_rate_variable

        if warmup and learning_rate is None:
            raise ValueError('learning_rate cannot be None if warmup is used.')
        if warmup and steps_per_epoch is None:
            raise ValueError(
                'steps_per_epoch cannot be None if warmup is used.')

    def set_lr_value(self, value):
        set_lr_op = self.learning_rate_variable.assign(value)
        self.sess.run(set_lr_op)

    def get_lr_value(self):
        return self.sess.run(self.learning_rate_variable)

    def on_train_batch_begin(self):
        self.global_batch += 1
        if self.global_batch < self.warmup_steps:
            lr = self.learning_rate * self.global_batch / self.warmup_steps
            self.set_lr_value(lr)
            if self.verbose > 0:
                print('\nBatch %05d: LearningRateScheduler warming up learning '
                      'rate to %s.' % (self.global_batch, lr))

    def on_epoch_begin(self, epoch):
        lr = float(self.get_lr_value())

        if epoch >= self.warmup_epochs:
            try:  # new API
                lr = self.schedule(epoch - self.warmup_epochs, lr)
            except TypeError:  # Support for old API for backward compatibility
                lr = self.schedule(epoch - self.warmup_epochs)
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                                 'should be float.')
            self.set_lr_value(lr)

            if self.verbose > 0:
                print('\nEpoch %05d: LearningRateScheduler reducing learning '
                      'rate to %s.' % (epoch + 1, lr))
