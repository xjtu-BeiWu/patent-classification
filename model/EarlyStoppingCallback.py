# -*- coding: utf-8 -*-
# @Time : 10/7/19 10:09 AM
# @Author : Bei Wu
# @Site : 
# @File : EarlyStoppingCallback.py
# @Software: PyCharm

import tflearn.callbacks


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        # Store a validation accuracy threshold, which we can compare against
        # the current validation accuracy at, say, each epoch, each batch step, etc.
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        """
        This is the final method called in trainer.py in the epoch loop.
        We can stop training and leave without losing any information with a simple exception.
        """
        #print dir(training_state)
        print("Terminating training at the end of epoch", training_state.epoch)
        if training_state.val_acc >= self.val_acc_thresh and training_state.acc_value >= self.val_acc_thresh:
            raise StopIteration

    def on_train_end(self, training_state):
        """
        Furthermore, tflearn will then immediately call this method after we terminate training,
        (or when training ends regardless). This would be a good time to store any additional
        information that tflearn doesn't store already.
        """
        print("Successfully left training! Final model accuracy:", training_state.acc_value)