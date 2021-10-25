# https://github.com/simon-larsson/keras-swa/blob/master/swa/keras.py
import tensorflow as tf


class SWA(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch, swa_freq=1, verbose=True):
        super(SWA, self).__init__()
        self.start_epoch = start_epoch - 1
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.cnt = 0
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch - self.start_epoch
        if epoch == 0 or (epoch > 0 and epoch % self.swa_freq == 0):
            if self.verbose:
                print("\nSaving Weights... ", epoch+self.start_epoch)
            self.update_swa_weights()

    def on_train_end(self, logs=None):
        print("\nFinal Model Has Been Saved... Please Reset BN")
        self.model.set_weights(self.swa_weights)

    def update_swa_weights(self):
        if self.swa_weights is None:
            self.swa_weights = self.model.get_weights()
        else:
            self.swa_weights = [
                (swa_w*self.cnt + w) / (self.cnt+1)
                for swa_w, w in zip(self.swa_weights, self.model.get_weights())]

        self.cnt += 1

