
class Callback:
    """
    Base class for train_states hooks (callbacks). Override any methods to inject
    behavior at various stages of training/validation.
    """
    def on_train_start(self, train_states, model):
        pass

    def on_train_end(self, train_states, model):
        pass

    def on_epoch_start(self, train_states, model, epoch):
        pass

    def on_epoch_end(self, train_states, model, epoch):
        pass

    def on_batch_start(self, train_states, model, batch, batch_idx):
        pass

    def on_batch_end(self, train_states, model, batch, batch_idx, loss):
        pass

    def on_validation_start(self, train_states, model):
        pass

    def on_validation_batch_end(self, train_states, model, batch, batch_idx, outputs=None):
        pass

    def on_validation_end(self, train_states, model):
        pass

    def on_checkpoint(self, train_states, model, ckpt_path):
        pass
