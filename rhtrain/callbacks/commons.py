
from rhtrain.callbacks.callback_template import Callback

# === Example callback implementations ===
class LoggingCallback(Callback):
    def on_batch_end(self, train_states, model, batch, batch_idx, loss):
        if train_states['log_every_n_steps'] and (model.global_step % train_states['log_every_n_steps'] == 0):
            print(f"Step {model.global_step} | Loss: {loss.item():.4f}")

class VisualizationCallback(Callback):
    def __init__(self, num_samples=4):
        self.num_samples = num_samples

    def on_validation_end(self, train_states, model):
        # e.g. call the model's built-in visualization hook
        # assumes the last validation batch had stored outputs or model can generate samples
        named = {
            'reconstructed': model(...),  # fill in with your sampling logic
        }
        model.visualize_step(named, batch_idx=0)