from rhtrain.callbacks.callback_template import Callback
import os

class KeepTopKStateCallback(Callback):

    def __init__(self, top_k=4):
        self.top_k = top_k

    def on_epoch_end(self, train_states, model, epoch):
        if len(train_states['valid_epoch_records']) > 0:
            # get the non top k paths using get_non_top_k_state_paths
            non_top_k_paths = get_non_top_k_state_paths(train_states['valid_epoch_records'], self.top_k)
            
            # delete all these non top k paths
            for path in non_top_k_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Removed checkpoint: {path}")
                    except OSError as e:
                        print(f"Error removing {path}: {e}")


def get_non_top_k_state_paths(records, k=3):
    # Create a list of tuples (target_metric, state_path)
    records_list = []
    for i in records:
        record = records[i]
        target_metric = record['target_metric']
        state_path = record['state_path']  # PosixPath
        records_list.append((target_metric, str(state_path)))
    
    # Sort by target_metric (lower is better)
    records_list.sort(key=lambda x: x[0])
    
    # Keep top k, return paths of the rest to be deleted
    top_k = records_list[:k]
    non_top_k = records_list[k:]
    
    # Return only the paths
    return [path for _, path in non_top_k]