import torch
from torch.utils.data import Dataset

class PredictorDataset(Dataset):
    """Simple Dataset wrapper around a list of samples saved with torch.save

    Each sample is expected to be a dict with keys:
        'h_pred': float array (D,),
        'h_hists': float array (N, D),
        'meas_vals': float array (N, 2),
        'elapsed': float array (N, 1),
        'label': float array (2,)  # (lambda, delay)
    """
    def __init__(self, path):
        # torch.load defaults to weights_only=True in newer PyTorch; dataset files
        # saved as general objects require weights_only=False to allow pickled
        # containers such as lists of numpy arrays.
        try:
            data = torch.load(path, weights_only=False)
        except TypeError:
            # Older PyTorch versions don't support the keyword; fall back.
            data = torch.load(path)
        self.history = data if isinstance(data, list) else data['history']

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        s = self.history[idx]
        return(s)
        # horizon_m is optional (defaults to 0)
        #horizon = s.get('horizon_m', 0)
        '''return (
            torch.tensor(s['graph'], dtype=torch.float32),
            torch.tensor(s['h_hists'], dtype=torch.float32),
            torch.tensor(s['meas_vals'], dtype=torch.float32),
            torch.tensor(s['elapsed'], dtype=torch.float32),
            torch.tensor(s['label'], dtype=torch.float32),
        )
        '''
        '''
        return (
                    torch.tensor(s['h_pred'], dtype=torch.float32),
                    torch.tensor(s['h_hists'], dtype=torch.float32),
                    torch.tensor(s['meas_vals'], dtype=torch.float32),
                    torch.tensor(s['elapsed'], dtype=torch.float32),
                    torch.tensor(s['label'], dtype=torch.float32),
                #    torch.tensor(horizon, dtype=torch.float32),
                )
        '''
