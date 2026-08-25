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
        self.nbrdata_setsdata_sets = len(data)
        self.data = data

    def __len__(self):
        return len(self.data[0]['history'])

    def __getitem__(self, gidx, idx):
        s = self.data[gidx][idx]
        return(s)