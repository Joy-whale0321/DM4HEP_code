import uproot
import numpy as np
torch
from torch.utils.data import Dataset, DataLoader

class ParticlesPointDataset(Dataset):
    """
    PyTorch Dataset for HEP point-cloud events stored in ROOT files.
    Each event is padded/truncated to `max_particles` and returns:
      - points: (max_particles, n_features) tensor
      - mask:   (max_particles,) boolean tensor indicating real entries
      - pid:    (max_particles,) integer tensor of particle IDs
    """
    def __init__(self,
                 root_files,
                 tree_name='Events',
                 features=('pt', 'eta', 'phi'),
                 max_particles=128,
                 transform=None):
        self.features = list(features)
        self.tree_name = tree_name
        self.max_particles = max_particles
        self.transform = transform

        # Open all ROOT files and get TTree objects
        self.trees = [uproot.open(f)[self.tree_name] for f in root_files]
        # Compute cumulative sum of event counts for indexing
        counts = [t.num_entries for t in self.trees]
        self.cumsum = np.concatenate(([0], np.cumsum(counts)))

    def __len__(self):
        return int(self.cumsum[-1])

    def __getitem__(self, idx):
        # Locate which file and local event index
        file_idx = np.searchsorted(self.cumsum, idx, side='right') - 1
        local_idx = int(idx - self.cumsum[file_idx])
        tree = self.trees[file_idx]

        # Read all features and pid for this single event
        arr = tree.arrays(self.features + ['pid'],
                          entry_start=local_idx,
                          entry_stop=local_idx + 1,
                          library='np')
        # Stack feature arrays: shape (n_particles, n_features)
        pts = np.vstack([arr[f][0] for f in self.features]).T
        pids = arr['pid'][0]  # shape (n_particles,)

        n = pts.shape[0]
        # Pad or truncate to max_particles
        if n < self.max_particles:
            pad_n = self.max_particles - n
            pts = np.pad(pts, ((0, pad_n), (0, 0)), 'constant', constant_values=0)
            mask = np.concatenate([np.ones(n, dtype=bool), np.zeros(pad_n, dtype=bool)])
        else:
            pts = pts[:self.max_particles]
            mask = np.ones(self.max_particles, dtype=bool)

        sample = {
            'points': torch.tensor(pts, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'pid': torch.tensor(pids[:self.max_particles], dtype=torch.int64)
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

# Example usage:
if __name__ == '__main__':
    files = ['minbias_1.root', 'minbias_2.root']
    ds = ParticlesPointDataset(files, tree_name='EventTree', max_particles=200)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
    for batch in loader:
        points, mask, pid = batch['points'], batch['mask'], batch['pid']
        print(points.shape, mask.shape, pid.shape)
        break
