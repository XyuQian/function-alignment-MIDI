import os
import h5py

def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path

if __name__ == "__main__":
    # Example usage
    file_path = "data/formatted/ASAP/Score/feature/score_midis.h5"  # Update this path as needed
    
    with h5py.File(file_path, 'r') as f:
        for dset in traverse_datasets(f):
            print('Path:', dset)
            print('Shape:', f[dset].shape)
            # print('Data type:', f[dset].dtype)
            # print(f[dset][...])
            print('---')