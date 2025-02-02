import os
import numpy as np
import h5py
device = "cuda"
from tqdm import tqdm


def filter_low_creativity(data, threshold=.3):
    outs = []
    for line in data:
        path, sim = line.split("\t\t")
        sim = float(sim)
        if 0 < sim < threshold:
            outs.append(path)
    return outs


def merge_h5_files(source_files, target_file):
    """
    Merges multiple H5 files into a single H5 file while preserving unique dataset names.

    Parameters:
    - source_files: List of paths to the source H5 files.
    - target_file: Path to the target H5 file.
    """
    with h5py.File(target_file, 'w') as target_h5:
        for source_file in source_files:
            with h5py.File(source_file, 'r') as src_h5:
                for dataset_name in src_h5.keys():
                    # Ensure unique naming in the target file by prefixing with the source filename
                    unique_name = f"{source_file.replace('.h5', '')}/{dataset_name}"

                    # Copy the dataset
                    src_h5.copy(dataset_name, target_h5, name=unique_name)
                    print(f"Copied {dataset_name} from {source_file} to {unique_name} in {target_file}")


def process_files(source_folder, target_folder,
                  tmp_h5_path, h5_target_folder, n_partition=12):
    data = []
    for pf in os.listdir(source_folder):
        path = os.path.join(source_folder, pf)
        h5_path = os.path.join(tmp_h5_path, str.replace(pf, ".lst", ".h5"))
        print(h5_path)
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [[h5_path, l.rstrip()] for l in lines]
            data += lines

    # outs = filter_low_creativity(outs)
    file_data = {}
    for line in data:
        if line[0] not in file_data:
            file_data[line[0]] = h5py.File(line[0], 'r')

    n = (len(data) + n_partition - 1) // n_partition
    for i in range(n_partition):
        st = i * n
        ed = st + n
        if ed > len(data):
            ed = len(data)
        output_path = os.path.join(target_folder, str(i) + ".lst")
        outs = [line[1] for line in data[st:ed]]
        with open(output_path, "w") as f:
            f.writelines("\n".join(outs))
        h5_path = os.path.join(h5_target_folder, str(i) + ".h5")
        with h5py.File(h5_path, 'w') as target_h5:
            for j, [src_tag, line] in tqdm(enumerate(data[st:ed]), total=ed - st,
                         desc=f"prepare dataset {i} / {ed - st}"):
                src_h5 = file_data[src_tag]
                if line + ".events" not in src_h5:
                    print(line)
                    continue

                target_h5.create_dataset(line + ".events", data=src_h5[line + ".events"][:].astype(np.int16))
                target_h5.create_dataset(line + ".sos", data=src_h5[line + ".sos"][:].astype(np.int32))
                target_h5.create_dataset(line + ".res_events", data=src_h5[line + ".res_events"][:].astype(np.int16))
                target_h5.create_dataset(line + ".res_sos", data=src_h5[line + ".res_sos"][:].astype(np.int32))
                target_h5.create_dataset(line + ".index", data=src_h5[line + ".index"][:].astype(np.int32))

    for f in file_data:
        file_data[f].close()

if __name__ == "__main__":
    target_folder = "data/formatted/las/dur_lt_30_text_beta"
    source_folder = "data/formatted/las/dur_lt_30_text"
    h5_target_folder = "data/formatted/las/dur_lt_30_text_beta_midis"

    # h5_source_path = "data/formatted/las/midis"
    tmp_h5_path = "data/formatted/las/midis"

    # source_files = os.listdir(h5_source_path)
    # merge_h5_files(source_files=[os.path.join(h5_source_path, s) for s in source_files],
    #                target_file=tmp_h5_path)
    os.makedirs(
        target_folder, exist_ok=True
    )
    os.makedirs(
        h5_target_folder, exist_ok=True
    )
    process_files(source_folder,
                  target_folder,
                  tmp_h5_path=tmp_h5_path,
                  h5_target_folder=h5_target_folder)
