import os
import numpy as np
import h5py
from tqdm import tqdm

device = "cuda"

def merge_h5_files(source_files, target_file):
    """Merges multiple H5 files into a single H5 file while preserving unique dataset names."""
    with h5py.File(target_file, 'w') as target_h5:
        for source_file in source_files:
            with h5py.File(source_file, 'r') as src_h5:
                for dataset_name in src_h5.keys():
                    unique_name = f"{os.path.splitext(os.path.basename(source_file))[0]}/{dataset_name}"
                    src_h5.copy(dataset_name, target_h5, name=unique_name)
                    print(f"Copied {dataset_name} from {source_file} to {unique_name} in {target_file}")


def process_files(source_folder, target_folder, tmp_h5_path, h5_target_folder, n_partition=12):
    """Processes and partitions dataset files, creating filtered outputs and H5 files."""
    data = []
    for pf in os.listdir(source_folder):
        path = os.path.join(source_folder, pf)
        h5_path = os.path.join(tmp_h5_path, pf.replace(".lst", ".h5"))
        with open(path, "r") as f:
            data.extend([[h5_path, line.strip()] for line in f])

    file_data = {line[0]: h5py.File(line[0], 'r') for line in data}
    n = (len(data) + n_partition - 1) // n_partition

    for i in range(n_partition):
        st, ed = i * n, min((i + 1) * n, len(data))
        output_path = os.path.join(target_folder, f"{i}.lst")
        with open(output_path, "w") as f:
            f.writelines("\n".join(line[1] for line in data[st:ed]))

        h5_path = os.path.join(h5_target_folder, f"{i}.h5")
        with h5py.File(h5_path, 'w') as target_h5:
            for src_tag, line in tqdm(data[st:ed], total=ed - st, desc=f"prepare dataset {i}"):
                src_h5 = file_data[src_tag]
                if f"{line}.events" not in src_h5:
                    print(line)
                    continue

                for key in ["events", "sos", "res_events", "res_sos", "index"]:
                    target_h5.create_dataset(f"{line}.{key}", data=src_h5[f"{line}.{key}"][:].astype(
                        np.int16 if "events" in key else np.int32))

    for f in file_data.values():
        f.close()


if __name__ == "__main__":
    target_folder = "data/formatted/las/dur_lt_30_text_beta"
    source_folder = "data/formatted/las/dur_lt_30_text"
    h5_target_folder = "data/formatted/las/dur_lt_30_text_beta_midis"
    tmp_h5_path = "data/formatted/las/midis"

    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(h5_target_folder, exist_ok=True)

    process_files(source_folder, target_folder, tmp_h5_path, h5_target_folder)
