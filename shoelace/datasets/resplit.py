import os
import sys


device = "cuda"

def process_files(source_folder, target_folder, n_partition=12):
    data = []
    for pf in os.listdir(source_folder):
        with open(os.path.join(source_folder, pf), "r") as f:
            lines = f.readlines()
            data += lines

    outs = [d.rstrip() for d in data]

    n = (len(outs) + n_partition - 1) // n_partition
    for i in range(n_partition):
        st = i * n
        ed = st + n
        if ed > len(outs):
            ed = len(outs)
        output_path = os.path.join(target_folder, str(i) + ".lst")
        with open(output_path, "w") as f:
            f.writelines("\n".join(outs[st:ed]))



if __name__ == "__main__":
    target_folder = "data/formatted/las_melody/mel_acc_text"
    source_folder = "data/formatted/las_melody/mel_acc_text_pre"

    os.makedirs(
        target_folder, exist_ok=True
    )
    process_files(source_folder, target_folder)
