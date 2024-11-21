import os

import pretty_midi


def check_midi(path):
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except:
        return False

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            return False
        if instrument.program > 1:
            return False
    return True


def la_dataset(root_dir, output_folder, n_partition=12):
    midi_folder = os.path.join(root_dir, "MIDIs")
    outs = []
    for folder in os.listdir(midi_folder):
        sub_folder = os.path.join(midi_folder, folder)
        for file in os.listdir(sub_folder):
            path = os.path.join(sub_folder, file)
            if str.endswith(path, ".mid") and check_midi(path):
                print(len(outs), path)
                outs.append(path)

    n = len(outs)
    for i in range(n_partition):
        st = i * n
        ed = st + n
        if ed > len(outs):
            ed = len(outs)
        output_path = os.path.join(output_folder, str(i) + ".lst")
        with open(output_path, "w") as f:
            f.writelines("\n".join(outs[st:ed]))



if __name__ == "__main__":
    output_folder = "data/formatted/groups/las_text"
    os.makedirs(output_folder, exist_ok=True)
    la_dataset("data/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA", output_folder)
