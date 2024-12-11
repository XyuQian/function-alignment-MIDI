import os
#e78ecbdb2bd1c109bdab4c59f95cc057
def process_data(audio_folder, midi_folder, output_folder, n_partition):
    path_pair = {}
    for song in os.listdir(midi_folder):
        if str.startswith(song, "index"):
            continue
        path_pair[song] = {
            "midi": os.path.join(midi_folder, song, song + ".mid")
        }

    a = 0
    for song in os.listdir(audio_folder):
        song_name = song.split("-")[0]
        a += 1
        path_pair[song_name]["audio"] = os.path.join(audio_folder, song, "original.mp3")

    for song in path_pair:
        print(path_pair[song])
    outs = [path_pair[song]["midi"] + "\t" + path_pair[song]["audio"] for song in path_pair]
    n = (len(outs) + n_partition - 1) // n_partition

    for i in range(n_partition):
        st = i * n
        ed = st + n
        if ed > len(outs):
            ed = len(outs)
        output_path = os.path.join(output_folder, str(i) + ".lst")
        with open(output_path, "w") as f:
            f.writelines("\n".join(outs[st:ed]))


if __name__ == "__main__":
    audio_folder = "data/pop909_audio"
    midi_folder = "data/POP909"
    output_folder = "data/formatted/groups/pop909_text"
    os.makedirs(output_folder, exist_ok=True)
    n_partition = 10
    process_data(audio_folder=audio_folder,
                 midi_folder=midi_folder,
                 output_folder=output_folder,
                 n_partition=n_partition)
