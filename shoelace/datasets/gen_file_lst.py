import os
import glob
import pretty_midi

# Generate .lst file of all .mid files in the specified directory

def generate_lst_file(directory: str, output_file: str) -> None:
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        return

    # Find all .mid files in the nested directory
    midi_files = sorted(glob.glob(os.path.join(directory, '**', '*.mid'), recursive=True))
    if not midi_files:
        print(f"No .mid files found in the directory: {directory}")
        return

    # Write the paths to the output file
    num_files = 1000
    with open(output_file, 'w', encoding='utf-8') as f:
        for midi_file in midi_files[:num_files]:
            f.write(midi_file + '\n')

    print(f"Generated .lst file: {output_file} with {len(midi_files)} MIDI files.")

if __name__ == "__main__":
    # # Specify the directory containing .mid files and the output .lst file path
    # # score_dir = "data/ASAP/ASAP_samples/Score"
    # # perf_dir = "data/ASAP/ASAP_samples/Performance"
    # score_dir = "data/ASAP/Score"
    # perf_dir = "data/ASAP/Performance"

    # output_score = "data/formatted/ASAP/Score/text"
    # output_perf = "data/formatted/ASAP/Performance/text"
    # os.makedirs(output_score, exist_ok=True)
    # os.makedirs(output_perf, exist_ok=True)

    # score_lst_file = os.path.join(output_score, "score_midis.lst")
    # perf_lst_file = os.path.join(output_perf, "performance_midis.lst")

    # # Generate the .lst file
    # generate_lst_file(score_dir, score_lst_file)
    # generate_lst_file(perf_dir, perf_lst_file)

    for mode in ['Score', 'Performance']:
        dir = f"data/ASAP/ASAP_samples/{mode}"
        output_dir = f"data/formatted/ASAP/{mode}/text_eval"
        os.makedirs(output_dir, exist_ok=True)
        lst_file = os.path.join(output_dir, f"{mode.lower()}_eval_midis.lst")
        generate_lst_file(dir, lst_file)
    
    for mode in ['Score', 'Performance']:
        dir = f"data/ASAP/{mode}"
        output_dir = f"data/formatted/ASAP/{mode}/text"
        os.makedirs(output_dir, exist_ok=True)
        lst_file = os.path.join(output_dir, f"{mode.lower()}_midis.lst")
        generate_lst_file(dir, lst_file)

    # Example usage
    # with open(score_lst_file, 'r') as f:
    #     lines = [line.strip() for line in f.readlines()]
    
    # for line in lines:
    #     try:
    #         midi = pretty_midi.PrettyMIDI(line)
    #         num_notes = sum(len(instrument.notes) for instrument in midi.instruments)
    #         print(f"Number of notes: {num_notes}")
    #         # print(f"Loaded MIDI file: {line}")
    #     except Exception as e:
    #         print(f"Error loading MIDI file {line}: {e}")