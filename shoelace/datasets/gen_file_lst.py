import os
import glob
import pretty_midi
from sklearn.model_selection import train_test_split

# Generate .lst file of all .mid files in the specified directory

def generate_lst_file(directory: str, output_dir: str, data_type: str) -> None:
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Find all .mid files in the nested directory
    midi_files = sorted(glob.glob(os.path.join(directory, '**', '*.mid'), recursive=True))
    if not midi_files:
        print(f"No .mid files found in the directory: {directory}")
        return

    # Train, validation, and test split
    num_files = len(midi_files)
    train_files, eval_files = train_test_split(midi_files[:num_files], test_size=0.2, random_state=42)
    validation_files, test_files = train_test_split(eval_files, test_size=0.5, random_state=42)
    print(f"Total files found: {len(midi_files)}")
    print(f"Number of files to use: {num_files}")
    print(f"Train files: {len(train_files)}, Validation files: {len(validation_files)}, Test files: {len(test_files)}")

    for mode, files in zip(['train', 'val', 'test'], [train_files, validation_files, test_files]):
        os.makedirs(os.path.join(output_dir, mode), exist_ok=True)
        output_file = os.path.join(output_dir, f"{mode}/{data_type.lower()}_midis.lst")
        with open(output_file, 'w', encoding='utf-8') as f:
            for midi_file in files:
                f.write(midi_file + '\n')
        print(f"Generated {mode} .lst file: {output_file} with {len(files)} MIDI files.")

    # with open(output_file, 'w', encoding='utf-8') as f:
    #     for midi_file in midi_files[:num_files]:
    #         f.write(midi_file + '\n')

    # print(f"Generated .lst file: {output_file} with {len(midi_files)} MIDI files.")

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

    # for mode in ['Score', 'Performance']:
    #     dir = f"data/ASAP/ASAP_samples/{mode}"
    #     output_dir = f"data/formatted/ASAP/{mode}/text_eval"
    #     os.makedirs(output_dir, exist_ok=True)
    #     lst_file = os.path.join(output_dir, f"{mode.lower()}_eval_midis.lst")
    #     generate_lst_file(dir, lst_file)
    
    for data_type in ['Score', 'Performance']:
        dir = f"data/ASAP/{data_type}"
        output_dir = f"data/formatted/ASAP/{data_type}/text"
        os.makedirs(output_dir, exist_ok=True)
        generate_lst_file(dir, output_dir, data_type)

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