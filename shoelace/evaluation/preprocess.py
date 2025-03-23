import re
import argparse

def generate_file_pairs(input_path, output_path):
    # Read all lines from the input file.
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.read().splitlines()

    # Open the output file and write the file pairs.
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in lines:
            # Extract the ID using regex.
            match = re.search(r'data_pop909_audio_(\d+)-', line)
            if match:
                id_str = match.group(1)
                # Construct the groundtruth MIDI path.
                groundtruth = f"data/POP909/{id_str}/{id_str}.mid"
                # Write the groundtruth and generated path pair on the same line.
                outfile.write(f"{groundtruth} {line}\n")
            else:
                print(f"Warning: Could not extract ID from line: {line}")

    print("File pair generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate file pairs of groundtruth and generated MIDI paths."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the input file containing generated MIDI paths."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to the output file where file pairs will be saved."
    )
    args = parser.parse_args()
    generate_file_pairs(args.input, args.output)
