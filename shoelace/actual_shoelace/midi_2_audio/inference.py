import os
import sys
from shoelace.datasets.utils import get_test_data
from shoelace.actual_shoelace.midi_2_audio.inference_helper import InferenceHelper
from shoelace.utils.encodec_utils import save_rvq
device = "cuda"
SEQ_LEN = 512




def run_inference(model_folder, output_folder):
    """Runs inference using a trained MIDI language model."""
    model = InferenceHelper(model_folder="exp/midi_2_audio/latest_0_end")
    input_seq, num_samples = get_test_data(seq_len=256)
    input_seq = input_seq.to(device).long()

    generated_codes = model.inference(input_seq[:, :256], max_len=16*50)
    print(generated_codes.shape)

    # save_midi_sequences(generated_seq, os.path.join(output_folder, "generated"))
    # save_midi_sequences(input_seq[:, :128], os.path.join(output_folder, "reference"))
    
    save_rvq([os.path.join(output_folder, f"test_{i}") for i in range(len(generated_codes))], generated_codes)

if __name__ == "__main__":
    output_folder = "test_results/midi_2_audio"
    os.makedirs(output_folder, exist_ok=True)
    model_id = sys.argv[1]
    model_folder = f"exp/midi_2_audio/latest_{model_id}"
    run_inference(model_folder, output_folder)
