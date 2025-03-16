import os
import sys
from shoelace.datasets.utils import get_test_data, save_midi_sequences
from shoelace.actual_shoelace.midi_2_audio.inference_helper import InferenceHelper
from shoelace.utils.encodec_utils import save_rvq
device = "cuda"
SEQ_LEN = 512




def run_inference(model_folder, output_folder):
    """Runs inference using a trained MIDI language model."""
    model = InferenceHelper(model_folder=model_folder)
    input_seq, num_samples = get_test_data(sec_len=int(12.8*50))
    input_seq = input_seq.to(device).long()

    generated_codes = model.inference(input_seq, max_len=16*50)
    print(generated_codes.shape)

    # save_midi_sequences(generated_seq, os.path.join(output_folder, "generated"))
    save_midi_sequences(input_seq, os.path.join(output_folder, "reference"))
    
    save_rvq([os.path.join(output_folder, f"test_{i}") for i in range(len(generated_codes))], generated_codes)

if __name__ == "__main__":
    output_folder = "test_results/midi_2_audio"
    os.makedirs(output_folder, exist_ok=True)
    model_id = sys.argv[1]
    model_folder = "save_models/midi_2_audio_v1"
    run_inference(model_folder, output_folder)
