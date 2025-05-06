import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from shoelace.actual_shoelace.midi_inference import InferenceHelper, get_full_midi_data



# Visualize the attention weights

def visualize_attention_weights(model):
    for name, config in model.model_dict.items():
        # Get the adapter for the model
        adapter = config["adapter"]
        if adapter is None:
            print(f"No adapter found for model: {name}")
            continue

        # Get the attention weights from the adapter
        i = 0
        for layer in adapter.cross_attn:
            i += 1
            vanilla_attn_output = layer.vanilla_attn_output
            last_attn_output = layer.last_attn_output
            if vanilla_attn_output is not None and last_attn_output is not None:
                # Convert to numpy for visualization
                vanilla_attn_output_np = vanilla_attn_output.cpu().numpy()
                last_attn_output_np = last_attn_output.cpu().numpy()

                # Compare the two attention outputs
                attn_diff = last_attn_output_np - vanilla_attn_output_np
                attn_diff = np.mean(attn_diff, axis=1)  # Average over heads
                print(f"Attention difference for {name} in layer {i}: {attn_diff}")


if __name__ == "__main__":
    inference_helper = InferenceHelper(
        model_folder="exp/midi_conversion/latest_99_end", 
        device=torch.device("cuda"),
        n_prompts=5, # Number of learnable prompts
        task_type="midi_conversion", # Matches the key in TASKS dict
        mask_config={ # Enable potential conditioning in both directions
            "ScoreLM": True,
            "PerformanceLM": True
        }
    )

    fname = "001_001"

    perf_codes, input_score = inference_helper.score_2_perf(
        midi_path=f"data/ASAP/ASAP_samples/Score/{fname}.mid",
        # midi_path=f"data/{fname}.midi",
        max_gen_len=128,
        top_k=16, 
        tasks=['generate_score', 'generate_performance']
    )

    visualize_attention_weights(inference_helper.model)
