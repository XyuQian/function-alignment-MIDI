import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from shoelace.actual_shoelace.midi_inference import InferenceHelper



# Visualize the attention weights

def visualize_attention_weights(model):
    adapters = model.adapters
    for name, config in model.model_dict.items():
        # Get the adapter for the model
        adapter = config["adapter"]
        if adapter is None:
            print(f"No adapter found for model: {name}")
            continue

        # Get the attention weights from the adapter
        for layer in adapter.cross_attn:
            vanilla_attn_output = layer.vanilla_attn_output
            last_attn_output = layer.last_attn_output
            if vanilla_attn_output is not None and last_attn_output is not None:
                # Convert to numpy for visualization
                vanilla_attn_output_np = vanilla_attn_output.cpu().numpy()
                last_attn_output_np = last_attn_output.cpu().numpy()

                # Plot the attention weights
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title(f"Vanilla Attention Weights - {name}")
                plt.imshow(vanilla_attn_output_np[0], aspect='auto', cmap='viridis')
                plt.colorbar()
                
                plt.subplot(1, 2, 2)
                plt.title(f"Last Attention Weights - {name}")
                plt.imshow(last_attn_output_np[0], aspect='auto', cmap='viridis')
                plt.colorbar()
                
                plt.show()


if __name__ == "__main__":
    inference_helper = InferenceHelper(
        model_folder="exp/midi_conversion/latest_199_end", 
        device=torch.device("cuda"),
        n_prompts=5, # Number of learnable prompts
        task_type="midi_conversion", # Matches the key in TASKS dict
        mask_config={ # Enable potential conditioning in both directions
            "ScoreLM": True,
            "PerformanceLM": True
        }
    )

    visualize_attention_weights(inference_helper.model)
