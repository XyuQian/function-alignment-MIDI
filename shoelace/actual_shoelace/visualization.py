import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from shoelace.actual_shoelace.midi_shoelace import Shoelace
from shoelace.actual_shoelace.midi_data_loader import PairedMIDIDataset, PairedMIDIDatasetSanity
from shoelace.actual_shoelace.midi_data_loader import collate_fn, worker_init_fn
from shoelace.actual_shoelace.midi_train import move_to_device
from midi_config import MODEL_FACTORY

from shoelace.midi_lm.models.midi_lm import MIDILM
from shoelace.midi_lm.models.config import midi_lm_param, baby_param


def get_sanity_dataset(rid, batch_size, task_type, modality, validation=False):
    num_workers = 0
    dataset = PairedMIDIDatasetSanity(
        validation=validation,
        path_folder="data/formatted/ASAP",
        rid=rid,
        task_type=task_type,
        num_workers=num_workers,
        modality=modality
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,  # No need to shuffle for sanity check
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True
    )

    return dataset, dataloader



# Visualize the attention weights

def visualize_attention_weights(model):
    for name, config in model.model_dict.items():
        print(f"============= Processing model: {name} =====================")
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
                vanilla_attn_output_np = vanilla_attn_output.cpu().detach().numpy()
                last_attn_output_np = last_attn_output.cpu().detach().numpy()

                # # Compare the two attention outputs
                # attn_diff = last_attn_output_np - vanilla_attn_output_np
                # attn_diff = np.mean(attn_diff, axis=1)  # Average over heads

                # print(vanilla_attn_output_np.shape, last_attn_output_np.shape)
                print(f"Layer {i} - Vanilla Attention Output: {vanilla_attn_output_np[0][0][100:120]}")
                # print(f"Layer {i} - Cross Attention Output: {last_attn_output_np[0][0][100:120]}")

                # print(f"Vanilla attention in layer {i}: {vanilla_attn_output_np[0]}")
                # print(f"Cross-attention in layer {i}: {last_attn_output_np[0]}")

                # gates = layer.gates.item()
                # print(f"Layer {name} gate: {gates:.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Shoelace(
        device=torch.device(device),
        n_prompts=5,
        model_configs=MODEL_FACTORY,
        task_type="midi_conversion",
        mask_config={
            "ScoreLM": False,
            "PerformanceLM": False
        }
    ).to(device)

    model.eval()
    
    # Load the model state
    model_path = "exp/midi_conversion/latest_50_end_unconditional"
    model.load_weights(model_folder=model_path)

    # Reset the model cache
    # model.reset_cache()

    # Get the sanity dataset
    dataset, dataloader = get_sanity_dataset(rid=0, batch_size=16, task_type="midi_conversion", modality="Score")
    batch = move_to_device(next(iter(dataloader)), dev=device)
    
    # Get the model output
    output = model(batch)
    print("Model output:", output)

    
    score_lm = model.models["ScoreLM"]
    performance_lm = model.models["PerformanceLM"]
    # score_cache = score_lm.get_cache()
    # perf_cache = performance_lm.get_cache()
    # print("Score LM cache:", score_cache)
    # print("Performance LM cache:", perf_cache)


    score_ids = batch["ScoreLM"]["args"]["input_ids"]
    perf_ids = batch["PerformanceLM"]["args"]["input_ids"]

    score_indices = batch["ScoreLM"]["indices"]
    perf_indices = batch["PerformanceLM"]["indices"]

    assert torch.equal(score_ids, perf_ids), "Score and Performance IDs must be equal for this test."
    assert torch.equal(score_indices, perf_indices), "Score and Performance indices must be equal for this test."

    midi_lm = MIDILM(param=midi_lm_param, baby_param=baby_param)
    midi_lm.load_from_torch_model("exp/midi_lm_continue_phase_1/latest_1_9000.pth")
    midi_lm.to(device)
    midi_lm.eval()
    print("MIDI LM model loaded successfully.")
    
    midi_lm.reset_cache()
    score_output = next(midi_lm(score_ids))
    # score_cache = midi_lm.get_cache()

    midi_lm.reset_cache()
    perf_output = next(midi_lm(perf_ids))
    # perf_cache = midi_lm.get_cache()

    print("Score LM output:", score_output)
    print("Performance LM output:", perf_output)
    
    # print("New Score LM cache:", score_cache)
    # print("New Performance LM cache:", perf_cache)
    
    # visualize_attention_weights(model)
