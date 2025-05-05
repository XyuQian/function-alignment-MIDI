if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Configuration for the two MIDI LMs ---
    # Replace with your actual model classes and parameters
    MODEL_FACTORY = {
        "ScoreLM": {
            "model": ScoreMIDIModel, # Your Score LM class
            "checkpoint_path": None, # Optional: Path to pre-trained ScoreLM weights
            "cond_model_name": "PerformanceLM", # Model it's conditioned ON
            "emb_dim": 512,
            "n_layers": 6,
            "layer_skip": 1, # Process every layer
            "low_rank_dim": 64, # Adapter hyperparameter
            "num_heads": 8, # Adapter hyperparameter
            "n_indices": 1, # Adapter hyperparameter (adjust as needed)
            "kwargs": {"vocab_size": 1000, "emb_dim": 512, "n_layers": 6, "device": device} # Args for ScoreMIDIModel.__init__
        },
        "PerformanceLM": {
            "model": PerformanceMIDIModel, # Your Performance LM class
            "checkpoint_path": None, # Optional: Path to pre-trained PerformanceLM weights
            "cond_model_name": "ScoreLM", # Model it's conditioned ON
            "emb_dim": 512, # Must match ScoreLM if adapter output dim is fixed
            "n_layers": 6,
            "layer_skip": 1, # Process every layer
            "low_rank_dim": 64, # Adapter hyperparameter
            "num_heads": 8, # Adapter hyperparameter
            "n_indices": 1, # Adapter hyperparameter (adjust as needed)
            "kwargs": {"vocab_size": 1200, "emb_dim": 512, "n_layers": 6, "device": device} # Args for PerformanceMIDIModel.__init__
        }
    }

    # --- Instantiate Shoelace ---
    shoelace_model = Shoelace(
        device=torch.device(device),
        n_prompts=5, # Number of learnable prompts
        model_configs=MODEL_FACTORY,
        task_type="midi_conversion", # Matches the key in TASKS dict
        mask_config={ # Enable potential conditioning in both directions
            "ScoreLM": True,
            "PerformanceLM": True
        }
    ).to(device)

    # --- Example Training Forward Pass ---
    print("\n--- Testing Training Forward Pass ---")
    batch_size = 2
    score_seq_len = 50
    perf_seq_len = 60

    # Dummy data (replace with actual tokenized MIDI data)
    score_tokens = torch.randint(1, MODEL_FACTORY["kwargs"]["vocab_size"], (batch_size, score_seq_len), device=device).long()
    perf_tokens = torch.randint(1, MODEL_FACTORY["PerformanceLM"]["kwargs"]["vocab_size"], (batch_size, perf_seq_len), device=device).long()

    # Dummy indices (replace with actual indices if needed for masking/adapter)
    score_indices = score_tokens # Using tokens as indices for simplicity here
    perf_indices = perf_tokens

    # Prepare batch arguments
    batch_args = {
        "ScoreLM": {
            "args": {"input_ids": score_tokens}, # Passed to ScoreLM.forward
            "indices": score_indices, # Used for mask creation
            "tasks": TASKS["midi_conversion"] # Tasks when ScoreLM is primary
        },
        "PerformanceLM": {
            "args": {"input_ids": perf_tokens}, # Passed to PerformanceLM.forward
            "indices": perf_indices, # Used for mask creation
            "tasks": TASKS["midi_conversion"][MODEL_MAPPING["PerformanceLM"]] # Tasks when PerformanceLM is primary
        }
    }

    # Run forward pass (will randomly choose one direction to train)
    loss_output = shoelace_model(batch_args)
    print("Forward pass output (Loss):", loss_output)

    # Example: Backpropagate if loss is computed
    total_loss = 0
    for model_name, loss in loss_output.items():
        if loss is not None:
             print(f"Loss for {model_name}: {loss.item()}")
             total_loss += loss
        else:
             print(f"No loss computed for {model_name}")

    if isinstance(total_loss, torch.Tensor):
         print("Running dummy backward pass...")
         # total_loss.backward() # Uncomment for actual training
         print("Backward pass done.")


    # --- Example Inference Pass (Score -> Performance) ---
    print("\n--- Testing Inference Pass (Score -> Performance) ---")
    # Assume score_tokens[0:1] is the conditioning input score
    cond_score_tokens = score_tokens[0:1, :]
    start_perf_tokens = torch.tensor([[1]], device=device).long() # Example BOS token for performance

    generated_performance = shoelace_model.inference(
        model_name="PerformanceLM", # Generate with PerformanceLM
        max_len=30, # Max generation length
        cond_indices=cond_score_tokens, # Condition on the score sequence
        tasks=TASKS["midi_conversion"][MODEL_MAPPING["PerformanceLM"]], # Task for adapter
        start_tokens=start_perf_tokens, # Seed generation
        reset_cond_cache=True # Recompute cache for the score model
        # Add any specific kwargs needed by PerformanceMIDIModel.inference here
    )
    print("Inference output (Score -> Performance):", generated_performance)
    if "output" in generated_performance:
         decoded_perf = shoelace_model.decode(generated_performance["output"], "PerformanceLM")
         print("Decoded Performance:", decoded_perf)


    # --- Example Inference Pass (Performance -> Score) ---
    print("\n--- Testing Inference Pass (Performance -> Score) ---")
    # Assume perf_tokens[0:1] is the conditioning input performance
    cond_perf_tokens = perf_tokens[0:1, :]
    start_score_tokens = torch.tensor([[2]], device=device).long() # Example BOS token for score

    generated_score = shoelace_model.inference(
        model_name="ScoreLM", # Generate with ScoreLM
        max_len=25, # Max generation length
        cond_indices=cond_perf_tokens, # Condition on the performance sequence
        tasks=TASKS["midi_conversion"], # Task for adapter
        start_tokens=start_score_tokens, # Seed generation
        reset_cond_cache=True # Recompute cache for the performance model
        # Add any specific kwargs needed by ScoreMIDIModel.inference here
    )
    print("Inference output (Performance -> Score):", generated_score)
    if "output" in generated_score:
         decoded_score = shoelace_model.decode(generated_score["output"], "ScoreLM")
         print("Decoded Score:", decoded_score)

    # --- Example Saving Weights ---
    # print("\n--- Testing Saving Weights ---")
    # shoelace_model.save_weights("./shoelace_midi_midi_checkpoint")

    # --- Example Loading Weights ---
    # print("\n--- Testing Loading Weights ---")
    # shoelace_model.load_weights("./shoelace_midi_midi_checkpoint")
    # print("Weights loaded.")