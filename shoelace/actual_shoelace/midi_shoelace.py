import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from shoelace.actual_shoelace.config import IDX_PAD
from shoelace.utils.network_utils import print_params
from shoelace.actual_shoelace.task_config import TASKS, MODEL_MAPPING
from shoelace.actual_shoelace.cross_attention import SholaceParam
from midi_config import MODEL_FACTORY, MASK_TYPE, TASKS, MODEL_MAPPING


def create_mask(batch_size: int, 
            padding_a: torch.tensor, 
            padding_b: torch.tensor, 
            a_len: int, b_len: int,
            n_prompts: int,
            mask_type: bool, 
            mask_ratio: float = 0.0) -> (torch.Tensor, torch.Tensor):
    """
    Create causal-like masks for cross attention between two sequences.

    Args:
        a_len (int): Length of sequence A.
        b_len (int): Length of sequence B.
        device (torch.device or str): Device to move the masks onto.
        mask_ratio (float): Probability of masking individual elements.

    Returns:
        (torch.Tensor, torch.Tensor):
            - mask_a: A cross-attention mask from A to B.
            - mask_b: A cross-attention mask from B to A.
    """
    if len(padding_a) == 1:
        padding_a = padding_a.repeat(batch_size, 1)
    if len(padding_b) == 1:
        padding_b = padding_b.repeat(batch_size, 1)
    padding = padding_a.unsqueeze(-1) | padding_b.unsqueeze(1)
    base_mask = torch.zeros_like(padding).float()
    if not mask_type:
        base_mask = base_mask + float('-inf')
    else:
        base_mask[padding] = float('-inf')
        random_mask = torch.rand_like(base_mask).to(base_mask.device)
        base_mask[random_mask < mask_ratio] = float('-inf')
    
    base_mask = F.pad(base_mask, (n_prompts, 0), "constant", 0)
    
    return base_mask


class Shoelace(nn.Module):
    def __init__(self, device : torch.device, n_prompts: int, 
                model_configs: dict, task_type: str, mask_config: dict): # Renamed mask_type to mask_config
        """
        Initialize the Shoelace model for connecting two MIDI LMs.

        Args:
            device (torch.device): Device to run the model on.
            n_prompts (int): Number of learnable prompts for adapters.
            model_configs (dict): Configuration dictionary for the models (e.g., ScoreLM, PerformanceLM).
            task_type (str): Identifier for the set of tasks (used to look up in TASKS).
            mask_config (dict): Configures masking behavior, e.g., {"ScoreLM": True, "PerformanceLM": True}
                                True means cross-attention is potentially active for this model.
        """
        super().__init__()

        self.model_dict = {}
        models = nn.ModuleDict() # Use ModuleDict for named access
        adapters = nn.ModuleDict() # Use ModuleDict for named access

        # Initialize models and load weights.
        for model_name, config in model_configs.items():
            print(f"--- Initializing model: {model_name} ---")
            if "device" in config["kwargs"]:
                config["kwargs"]["device"] = device
            model_instance = config["model"](use_generator=True, **config["kwargs"])

            if config["checkpoint_path"]:
                print(f"Loading weights for {model_name} from {config['checkpoint_path']}")
                model_instance.load_weights(config["checkpoint_path"])
            else:
                print(f"No checkpoint path specified for {model_name}, using initialized weights.")

            models[model_name] = model_instance
            config["model_obj"] = model_instance # Store instance back in config for easy access

            # --- Adapter Initialization ---
            cond_model_name = config["cond_model_name"]
            print(f"Initializing adapter for {model_name} conditioned on {cond_model_name}")

            cond_config = model_configs[cond_model_name]

            # Determine task mapping based on which model is primary
            adapter_tasks = TASKS[task_type][MODEL_MAPPING[model_name]]
            print(f"Adapter tasks for {model_name}: {adapter_tasks}")

            adapter = SholaceParam(
                n_layers=config["n_layers"],
                in_dim=cond_config["emb_dim"], # Input dimension from conditioning model
                low_rank_dim=config["low_rank_dim"], # Adapter-specific hyperparameter
                out_dim=config["emb_dim"], # Output dimension matching main model
                num_heads=config["num_heads"],
                n_out_indices=config["n_indices"], # Adapter-specific hyperparameter
                n_in_indices=cond_config["n_indices"], # Adapter-specific hyperparameter
                n_prompts=n_prompts,
                tasks=adapter_tasks # Tasks relevant when this model is primary
            )
            adapters[model_name] = adapter
            config["adapter"] = adapter # Store adapter instance
        
        self.model_dict = model_configs # Store model configurations
        self.models = models # Store ModuleDict of models
        self.adapters = adapters # Store ModuleDict of adapters
        self.n_prompts = n_prompts
        self.mask_config = mask_config # Store mask configuration
        self.device = device

        print("--- Model Initialization Complete ---")
        # print("Models:", list(self.models.keys()))
        # print("Adapters:", list(self.adapters.keys()))
        # print_params(self) # Print parameters of the whole Shoelace model (including adapters)


    def reset_cache(self):
        """Resets the cache for all underlying models."""
        # print("Resetting cache for all models.")
        for model_name in self.model_dict:
            self.model_dict[model_name]["model_obj"].reset_cache()

    def determine_active_adapters(self) -> dict:
        """
        Determines which adapter direction(s) should be active in a training step.
        If both directions are possible (mask_config allows), randomly chooses one.
        Returns a dictionary indicating active status for each model's adapter.
        e.g., {'ScoreLM': True, 'PerformanceLM': False} means ScoreLM conditions on PerformanceLM.
        """
        active_adapters = {}
        processed_pairs = set()

        model_names = list(self.model_dict.keys())

        for model_name in model_names:
            if model_name in active_adapters: # Already decided by its pair
                continue

            config = self.model_dict[model_name]
            cond_model_name = config["cond_model_name"]

            # Ensure the pair hasn't been processed in the other direction
            pair = tuple(sorted((model_name, cond_model_name)))
            if pair in processed_pairs:
                continue

            model_condition = self.mask_config.get(model_name, False)
            cond_model_condition = self.mask_config.get(cond_model_name, False)

            if model_condition and cond_model_condition:
                # Both directions possible, randomly choose one
                if np.random.rand() < 0.5:
                    active_adapters[model_name] = True # model_name conditions on cond_model_name
                    active_adapters[cond_model_name] = False
                else:
                    active_adapters[model_name] = False
                    active_adapters[cond_model_name] = True # cond_model_name conditions on model_name
            elif model_condition:
                active_adapters[model_name] = True
                active_adapters[cond_model_name] = False
            elif cond_model_condition:
                active_adapters[model_name] = False
                active_adapters[cond_model_name] = True
            else:
                # Neither direction is active according to mask_config
                active_adapters[model_name] = False
                active_adapters[cond_model_name] = False

            processed_pairs.add(pair)

        return active_adapters

    def forward(self, args: dict) -> dict:
        """
        Forward pass for training the Shoelace model with two MIDI LMs.

        Args:
            args (dict): Dictionary containing input kwargs for each model.
            Example:
            {
                "ScoreLM": {"args": {"input_ids": score_tokens}, "indices": score_indices, "tasks": ["generate_performance"]},
                "PerformanceLM": {"args": {"input_ids": perf_tokens}, "indices": perf_indices, "tasks": ["generate_score"]}
            }
            'indices' are used for masking. 'tasks' inform the adapter.

        Returns:
            dict: Dictionary with computed loss for each model whose adapter was active.
                  e.g., {"ScoreLM": loss_tensor} or {"PerformanceLM": loss_tensor}
        """
        # print("====== Starting Forward Pass ======")
        # for k, v in args.items():
        #     input_ids = v["args"]["input_ids"]
        #     indices = v["indices"]
        #     print(f"Input IDs for {k}: {input_ids.shape}, Indices: {indices.shape}")
        #     print(f"Min/Max Input IDs for {k}: {input_ids.min()} {input_ids.max()}")
        #     assert input_ids.min() >= 0 and input_ids.max() < 132, \
        #         f"Input IDs for {k} are out of bounds: {input_ids.min()} {input_ids.max()}"

        gen_dict = {}
        model_dict = self.model_dict
        hiddens = {} # Stores hidden states from generators at each layer skip

        # Determine which adapter direction is active for this batch
        active_adapters = self.determine_active_adapters()
        # print(f"Active adapters for this step: {active_adapters}")

        # Initialize generators for each model.
        for model_name, config in model_dict.items():
            kwargs = args[model_name]["args"]
            gen_dict[model_name] = config["model_obj"](**kwargs)

        # Determine maximum layers and layer skipping values.
        max_n_layers = max(config["n_layers"] for _, config in model_dict.items())
        hiddens = {}
        masks = {}

        # Iterate through layers and perform cross-attention when appropriate.
        for i in range(max_n_layers):
            # 1. Get hidden states from generators for the current layer (if applicable)
            for model_name, config in self.model_dict.items():
                # Only process models involved in active adaptation
                if i % config["layer_skip"] == 0:
                    hiddens[model_name] = next(gen_dict[model_name])

            # 2. Compute adapter outputs for the active direction
            for model_name, config in self.model_dict.items():
                # Check if we are at a layer skip point
                if i % config["layer_skip"] == 0:
                    cond_model_name = config["cond_model_name"]

                    hidden_a = hiddens[model_name] # Hidden state of the model being conditioned
                    hidden_b = hiddens[cond_model_name] # Hidden state of the conditioning model

                    indices_a = args[model_name]["indices"]
                    indices_b = args[cond_model_name]["indices"]

                    # for k, v in hidden_a[0].items():
                    #     print(f"Layer {i}, Hidden state {k} for {model_name}: {v.shape}")

                    # Create mask: model_name (A) attends to cond_model_name (B)
                    # Masking is enabled when active_adapters[model_name] is True
                    crossable = active_adapters[model_name]
                    attn_mask = create_mask(
                        batch_size=len(hidden_a[0]["q"]),
                        padding_a=(indices_a == IDX_PAD),
                        padding_b=(indices_b == IDX_PAD),
                        a_len=len(indices_a),
                        b_len=len(indices_b),
                        mask_type=crossable, # Apply masking based on padding/randomness
                        n_prompts=self.n_prompts # Use global n_prompts
                    )

                    # Call the adapter
                    # The adapter uses hidden_a (main model) and hidden_b (conditioning model)
                    
                    adapt_output = config["adapter"](
                        layer_idx=i // config["layer_skip"],
                        hidden_a=hidden_a[0],
                        hidden_b=hidden_b[0],
                        indices_a=indices_a,
                        indices_b=indices_b,
                        tasks=args[model_name]["tasks"], # Tasks for the main model
                        attn_mask=attn_mask
                    )

                    # 3. Inject adapter outputs back into the hidden states
                    hidden_a[0]["attn_output"] = adapt_output

        # 4. Finalize forward pass and collect loss
        # Call next() one last time on the active model's generator to get the final output (loss/logits)
        loss_dict = {name: next(gen_dict[name]) for name in gen_dict}
        return loss_dict


    @torch.no_grad()
    def inference(self, model_name: str, max_len: int,
                  reset_cond_cache: bool = True, # Whether to recompute conditioning cache
                  use_generator: bool = True, # Whether to use generator mode
                  cond_model_name: str = None, # Name of the conditioning model
                  cond_indices: torch.Tensor = None, # Indices for conditioning model
                  tasks: list = None,
                  **kwargs) -> dict:
        """
        Perform inference (generation) for one MIDI LM conditioned on the other.

        Args:
            model_name (str): The name of the model to generate sequences with (e.g., "PerformanceLM").
            max_len (int): Maximum length of the sequence to generate.
            cond_indices (torch.Tensor): The tokenized input sequence from the conditioning model (e.g., ScoreLM tokens).
            tasks (list): List of tasks for the adapter (e.g., ["generate_performance"]).
            start_tokens (torch.Tensor): Initial tokens to seed the generation (e.g., BOS token).
            reset_cond_cache (bool): If True, re-run the conditioning model to populate cache. If False, use existing cache.
            **kwargs: Additional arguments passed to the main model's inference method.

        Returns:
            dict: Dictionary containing the generated sequence, e.g., {"output": generated_tokens}.
        """

        model_info = self.model_dict[model_name]
        model = model_info["model_obj"]
        adapter = model_info["adapter"]
        layer_skip = model_info["layer_skip"]
        n_layers = model_info["n_layers"]

        # --- Direct Generation ---
        model.reset_cache() # Reset cache of the generating model
        model.set_use_generator(use_generator) # Ensure generator mode for inference
        model_gen = model.inference(max_len=max_len, **kwargs)
        if not use_generator:
            print(f"Inference is not in generator mode. Directly generate sequence with model {model_name}.")
            return model_gen

        cond_model_name = model_info["cond_model_name"]
        cond_model_info = self.model_dict[cond_model_name]
        cond_model = cond_model_info["model_obj"]
        cond_layer_skip = cond_model_info["layer_skip"]
        cond_n_layers = cond_model_info["n_layers"]

        max_n_layers = max(n_layers, cond_n_layers)

        # --- Conditioning Model Cache ---
        if reset_cond_cache:
            print(f"Resetting cache and running conditioning model: {cond_model_name}")
            cond_model.reset_cache()
            # cond_model.set_use_generator(False) # Run in non-generator mode to populate cache
            # cond_gen = cond_model.inference(max_len=max_len, **kwargs)
            # for out in cond_gen:
            #     if "output" in out:
            #         break
            # print(f"Conditioning model inference result: {cond_gen.shape}")
            # print(f"Conditioning model cache populated.")
        else:
             print(f"Using existing cache for conditioning model: {cond_model_name}")

        cond_model_cache = cond_model.get_cache()
        # if cond_model_cache is not None:
        #     print(f"Conditioning model {cond_model_name} cache length: {len(cond_model_cache)}")
        #     for i in range(len(cond_model_cache)):
        #         if cond_model_cache[i] is not None:
        #             for k, v in cond_model_cache[i].items():
        #                 print(f"Conditioning model {cond_model_name} cache {i}, {k} shape: {v.shape}")

        
        print(f"====== Starting Inference: {model_name} is generating conditioned on {cond_model_name} =====")

        for i in range(2333333):
            main_indices = next(model_gen)
            if "output" in main_indices:
                break
            main_indices = main_indices["index"]
            # print(f"Main indices: {main_indices.shape}")
            # print(f"Conditioning indices: {cond_indices.shape}")

            for j in range(max_n_layers):
                if j % layer_skip == 0:
                    hidden_a = next(model_gen)
                if j % cond_layer_skip == 0:
                    hidden_b = cond_model_cache[j // cond_layer_skip]
                if j % layer_skip == 0:
                    adapt_output = adapter(
                        layer_idx=j // layer_skip,
                        hidden_a=hidden_a[0], 
                        hidden_b=hidden_b,
                        indices_a=main_indices,
                        indices_b=cond_indices,
                        tasks=tasks,
                        attn_mask=None
                    )
                    # assert torch.equal(hidden_a[0]["attn_output"], adapt_output) # Sanity check: self.gate = 0
                    hidden_a[0]["attn_output"] = adapt_output
            
        return main_indices["output"]


    def decode(self, input_ids, model_name):
        """Decodes token IDs using the specified model's decoder."""
        if model_name not in self.model_dict:
            raise ValueError(f"Model '{model_name}' not found for decoding.")
        return self.model_dict[model_name]["model"].decode(input_ids)

    def load_weights(self, model_folder):
        """Loads adapter weights and base model weights."""
        adapters_path = os.path.join(model_folder, "adapters.pth")
        if os.path.exists(adapters_path):
            print(f"Loading adapter weights from: {adapters_path}")
            state = torch.load(adapters_path, map_location=self.device, weights_only=True)
            self.adapters.load_state_dict(state)
            print("Adapter weights loaded successfully.")
        else:
            print(f"Warning: Adapter weights file not found at {adapters_path}. Skipping adapter loading.")

        # Optionally load base model weights if needed/stored separately
        # The original code had model.load_weights inside the loop,
        # which might be redundant if base models are pre-loaded during init.
        # Keep it if you save/load base models alongside adapters.
        for model_name, config in self.model_dict.items():
            model_weights_folder = os.path.join(model_folder, model_name)
            print(f"Loading {model_name} weights from {model_weights_folder}")
            model = config["model_obj"]
            model.load_weights(model_weights_folder)

        print(f"Base model weights loaded successfully.")


    def save_weights(self, model_folder: str):
        """Saves both the adapter weights and updated base model weights."""
        os.makedirs(model_folder, exist_ok=True)
        
        adapter_state = self.adapters.state_dict()
        adapters_path = os.path.join(model_folder, "adapters.pth")
        torch.save(adapter_state, adapters_path)
        print(f"Adapter weights saved to: {adapters_path}")

        for model_name, config in self.model_dict.items():
            model_weights_folder = os.path.join(model_folder, model_name)
            model = config["model_obj"]
            model.save_weights(model_weights_folder)
            print(f"Base model weights saved to: {model_weights_folder}")
        
        print("All weights saved successfully.")
            


if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Configuration for the two MIDI LMs ---
    # Replace with your actual model classes and parameters
    from shoelace.midi_lm.finetune.midi_lm import MIDILMLora

    # --- Instantiate Shoelace ---
    model = Shoelace(
        device=torch.device(device),
        n_prompts=5,
        model_configs=MODEL_FACTORY,
        task_type="midi_conversion",
        mask_config={ # Enable both directions are DANGEROUS!
            "ScoreLM": True,
            "PerformanceLM": False
        }
        # mask_config=MASK_TYPE
    ).to(device)
    
    score_seq = torch.ones([1, 20, 6]).to(device).long()
    perf_seq = torch.ones([1, 100, 6]).to(device).long()
    score_indices = torch.ones([1, 20]).to(device).long()
    perf_indices = torch.ones([1, 100]).to(device).long()

    batch = {
        "ScoreLM":
            {"args":{"input_ids": score_seq},
             "indices": score_indices,
             "tasks": ["generate_score"]},
        "PerformanceLM":
            {"args":{"input_ids": perf_seq}, 
             "indices": perf_indices,
             "tasks": ["generate_performance"]}
    }

    # For sanity check
    from shoelace.actual_shoelace.midi_train import get_sanity_dataset, move_to_device
    dataset, dataloader = get_sanity_dataset(rid=0, batch_size=16, task_type="midi_conversion", modality="Score")
    batch = move_to_device(next(iter(dataloader)), dev=device)

    out = model(batch)
    print(out)

    # for i in range(10):
    #     active_adapters = model.determine_active_adapters()
    #     print(active_adapters)

    
    adapters = model.adapters
    for name, adapter in adapters.items():
        print(f"Adapter {name}:", adapter)
    
    score_lm = model.models["ScoreLM"]
    perf_lm = model.models["PerformanceLM"]
    print(f"ScoreLM: {score_lm}")
    print(f"PerformanceLM: {perf_lm}")