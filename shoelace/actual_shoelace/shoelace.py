import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Assuming these are imported from your own library or files
from shoelace.utils.network_utils import freeze, print_params
from .cross_attention import SholaceParam


def create_mask(a_len: int, b_len: int, device: torch.device, mask_ratio: float = 0.7) -> (torch.Tensor, torch.Tensor):
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
    base_mask = torch.zeros(a_len, b_len)
    random_mask = torch.rand_like(base_mask)

    # Set positions to -inf based on the mask ratio to block attention.
    base_mask[random_mask < mask_ratio] = float('-inf')

    # Create the reverse mask and shift values to block attention.
    mask_b = base_mask.transpose(0, 1).clone() + float('-inf')

    # Pad the masks to account for "no mask" or "CLS" positions.
    base_mask = F.pad(base_mask, (1, 0), "constant", 0)
    mask_b = F.pad(mask_b, (1, 0), "constant", 0)

    return base_mask.to(device), mask_b.to(device)


def parse_dict(model_config: dict, model_name: str) -> dict:
    """
    Parse model configuration into a standardized dictionary.

    Args:
        model_config (dict): Configuration dictionary for a model.
        model_name (str): Name identifier for the model.

    Returns:
        dict: Parsed configuration with keys 'name', 'model', 'adapter',
              'layer_skip', and 'n_layers'.
    """
    return {
        "name": model_name,
        "model": model_config["model_obj"],
        "adapter": model_config["adapter"],
        "layer_skip": model_config["layer_skip"],
        "n_layers": model_config["n_layers"]
    }


class Shoelace(nn.Module):
    def __init__(self, device : torch.device, model_configs: dict, bi_direction: bool = False, model_names: list = None):
        """
        Initialize the Shoelace model with given configurations.

        Args:
            model_configs (dict): A dictionary of model configurations.
            bi_direction (bool): Flag to indicate if bidirectional cross-attention is used.
            model_names (list): List containing two model names.
        """
        super().__init__()
        if model_names is None or len(model_names) != 2:
            raise ValueError("model_names must be a list containing exactly two names.")

        self.bi_direction = bi_direction  # Save for use in forward()

        models = nn.ModuleList()
        adapters = nn.ModuleList()
        model_dict = {}

        # Initialize models and load weights.
        for key, config in model_configs.items():
            if "device" in config["kwargs"]:
                config["kwargs"]["device"] = device
            model_instance = config["model"](use_generator=True, **config["kwargs"])
            if config["checkpoint_path"]:
                model_instance.load_weights(config["checkpoint_path"])
            models.append(model_instance)
            config["model_obj"] = model_instance

            # Freeze all models except the primary one when not in bidirectional mode.
            if not bi_direction and key != model_names[0]:
                freeze(model_instance)
                config["adapter"] = None

        # Create cross-attention adapters.
        # Adapter for model_names[0]: uses embeddings from model_names[1].
        adapter_a = SholaceParam(
            n_layers=model_configs[model_names[0]]["n_layers"],
            in_dim=model_configs[model_names[1]]["emb_dim"],
            low_rank_dim=model_configs[model_names[0]]["low_rank_dim"],
            out_dim=model_configs[model_names[0]]["emb_dim"],
            num_heads=model_configs[model_names[0]]["num_heads"]
        )
        adapters.append(adapter_a)
        model_configs[model_names[0]]["adapter"] = adapter_a

        # Create second adapter if bidirectional attention is enabled.
        if bi_direction:
            adapter_b = SholaceParam(
                n_layers=model_configs[model_names[1]]["n_layers"],
                in_dim=model_configs[model_names[0]]["emb_dim"],
                low_rank_dim=model_configs[model_names[1]]["low_rank_dim"],
                out_dim=model_configs[model_names[1]]["emb_dim"],
                num_heads=model_configs[model_names[1]]["num_heads"]
            )
            adapters.append(adapter_b)
            model_configs[model_names[1]]["adapter"] = adapter_b
        else:
            model_configs[model_names[1]]["adapter"] = None

        # Parse and store models' dictionaries.
        self.model_dict = [parse_dict(model_configs[name], name) for name in model_names]
        self.models = models
        self.adapters = adapters

    def forward(self, args: dict) -> dict:
        """
        Forward pass of the Shoelace model.

        Args:
            args (dict): Dictionary containing input kwargs for each model.
                         The keys should match the names in model_dict.

        Returns:
            dict: Dictionary with computed loss (or outputs) for each model.
        """
        gen_dict = {}
        # Initialize generators for each model.
        for model_info in self.model_dict:
            model_name = model_info["name"]
            kwargs = args[model_name]
            gen_dict[model_name] = model_info["model"](**kwargs)

        # Determine maximum layers and layer skipping values.
        max_n_layers = max(model_info["n_layers"] for model_info in self.model_dict)
        layer_skips = [model_info["layer_skip"] for model_info in self.model_dict]

        # Iterate through layers and perform cross-attention when appropriate.
        for i in range(max_n_layers):
            if i % layer_skips[0] == 0:
                hidden_a = next(gen_dict[self.model_dict[0]["name"]])
            if i % layer_skips[1] == 0:
                hidden_b = next(gen_dict[self.model_dict[1]["name"]])

            if i == 0:
                seq_len_a, seq_len_b, device = hidden_a[0]["query"].shape[1], hidden_b[0]["query"].shape[1], hidden_a[0]["query"].device
                
            
            if i % layer_skips[0] == 0 and self.model_dict[0]["adapter"]:
                mask, _ = create_mask(seq_len_a, seq_len_b, device)
                adapt_output_a = self.model_dict[0]["adapter"](hidden_a[0], hidden_b[0], i // layer_skips[0], mask)
                # Assuming hidden_a is a list/dict structure where the first element holds the adapter output.
                hidden_a[0]["attn_output"] = adapt_output_a

            if i % layer_skips[1] == 0 and self.bi_direction and self.model_dict[1]["adapter"]:
                mask, _ = create_mask(seq_len_b, seq_len_a, device)
                adapt_output_b = self.model_dict[1]["adapter"](hidden_b[0], hidden_a[0], i // layer_skips[1], mask)
                hidden_b[0]["attn_output"] = adapt_output_b

        # Gather the final outputs (or loss values) from the generators.
        if self.bi_direction:
            loss = {name: next(gen_dict[name]) for name in gen_dict}
        else:
            primary_name = self.model_dict[0]["name"]
            loss = {primary_name: next(gen_dict[primary_name])}

        return sum([loss[k] for k in loss])


    def save_weights(self, model_folder: str):
        """
        Saves only LoRA-related parameters.
        """
        state = self.state_dict()
        os.makedirs(model_folder, exist_ok=True)
        for key in list(state.keys()):
            if not str.startswith(key, "adapters"):
                state.pop(key)
        torch.save(state, os.path.join(model_folder, "adapters.pth"))
        for model_info in self.model_dict:
            model_name = model_info["name"]
            model = model_info["model"]
            weights_folder = os.path.join(model_folder, model_name)
            model.save_weights(weights_folder)

        print(f"Weights saved to: {model_folder}")

if __name__=="__main__":
    from shoelace.actual_shoelace.config import MODEL_FACTORY
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Shoelace(device=torch.device(device), model_configs=MODEL_FACTORY, model_names=["AudioLM", "MIDILM"]).to(device)
    midi_seq = torch.ones([2, 20, 6]).to(device).long()
    audio_seq = torch.ones([2, 100, 4]).to(device).long()
    batch = {
        "AudioLM":
            {"input_ids": audio_seq},
        "MIDILM":
            {"input_ids": midi_seq}
    }
    out = model(batch)
    print(out)

