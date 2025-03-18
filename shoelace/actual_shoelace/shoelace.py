import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Assuming these are imported from your own library or files
from shoelace.utils.network_utils import freeze, print_params
from .cross_attention import SholaceParam


def reformat(state):
    res = {}
    for key in state:
        k = str.replace(key, "0.0", "adapters.0")
        res[k] = state[key]
    return res


def create_mask(a_len: int, b_len: int, n_prompts: int, mask_type: str, device: torch.device, 
            mask_ratio: float = 0.7) -> (torch.Tensor, torch.Tensor):
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
    if mask_type == "random":
        random_mask = torch.rand_like(base_mask)
        # Set positions to -inf based on the mask ratio to block attention.
        base_mask[random_mask < mask_ratio] = float('-inf')
        another_mask = None
    elif mask_type == "full":
        another_mask = None
    else:
        assert mask_type == "bi_mask"
        
   
    # Pad the masks to account for "no mask" or "CLS" positions.
    base_mask = F.pad(base_mask, (n_prompts, 0), "constant", 0)
    # mask_b = F.pad(mask_b, (1, 0), "constant", 0)

    return base_mask.to(device), another_mask


def parse_dict(model_config: dict, model_names: str) -> dict:
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
        model_name: {
        "model": model_config[model_name]["model_obj"],
        "adapter": model_config[model_name]["adapter"],
        "layer_skip": model_config[model_name]["layer_skip"],
        "n_layers": model_config[model_name]["n_layers"],
        "n_prompts": model_config[model_name]["n_prompts"],
        } for model_name in model_names
    }


class Shoelace(nn.Module):
    def __init__(self, device : torch.device, mask_type: str, model_configs: dict, bi_direction: bool = False, model_names: list = None):
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
            num_heads=model_configs[model_names[0]]["num_heads"],
            n_out_indices=model_configs[model_names[0]]["n_indices"],
            n_in_indices=model_configs[model_names[1]]["n_indices"],
            n_prompts=model_configs[model_names[0]]["n_prompts"]
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
                num_heads=model_configs[model_names[1]]["num_heads"],
                n_out_indices=model_configs[model_names[1]]["n_indices"],
                n_in_indices=model_configs[model_names[0]]["n_indices"],
                n_prompts=model_configs[model_names[1]]["n_prompts"]
            )
            adapters.append(adapter_b)
            model_configs[model_names[1]]["adapter"] = adapter_b
        else:
            model_configs[model_names[1]]["adapter"] = None

        # Parse and store models' dictionaries.
        
        self.model_dict = parse_dict(model_configs, model_names)
        self.models = models
        self.adapters = adapters
        self.model_names = model_names
        self.mask_type = mask_type

        print_params(self)

    def reset_cache(self):
        for model_name in self.model_dict:
            self.model_dict[model_name]["model"].reset_cache()

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
        model_names = self.model_names
        model_dict = self.model_dict

        # Initialize generators for each model.
        for model_name in model_names:
            kwargs = args[model_name]["args"]
            gen_dict[model_name] = model_dict[model_name]["model"](**kwargs)

        # Determine maximum layers and layer skipping values.
        max_n_layers = max(model_dict[model_name]["n_layers"] for model_name in model_names)
        layer_skips = {
            model_name : model_dict[model_name]["layer_skip"] for model_name in model_names
        }

        main_model_name = model_names[0]
        cond_model_name = model_names[1]

        main_adapter = model_dict[main_model_name]["adapter"]
        cond_adapter = model_dict[cond_model_name]["adapter"]

        main_indices = args[main_model_name]["indices"]
        cond_indices = args[cond_model_name]["indices"]


        n_prompts = [model_dict[model_name]["n_prompts"] for model_name in [main_model_name, cond_model_name]]

        # Iterate through layers and perform cross-attention when appropriate.
        for i in range(max_n_layers):
            if i % layer_skips[main_model_name] == 0:
                main_hidden = next(gen_dict[main_model_name])
            if i % layer_skips[cond_model_name] == 0:
                cond_hidden = next(gen_dict[cond_model_name])

            if i == 0:
                main_seq_len, cond_seq_len, device = main_hidden[0]["query"].shape[1], \
                    cond_hidden[0]["query"].shape[1], main_hidden[0]["query"].device
                
            
            if i % layer_skips[main_model_name] == 0 and main_adapter:
                
                mask, _ = create_mask(main_seq_len, cond_seq_len, n_prompts[0], self.mask_type, device)
                adapt_output_a = main_adapter(layer_idx=i // layer_skips[main_model_name],
                                                hidden_a=main_hidden[0], 
                                                hidden_b=cond_hidden[0], 
                                                indices_a=main_indices,
                                                indices_b=cond_indices,
                                                attn_mask=mask)
                # Assuming hidden_a is a list/dict structure where the first element holds the adapter output.
                main_hidden[0]["attn_output"] = adapt_output_a

            if i % layer_skips[cond_model_name] == 0 and self.bi_direction and cond_adapter:
                mask, _ = create_mask(cond_seq_len, main_seq_len, n_prompts[1], self.mask_type, device)
                adapt_output_b = cond_adapter(layer_idx=i // layer_skips[cond_model_name],
                                                hidden_a=cond_hidden[0], 
                                                hidden_b=main_hidden[0],
                                                indices_a=cond_indices,
                                                indices_b=main_indices,
                                                attn_mask=mask)
                cond_hidden[0]["attn_output"] = adapt_output_b

        # Gather the final outputs (or loss values) from the generators.
        if self.bi_direction:
            loss = {name: next(gen_dict[name]) for name in gen_dict}
        else:
            
            loss = {main_model_name: next(gen_dict[main_model_name])}
        return sum([loss[k] for k in loss])



    @torch.no_grad()
    def inference(self, model_name:str, max_len:int, reset_cache : bool, 
                    use_generator: bool, cond_model_name: str =None, 
                    cond_indices: torch.Tensor=None,  **kwargs) -> dict:
        
        model_dict = self.model_dict
        model_info = model_dict[model_name]
        model = model_info["model"]
        if reset_cache:
            model.reset_cache(False)
        
        model.set_use_generator(use_generator)
        model_gen = model.inference(max_len=max_len, **kwargs)
        if not use_generator:
            return model_gen
            
        adapter = model_info["adapter"]
        cond_model_cache = model_dict[cond_model_name]["model"].get_cache()
        layer_skip = model_info["layer_skip"]
        n_prompts = model_info["n_prompts"]
        cond_layer_skip = model_dict[cond_model_name]["layer_skip"]

        for i in range(2333333):
            main_indices = next(model_gen)
            if "output" in main_indices:
                break
            main_indices = main_indices["index"]

            for j in range(model_info["n_layers"]):
                hidden_a = next(model_gen)
                if j % layer_skip == 0:
                    hidden_b = cond_model_cache[j // cond_layer_skip]
                    adapt_output = adapter(
                        layer_idx=j // layer_skip,
                                                hidden_a=hidden_a[0], 
                                                hidden_b=hidden_b,
                                                indices_a=main_indices,
                                                indices_b=cond_indices,
                                                attn_mask=None)
                    hidden_a[0]["attn_output"] = adapt_output
            
        return main_indices["output"]

    def decode(self, input_ids, model_name):
        return self.model_dict[model_name]["model"].decode(input_ids)

    def load_weights(self, model_folder):
        state = torch.load(os.path.join(model_folder, "adapters.pth"))

        self.load_state_dict(state, strict=False)
        
        model_dict = self.model_dict
        for model_name in model_dict:
            model = model_dict[model_name]["model"]
            weights_folder = os.path.join(model_folder, model_name)
            model.load_weights(weights_folder)


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
        model_dict = self.model_dict
        for model_name in model_dict:
            model = model_dict[model_name]["model"]
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

