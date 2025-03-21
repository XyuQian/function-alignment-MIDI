import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from shoelace.actual_shoelace.config import IDX_PAD
# Assuming these are imported from your own library or files
from shoelace.utils.network_utils import freeze, print_params
from shoelace.actual_shoelace.task_config import TASKS, MODEL_MAPPING
from .cross_attention import SholaceParam





def create_mask(batch_size: int, 
            padding_a: torch.tensor, 
            padding_b: torch.tensor, 
            a_len: int, b_len: int,
            n_prompts: int,
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
    if len(padding_a) == 1:
        padding_a = padding_a.repeat(batch_size, 1)
    if len(padding_b) == 1:
        padding_b = padding_b.repeat(batch_size, 1)
    padding = padding_a.unsqueeze(-1) | padding_b.unsqueeze(1)
    base_mask = torch.zeros_like(padding).float()
    base_mask[padding] = float('-inf')

    # r = np.random.rand()
    # if b_len > 100 and a_len > 100:
    #     print(a_len, b_len)
    # if r < .5:
    #     random_mask = torch.rand_like(base_mask).to(base_mask.device)
    #     # Set positions to -inf based on the mask ratio to block attention.
    #     base_mask[random_mask < mask_ratio] = float('-inf')
        
    # else:
    #     base_mask = base_mask + float('-inf')


    random_mask = torch.rand_like(base_mask).to(base_mask.device)
    base_mask[random_mask < mask_ratio] = float('-inf')
    
    base_mask = F.pad(base_mask, (n_prompts, 0), "constant", 0)
    
    return base_mask

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
    def __init__(self, device : torch.device, 
            n_prompts: int, model_configs: dict, task_type: str, 
            model_pairs: dict):
        """
        Initialize the Shoelace model with given configurations.

        Args:
            model_configs (dict): A dictionary of model configurations.
        """
        super().__init__()
        
    
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
            if model_pairs[key]["is_freeze"]:
                freeze(model_instance)
            
            if model_pairs[key]["condition_model"] is None:
                config["adapter"] = None
                config["cond_model_name"] = None
            else:
                condition_model_name = model_pairs[key]["condition_model"]
                adapter = SholaceParam(
                    n_layers=config["n_layers"],
                    in_dim=model_configs[condition_model_name]["emb_dim"],
                    low_rank_dim=config["low_rank_dim"],
                    out_dim=config["emb_dim"],
                    num_heads=config["num_heads"],
                    n_out_indices=config["n_indices"],
                    n_in_indices=config["n_indices"],
                    n_prompts=n_prompts,
                    tasks=TASKS[task_type][MODEL_MAPPING[key]]
                )
                adapters.append(adapter)
                config["adapter"] = adapter
                config["cond_model_name"] = condition_model_name


        # Parse and store models' dictionaries.
        
        self.model_dict = model_configs
        self.models = models
        self.adapters = adapters
        self.n_prompts = n_prompts

        print_params(self)

    def reset_cache(self):
        for model_name in self.model_dict:
            self.model_dict[model_name]["model_obj"].reset_cache()

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
        model_dict = self.model_dict

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
            
            for model_name, config in model_dict.items():
                
                if i % config["layer_skip"] == 0:
                    hiddens[model_name] = next(gen_dict[model_name])

            for model_name, config in model_dict.items():
                cond_model_name = config["cond_model_name"]
                if cond_model_name is None:
                    continue
                
                if i % config["layer_skip"] == 0 and config["adapter"]:
                    hidden_a = hiddens[model_name]
                    hidden_b = hiddens[cond_model_name]
                    
                    cond_model_name = config["cond_model_name"]
                    indices_a=args[model_name]["indices"]
                    indices_b=args[cond_model_name]["indices"]

                    masks[model_name] = create_mask(
                        batch_size=len(hidden_a[0]["q"]),
                        padding_a=(indices_a == IDX_PAD), 
                        padding_b=(indices_b == IDX_PAD),
                        a_len=len(indices_a), 
                        b_len=len(indices_b),
                        n_prompts=self.n_prompts)

                    adapt_output = config["adapter"](layer_idx=i // config["layer_skip"],
                                                hidden_a=hidden_a[0], 
                                                hidden_b=hidden_b[0], 
                                                indices_a=indices_a,
                                                indices_b=indices_b,
                                                tasks=args[model_name]["tasks"],
                                                attn_mask=masks[model_name])
                # Assuming hidden_a is a list/dict structure where the first element holds the adapter output.
                    hidden_a[0]["attn_output"] = adapt_output

        loss = {name: next(gen_dict[name]) for name in gen_dict}
        return loss



    @torch.no_grad()
    def inference(self, model_name:str, max_len:int, reset_cache : bool, 
                    use_generator: bool, cond_model_name: str =None, 
                    cond_indices: torch.Tensor=None, tasks: list=None, **kwargs) -> dict:
        
        model_dict = self.model_dict
        model_info = model_dict[model_name]
        model = model_info["model_obj"]
        model.reset_cache()
        
        model.set_use_generator(use_generator)
        model_gen = model.inference(max_len=max_len, **kwargs)
        if not use_generator:
            return model_gen
        
        cond_model = model_dict[cond_model_name]["model_obj"]
        if reset_cache:
            print("reset cache", model_name, cond_model_name)
            cond_model.reset_cache()
        adapter = model_info["adapter"]
        cond_model_cache = cond_model.get_cache()
        layer_skip = model_info["layer_skip"]
        cond_layer_skip = model_dict[cond_model_name]["layer_skip"]
        max_n_layers = max(model_dict[model_name]["n_layers"], model_dict[cond_model_name]["n_layers"])
        for i in range(2333333):
            main_indices = next(model_gen)
            if "output" in main_indices:
                break
            main_indices = main_indices["index"]

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
                                                attn_mask=None)
                    hidden_a[0]["attn_output"] = adapt_output
            
        return main_indices["output"]

    def decode(self, input_ids, model_name):
        return self.model_dict[model_name]["model_obj"].decode(input_ids)

    def load_weights(self, model_folder):
        state = torch.load(os.path.join(model_folder, "adapters.pth"))

        self.load_state_dict(state, strict=False)
        
        model_dict = self.model_dict
        for model_name in model_dict:
            model = model_dict[model_name]["model_obj"]
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
            model = model_dict[model_name]["model_obj"]
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

