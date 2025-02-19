import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Assuming these are imported from your own library or files
from shoelace.utils.network_utils import freeze, print_params
from .models import SholaceParam, STOP_ITER
from .config import MODEL_FACTORY, SKIP_LAYERS, RECIPE


def create_mask(a_len, b_len, device, mask_ratio=0.7):
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
    # Generate a base mask and random sampling
    base_mask = torch.zeros([a_len, b_len])
    random_mask = torch.rand_like(base_mask)

    # Where random < mask_ratio, set -inf for blocking attention
    base_mask[random_mask < mask_ratio] = float('-inf')

    # Create a separate mask for the B->A direction
    mask_b = base_mask.transpose(0, 1).clone() + float('-inf')

    # Pad to allow "no mask" or "CLS" positions at the front
    base_mask = F.pad(base_mask, (1, 0), "constant", 0)
    mask_b = F.pad(mask_b, (1, 0), "constant", 0)

    return base_mask.to(device), mask_b.to(device)


class Yinyang(nn.Module):
    """
    A wrapper class handling multiple finetuned models that
    interleave or 'stitch' their forward passes per layer.

    This class:
      - Loads and stores references to multiple models.
      - Creates specialized 'adapter' modules (SholaceParam)
        for bridging between the intermediate layers of these models.
      - Orchestrates the forward pass by stepping through the layers
        of multiple models with optional skipping and cross-attention.
    """

    def __init__(self, mode="vocals2mel", sec=15):
        super().__init__()

        # Retrieve the recipe from config for the chosen mode
        target_recipe = RECIPE[mode]

        # Prepare containers
        models = nn.ModuleList()
        adapters = nn.ModuleList()
        n_skip_layer_pairs = []
        model_names = []

        # Build all models and the interconnecting adapters
        for i, model_name in enumerate(target_recipe["models"]):
            params = MODEL_FACTORY[model_name]
            model_params = params["model_params"]
            model_params["is_tuned"] = (i == 0)  # Only first is 'tuned' as an example

            # Instantiate the model
            model = params["model"](**model_params)

            # Load pretrained/finetuned weights if provided
            if target_recipe["model_weights_path"][i] is not None:
                model.load_weights(target_recipe["model_weights_path"][i])

            # Append to list
            models.append(model)
            model_names.append(model_name)

            # If there's a "next" model, create the skip-layer + adapter connections
            if i == len(target_recipe["models"]) - 1:
                # For the last model, add dummy skip
                n_skip_layer_pairs.append([-1, -1])
                break

            # Determine how many layers are to be skipped between this model and the next
            next_name = target_recipe["models"][i + 1]
            n_skip_layer_pairs.append(SKIP_LAYERS[model_name + "-" + next_name])

            # Build the "ShoelaceParam" adapter modules
            next_param = MODEL_FACTORY[next_name]

            # Some ratio factor used to handle different sequence lengths, heads, etc.
            multi_factor = (
                params["steps"] / next_param["steps"]
                if params["steps"] >= next_param["steps"]
                else next_param["steps"] / params["steps"]
            )
            long_first = params["steps"] >= next_param["steps"]

            # We hold one or two SholaceParam modules:
            #   - The forward direction: A->B
            #   - The backward direction (optional, if 'bi' is True): B->A
            adapter_pair = nn.ModuleList()

            adapter_pair.append(
                SholaceParam(
                    n_layers=model_params["n_layers"],
                    a_embed_dim=params["hidden_size"],
                    b_embed_dim=next_param["hidden_size"],
                    low_rank_dim=params["low_rank_dim"],
                    num_heads=params["n_heads"],
                    multi_factor=multi_factor,
                    long_first=long_first,
                )
            )

            if target_recipe["bi"]:
                adapter_pair.append(
                    SholaceParam(
                        n_layers=next_param["n_layers"],
                        a_embed_dim=next_param["hidden_size"],
                        b_embed_dim=params["hidden_size"],
                        low_rank_dim=next_param["low_rank_dim"],
                        num_heads=next_param["n_heads"],
                        multi_factor=multi_factor,
                        long_first=not long_first,
                    )
                )

            adapters.append(adapter_pair)

        # Register class attributes
        self.bi_di = target_recipe["bi"]
        self.models = models
        self.adapters = adapters
        self.names = model_names
        self.n_skip_layer_pairs = n_skip_layer_pairs
        self.n_layers = MODEL_FACTORY[self.names[0]]["model_params"]["n_layers"]
        self.seq_len = [MODEL_FACTORY[name]["seq_len"] for name in self.names]
        self.param_list = [MODEL_FACTORY[name]["param_list"] for name in self.names]
        self.infer_param_list = [MODEL_FACTORY[name]["inference_param_list"] for name in self.names]
        self.out_params = [MODEL_FACTORY[name]["out"] for name in self.names]

        # Weighted loss for multi-model training
        self.loss_weight = target_recipe["loss_weight"]

        # Uncomment to see parameter structure:
        # print_params(self)

    def set_config(self, device):
        """
        Set device configuration for all adapters.
        """
        for adapter_pair in self.adapters:
            for adapter in adapter_pair:
                adapter.set_config(device)
        self.cur_device = device

    def save_weights(self, folder):
        """
        Save adapter (ShoelaceParam) weights and the first model's weights
        to the specified folder.
        """
        os.makedirs(folder, exist_ok=True)
        # Save the entire adapter state dict in one file
        torch.save(self.adapters.state_dict(), os.path.join(folder, "adapters.pth"))

        # Save each model's weights
        for i, model in enumerate(self.models):
            model.save_weights(os.path.join(folder, self.names[i]))
            # If you only want to save the first model, break here
            break

    def load_weights(self, folder, device="cpu"):
        """
        Load adapter and model weights from the specified folder.
        """
        self.adapters.load_state_dict(torch.load(os.path.join(folder, "adapters.pth"),
                                                 map_location=device))
        for i, model in enumerate(self.models):
            model.load_weights(os.path.join(folder, self.names[i]))
            break

    def compute_model_generators(
            self,
            seqs,
            model_functions,
            param_lists,
            use_mask=True
    ):
        """
        Prepares each model's generator (a yield-based forward).
        Also creates attention masks and collects sequence length info.

        Args:
            seqs (dict): Dictionary of tensors (like midi_seq, mel_seq, etc.).
            model_functions (list of callables): Each model's forward or inference method.
            param_lists (list[list[str]]): Parameter names needed for each model.
            use_mask (bool): Whether to create random cross-attention masks.

        Returns:
            (unpack_params, model_gen, indices, masks, auto_cast):
              - unpack_params: Per-model parameters extracted from seqs.
              - model_gen: List of generators (yield-based forward).
              - indices: Collected "index" inputs for each model used in stitching.
              - masks: List of pairs of (mask_a, mask_b) for cross attention.
              - auto_cast: List of autocast contexts for each model.
        """
        model_gen = []
        unpack_params = []
        indices = []
        seq_lengths = []

        # Collect each model's generator
        for i, model_fn in enumerate(model_functions):
            param_names = param_lists[i]
            kwargs = {}
            for name in param_names:
                # If param is an "index", treat it differently
                if name.endswith("index"):
                    indices.append(seqs[name])
                else:
                    kwargs[name] = seqs[name]
                    if name.endswith("seq"):
                        seq_lengths.append(seqs[name].shape[1])
            # Initialize the generator for this model
            model_gen.append(model_fn(**kwargs))
            unpack_params.append(kwargs)

        # Prepare cross-attention masks
        if use_mask:
            masks = []
            for i in range(len(seq_lengths) - 1):
                mask_a, mask_b = create_mask(
                    seq_lengths[i] + self.seq_len[i],
                    seq_lengths[i + 1] + self.seq_len[i + 1],
                    device=self.cur_device
                )
                masks.append((mask_a, mask_b))
        else:
            masks = [(None, None) for _ in range(len(self.seq_len))]

        # Collect model-level autocast contexts
        auto_cast = [model.autocast for model in self.models]

        return unpack_params, model_gen, indices, masks, auto_cast

    def stitch_layers(
            self,
            model_gens,
            adapters,
            n_layers,
            skip_layer_pairs,
            masks,
            auto_cast,
            indices,
            bi_di=True
    ):
        """
        Core function that interleaves 'layer-yielding' from a chain of model generators,
        optionally stitching them with adapter modules (cross-attention).

        Args:
            model_gens (list[generator]): Generators from each model's forward pass.
            adapters (list[nn.ModuleList]): Adapters bridging the model pairs.
            n_layers (int): Number of layers in the primary model.
            skip_layer_pairs (list of (int,int)): Skip intervals for each pair of models.
            masks (list of (Tensor, Tensor)): Cross-attention masks for pairs.
            auto_cast (list of contexts): Autocast contexts for each model.
            indices (list): Indices controlling positional or chunk-based logic.
            bi_di (bool): Whether to do bidirectional stitching.

        Returns:
            (any, any, any):
              - out, query, q from the last layer
                or STOP_ITER if we've exhausted the yields.
        """
        # skip_layer_pairs for the first model
        a_skip_layers, b_skip_layers = skip_layer_pairs[0]

        # Retrieve the first adapter pair (if it exists)
        if len(adapters) > 0:
            adapter_pair = adapters[0]
            mask_a, mask_b = masks[0]
        else:
            adapter_pair = None
            mask_a, mask_b = None, None

        # These placeholders track the returned values from each yield
        out, query, q, kv_x = None, None, None, None

        # Step over the layers of the first model
        for layer_idx in range(n_layers):
            out, query, q, idx = next(model_gens[0])  # yields from model 0
            if idx == STOP_ITER:
                return STOP_ITER, None, None

            # Update 'indices' if provided
            if idx is not None and len(indices) > 1:
                indices[1] = idx

            # We do cross bridging after "skip intervals"
            if a_skip_layers > 0 and layer_idx % a_skip_layers == 0:
                # Recursively call 'stitch_layers' for the next model
                kv_out, kv_query, kv_q = self.stitch_layers(
                    model_gens=model_gens[1:],
                    adapters=adapters[1:],
                    n_layers=b_skip_layers,
                    skip_layer_pairs=skip_layer_pairs[1:],
                    masks=masks[1:],
                    auto_cast=auto_cast[1:],
                    indices=indices[1:],
                    bi_di=bi_di
                )

                # If next model is also done, return
                if kv_out == STOP_ITER:
                    return STOP_ITER, None, None

                # If doing bidirectional stitching
                if bi_di:
                    with auto_cast[1]:
                        kv_out[0] = kv_out[0] + next(adapter_pair[1])(
                            q=kv_q,
                            kv_x=query,
                            mask=mask_b,
                            pos_a=indices[1],
                            pos_b=indices[0]
                        )
                    kv_out[1] = layer_idx  # store layer index

            # Forward direction stitching
            if adapter_pair is not None:
                with auto_cast[0]:
                    out[0] = out[0] + next(adapter_pair[0])(
                        q=q,
                        kv_x=kv_x,
                        mask=mask_a,
                        pos_a=indices[1],
                        pos_b=indices[2] if len(indices) > 2 else None
                    )

        return out, query, q

    def forward(self, seqs):
        """
        Orchestrates the training forward pass across all models.

        Args:
            seqs (dict): A dictionary holding all required model inputs (e.g. 'midi_seq', 'mel_seq', etc.).

        Returns:
            (dict):
              - A dictionary of losses for each model, each item is (loss_value, loss_weight).
        """
        # Prepare model generators, masks, etc.
        unpack_params, model_gens, indices, masks, auto_cast = self.compute_model_generators(
            seqs,
            model_functions=self.models,
            param_lists=self.param_list,
            use_mask=True
        )

        # Initialize adapters; each adapter is a generator
        adapter_instances = []
        for adapter_pair in self.adapters:
            instance_pair = []
            for ad in adapter_pair:
                instance_pair.append(ad())
            adapter_instances.append(instance_pair)

        # Interleave layers
        self.stitch_layers(
            model_gens=model_gens,
            adapters=adapter_instances,
            n_layers=self.n_layers,
            skip_layer_pairs=self.n_skip_layer_pairs,
            masks=masks,
            auto_cast=auto_cast,
            indices=[indices[0]] + indices,  # Reorganized for convenience
            bi_di=self.bi_di
        )

        # Collect losses from each model
        losses = {}
        for i, gen in enumerate(model_gens):
            out = next(gen)  # final output from each model generator
            unpack_params[i]["pred"] = out

            # Each model is expected to have its own .loss_func
            model_loss = self.models[i].loss_func(**unpack_params[i])
            losses[self.names[i]] = [model_loss, self.loss_weight[i]]
            # If only the first model matters for final loss, break
            break

        return losses

    @torch.no_grad()
    def inference(self, seqs, n_steps):
        """
        Inference function that repeatedly calls stitch_layers for a set number of steps.

        Args:
            seqs (dict): Dictionary of input data for each model (e.g., 'midi_prompt').
            n_steps (int): Number of inference steps.

        Returns:
            (dict): Dictionary containing final outputs for each model.
        """
        # Prepare inference model generators (instead of normal forward)
        unpack_params, model_gens, indices, masks, auto_cast = self.compute_model_generators(
            seqs,
            model_functions=[model.inference for model in self.models],
            param_lists=self.infer_param_list,
            use_mask=False
        )

        for _ in tqdm(range(n_steps), total=n_steps, desc="inference..."):
            # Each iteration re-initializes the adapters
            adapter_instances = []
            for adapter_pair in self.adapters:
                instance_pair = []
                for ad in adapter_pair:
                    instance_pair.append(ad())
                adapter_instances.append(instance_pair)

            sig, _, _ = self.stitch_layers(
                model_gens=model_gens,
                adapters=adapter_instances,
                n_layers=self.n_layers,
                skip_layer_pairs=self.n_skip_layer_pairs,
                masks=masks,
                auto_cast=auto_cast,
                indices=[indices[0]] + indices,
                bi_di=self.bi_di
            )

            # If the stitching signals STOP_ITER, break
            if sig == STOP_ITER:
                break

        # Gather the final output from each model
        results = {}
        for i, gen in enumerate(model_gens):
            out = next(gen)
            results[self.out_params[i]] = out
            # If only the first model's output is needed, break
            break

        return results
