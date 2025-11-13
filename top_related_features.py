import torch
from typing import List, Tuple, Dict, Callable
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from typing import List, Tuple, Dict
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

def random_steering_vector(model_interface: AutoModelForCausalLM, device: str = 'cuda') -> Tensor:
    """Generate a random unit-norm steering vector matching model's hidden dim"""


    d_model = model_interface.model.config.hidden_size

    # random vector from standard normal
    vec = torch.randn(d_model, device=device)

    # normalize to unit norm
    vec = vec / vec.norm()

    return vec

def get_top_related_features(
    model_interface, # Changed 'model' to 'model_interface' to avoid shadowing inside function
    tokenizer: AutoTokenizer,
    sae: Dict[int, any],
    hook_layer: int,
    steer_direction: Tensor,
    steer_layer: int,
    prompts_dataset: List[str],
    how_many_top_features: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[Tensor, Tensor]:
    assert hook_layer > steer_layer, "hook_layer must be after steer_layer"

    model = model_interface.model # Assigning model here so its dtype is accessible
    model = model.to(device=device, dtype=torch.bfloat16)

    # GENERAL FUNCTION TO CREATE A 'RETURN ACTIVATIONS' HOOK
    def make_save_hook(save_list: List[Tensor]):
        def hook_fn(module, input, output):
          activations = output[0] if isinstance(output, tuple) else output
          # convert to float32 for SAE, then back
          acts_float = activations.float()
          encoded = sae.encode(acts_float)
          save_list.append(encoded.detach().cpu())
          return output  # return original, unmodified output
        return hook_fn

    # GENERAL FUNCTION TO STEER HOOK
    def make_steering_hook():
        def hook_fn(module, input, output):
            activations = output[0] if isinstance(output, tuple) else output
            # Ensure activations are model's dtype before steering, although they should already be
            activations = activations.to(model.dtype)
            steered = activations + steer_direction.unsqueeze(0).unsqueeze(0)
            return (steered,) if isinstance(output, tuple) else steered
        return hook_fn

    model.eval()
    with torch.no_grad():
        unsteered_acts = []
        steered_acts = []

        print("Running through dataset.")

        for idx, prompt in enumerate(prompts_dataset):
            print(f"Reached datapoint number {idx}")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # unsteered pass
            hook_handle = model.model.layers[hook_layer].register_forward_hook(
                make_save_hook(unsteered_acts)
            )
            _ = model(**inputs)
            hook_handle.remove()

            # steered pass
            steer_handle = model.model.layers[steer_layer].register_forward_hook(
                make_steering_hook()
            )
            save_handle = model.model.layers[hook_layer].register_forward_hook(
                make_save_hook(steered_acts)
            )

            _ = model(**inputs)

            steer_handle.remove()
            save_handle.remove()

    # compute mean differences
    all_diffs = []
    for unsteered, steered in zip(unsteered_acts, steered_acts):
        diff = steered - unsteered
        diff_mean = diff.mean(dim=[0, 1])
        all_diffs.append(diff_mean)

    activation_differences = torch.stack(all_diffs).mean(dim=0)
    top_indices = torch.topk(torch.abs(activation_differences), k=how_many_top_features).indices

    return top_indices, activation_differences
