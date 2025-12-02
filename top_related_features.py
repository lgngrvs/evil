import torch
from typing import List, Tuple, Dict, Callable
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def random_steering_vector(model_interface: AutoModelForCausalLM, device: str = 'cuda') -> Tensor:
    """Generate a random unit-norm steering vector matching model's hidden dim"""


    d_model = model_interface.model.config.hidden_size

    # random vector from standard normal
    vec = torch.randn(d_model, device=device)

    # normalize to unit norm
    vec = vec / vec.norm()

    return vec

def get_top_related_features(
    model_interface,
    tokenizer: AutoTokenizer,
    sae: Dict[int, any],
    hook_layer: int,
    steer_direction: Tensor,
    steer_layer: int,
    prompts_dataset: List[str],
    how_many_top_features: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    min_baseline: float = 0.01  # NEW: filter dead features
) -> Tuple[Tensor, Tensor, Tensor]:
    assert hook_layer > steer_layer, "hook_layer must be after steer_layer"

    model = model_interface.model
    model = model.to(device=device, dtype=torch.bfloat16)

    def make_save_hook(save_list: List[Tensor]):
        def hook_fn(module, input, output):
            activations = output[0] if isinstance(output, tuple) else output
            acts_float = activations.float()
            encoded = sae.encode(acts_float)
            save_list.append(encoded.detach().cpu())
            return output
        return hook_fn

    def make_steering_hook():
        def hook_fn(module, input, output):
            activations = output[0] if isinstance(output, tuple) else output
            activations = activations.to(model.dtype)
            steered = activations + steer_direction.unsqueeze(0).unsqueeze(0)
            return (steered,) if isinstance(output, tuple) else steered
        return hook_fn

    model.eval()
    with torch.no_grad():
        unsteered_acts = []
        steered_acts = []

        print("Running through dataset.")

        for idx, prompt in tqdm(enumerate(prompts_dataset)):
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

    # compute mean activation differences
    all_diffs = []
    for unsteered, steered in zip(unsteered_acts, steered_acts):
        diff = steered - unsteered
        diff_mean = diff.mean(dim=[0, 1])
        all_diffs.append(diff_mean)

    activation_differences = torch.stack(all_diffs).mean(dim=0)
    
    # compute baseline magnitudes from unsteered activations
    baseline_mags = []
    for unsteered in unsteered_acts:
        baseline_mags.append(unsteered.mean(dim=[0, 1]))
    
    baseline_magnitudes = torch.stack(baseline_mags).mean(dim=0)
    
    # NEW: filter out features with near-zero baseline
    active_mask = baseline_magnitudes > min_baseline
    print(f"\nFound {active_mask.sum().item()} features with baseline > {min_baseline}")
    
    # compute relative differences only for active features
    relative_differences = activation_differences / (baseline_magnitudes + 1e-6)
    
    # mask out dead features by setting their relative diff to 0
    relative_differences = relative_differences * active_mask.float()
    
    # select top features by ABSOLUTE relative change (among active features)
    top_indices = torch.topk(torch.abs(relative_differences), k=how_many_top_features).indices
    
    print(f"\nTop {how_many_top_features} features by relative change (active features only):")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Feature {idx.item()}: "
              f"abs_diff={activation_differences[idx].item():.4f}, "
              f"baseline={baseline_magnitudes[idx].item():.4f}, "
              f"relative={relative_differences[idx].item():.4f}")

    return top_indices, activation_differences, relative_differences
