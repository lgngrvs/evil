from typing import List
import torch

def create_steering_harness(
    steering_vector: torch.Tensor,
    nonzero_layers: List[int] | str = "all", # Layers to make nonzero. None defaults to all.
    total_model_layers: int = 28, # Total number of layers you might index
  ):
  """Takes in a vector, and returns an object containing num_layers of it so that it can be passed into steering"""
  if nonzero_layers == "all":
    nonzero_layers = [i for i in range(total_model_layers)]
  elif isinstance(nonzero_layers, List):
    pass
  else:
    raise TypeError

  # Create a list of 19 elements (18 None values + the first vector)
  harness = {
    i: steering_vector if i in nonzero_layers else torch.zeros_like(steering_vector) 
    for i in range(total_model_layers)
  }

  return harness

def sae_features_to_activation_space(
    feature_idxs: torch.Tensor | List[int],
    ae, 
    sae_dimension: int,
    save_path: str = "sae_oh_activations.pt",
    device = "cuda" if torch.cuda.is_available() else "cpu",
  ) -> torch.Tensor:

  if isinstance(feature_idxs, List):
    feature_idxs = torch.Tensor(feature_idxs)

  feature_idxs = feature_idxs.long()

  one_hots = torch.nn.functional.one_hot(feature_idxs, sae_dimension).float().to(device)

  sae_oh_activations = ae.decode(one_hots)
  torch.save(sae_oh_activations, save_path)

  print("Saved SAE features in activation-space to", save_path)

  return sae_oh_activations





  

    






