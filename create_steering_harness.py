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
    






