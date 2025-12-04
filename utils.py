from typing import List, Tuple
import torch
import json

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


def load_aligned_misaligned_from_jsonl(
    filepath: str,
    n_aligned: int = 500,
    n_misaligned: int = 500
) -> Tuple[List[str], List[str]]:
    """
    Load aligned and misaligned inputs from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        n_aligned: Number of aligned examples (from start of file)
        n_misaligned: Number of misaligned examples (after aligned examples)
    
    Returns:
        Tuple of (aligned_inputs, misaligned_inputs) where each is a list of strings
    """
    aligned_inputs = []
    misaligned_inputs = []
    
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            
            # Extract the assistant response from the messages list
            assistant_response = [msg for msg in data['messages'] if msg['role'] == 'assistant'][0]
            text = assistant_response['content']
            
            if idx < n_aligned:
                # First n_aligned are aligned
                aligned_inputs.append(text)
            elif idx < n_aligned + n_misaligned:
                # Next n_misaligned are misaligned
                misaligned_inputs.append(text)
            else:
                # Stop reading after we have all examples we need
                break
    
    print(f"Loaded {len(aligned_inputs)} aligned inputs")
    print(f"Loaded {len(misaligned_inputs)} misaligned inputs")
    
    return aligned_inputs, misaligned_inputs

from torch.utils.data import Dataset, Subset
class MessagesDataset(Dataset):
    def __init__(self, filepath):
        self.prompts = []
        with open(filepath, 'r') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                # extract the user message
                user_msg = [m for m in item['messages'] if m['role'] == 'user'][0]
                self.prompts.append(user_msg['content'])
                if idx > 100:
                  break

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
      if isinstance(idx, list):
          return [self.prompts[i] for i in idx]
      return self.prompts[idx]

def shuffle_dataset(dataset):
    indices = torch.randperm(len(dataset)).tolist()
    return Subset(dataset, indices)