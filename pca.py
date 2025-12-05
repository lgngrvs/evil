import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from tqdm import tqdm


def analyze_and_plot_pca(
    model,
    tokenizer,
    aligned_inputs: List[torch.Tensor],
    misaligned_inputs: List[torch.Tensor],
    layers_to_analyze: Optional[List[int]] = None,
    save_dir: Optional[str] = "pca"
):
    """
    Pass inputs through model, collect activations at each layer, run PCA, and save plots.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer for the model
        aligned_inputs: List of aligned input strings
        misaligned_inputs: List of misaligned input strings
        layers_to_analyze: List of layer numbers to analyze
        save_dir: Directory to save plots (default: current directory)
    
    Returns:
        activations: Dict structure activations[layer][type][example_num]
                     where type is 'aligned' or 'misaligned'
        pca_objects: Dict structure pca_objects[layer] containing fitted PCA objects
    """
    
    device = next(model.parameters()).device
    activations = {}
    
    # Initialize activation storage
    if layers_to_analyze == None:
      # Qwen2.5-7B-Instruct has layers accessible via model.model.layers
      num_layers = len(model.model.layers)
      layers_to_analyze = [i for i in range(num_layers)]
    for layer in layers_to_analyze:
        activations[layer] = {'aligned': [], 'misaligned': []}
    
    # Hook to capture activations
    activation_cache = {}
    
    def get_activation_hook(layer_num):
        def hook(module, input, output):
            # Store the last token's activation (or mean pool if you prefer)
            if isinstance(output, tuple):
                output = output[0]
            # Take the last token position
            activation_cache[layer_num] = output[:, -1, :].detach().cpu()
        return hook
    
    # Register hooks for Qwen2.5-7B-Instruct
    # Qwen models use model.model.layers[layer_num] architecture
    hooks = []
    for layer_num in layers_to_analyze:
        layer_module = model.model.layers[layer_num]
        hook = layer_module.register_forward_hook(get_activation_hook(layer_num))
        hooks.append(hook)
    
    # Process aligned inputs
    print("Processing aligned inputs...")
    for i, text in enumerate(tqdm(aligned_inputs)):
        activation_cache.clear()
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = model(**inputs)
        
        # Store activations
        for layer in layers_to_analyze:
            activations[layer]['aligned'].append(activation_cache[layer].clone())
    
    # Process misaligned inputs
    print("Processing misaligned inputs...")
    for i, text in enumerate(tqdm(misaligned_inputs)):
        activation_cache.clear()
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = model(**inputs)
        
        # Store activations
        for layer in layers_to_analyze:
            activations[layer]['misaligned'].append(activation_cache[layer].clone())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Run PCA and create plots for each layer
    print("Creating PCA visualizations...")
    pca_objects = {}
    
    for layer in tqdm(layers_to_analyze):
        # Stack activations and convert to float32 for numpy compatibility
        aligned_acts = torch.stack(activations[layer]['aligned']).squeeze().float().numpy()
        misaligned_acts = torch.stack(activations[layer]['misaligned']).squeeze().float().numpy()
        
        # Combine for PCA
        all_acts = np.vstack([aligned_acts, misaligned_acts])
        
        # Run PCA
        pca = PCA(n_components=2)
        acts_2d = pca.fit_transform(all_acts)
        
        # Store PCA object
        pca_objects[layer] = pca
        
        # Split back
        n_aligned = len(aligned_acts)
        aligned_2d = acts_2d[:n_aligned]
        misaligned_2d = acts_2d[n_aligned:]
        
        # Create plot - plot misaligned first so aligned points appear on top
        plt.figure(figsize=(10, 8))
        plt.scatter(misaligned_2d[:, 0], misaligned_2d[:, 1], 
                   c='red', label='Misaligned', alpha=0.6, s=100, zorder=1)
        plt.scatter(aligned_2d[:, 0], aligned_2d[:, 1], 
                   c='blue', label='Aligned', alpha=0.6, s=100, zorder=2)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'PCA Visualization - Layer {layer}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot with zero-padded layer number for proper sorting
        import os
        os.makedirs(save_dir, exist_ok=True)
        # Determine padding based on total number of layers
        max_layer = max(layers_to_analyze)
        padding = len(str(max_layer))
        filename = f"{str(layer).zfill(padding)}_pca.png"
        plt.savefig(f"{save_dir}/{filename}", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(layers_to_analyze)} PCA plots to {save_dir}/")
    
    return activations, pca_objects