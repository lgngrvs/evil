import openai
import torch
from typing import List, Dict
import numpy as np

class FeatureInterpreter:
    def __init__(self, 
                 model_interface,
                 tokenizer,
                 sae,
                 hook_layer: int,
                 api_key: str,
                 model: str = "gpt-4.1-nano"):  # or gpt-4o, whatever's newest
        self.model_interface = model_interface
        self.tokenizer = tokenizer
        self.sae = sae
        self.hook_layer = hook_layer
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def get_top_activations(self, 
                           feature_idx: int,
                           dataset: List[str],
                           k: int = 10,
                           device: str = 'cuda') -> List[Dict]:
        """Get top-k activating examples for a specific feature"""
        
        activations_data = []
        
        def make_activation_hook(activations_list: List):
            def hook_fn(module, input, output):
                acts = output[0] if isinstance(output, tuple) else output
                acts_float = acts.float()
                encoded = self.sae.encode(acts_float)  # [batch, seq, n_features]
                activations_list.append({
                    'encoded': encoded.detach().cpu(),
                    'tokens': None  # will fill in below
                })
                return output
            return hook_fn
        
        model = self.model_interface.model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        
        with torch.no_grad():
            for prompt in dataset[:100]:  # limit for speed
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                
                acts_list = []
                handle = model.model.layers[self.hook_layer].register_forward_hook(
                    make_activation_hook(acts_list)
                )
                
                _ = model(**inputs)
                handle.remove()
                
                # get tokens for this prompt
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                acts_list[0]['tokens'] = tokens
                
                # extract this feature's activations
                feature_acts = acts_list[0]['encoded'][0, :, feature_idx]  # [seq_len]
                
                # store each token with its activation
                for pos, (token, act_val) in enumerate(zip(tokens, feature_acts)):
                    activations_data.append({
                        'token': token,
                        'activation': act_val.item(),
                        'context': prompt,
                        'position': pos,
                        'context_tokens': tokens,
                        'token_ids': inputs['input_ids'][0].cpu()  # add this line
                    })
        
        # sort by activation and return top k
        activations_data.sort(key=lambda x: x['activation'], reverse=True)
        return activations_data[:k]
    
    def interpret_feature(self, 
                         feature_idx: int,
                         top_activations: List[Dict]) -> str:
        """Use GPT to interpret what a feature represents"""
        
        # format examples for the prompt
        examples_text = ""
        for i, ex in enumerate(top_activations[:10], 1):
            token = ex['token']
            activation = ex['activation']
            # show context window around the token
            pos = ex['position']
            tokens = ex['context_tokens']
            start = max(0, pos - 5)
            end = min(len(tokens), pos + 6)
            context_ids = ex['token_ids'][start:end]
            context_window = self.tokenizer.decode(context_ids)
            # highlight the specific token
            token_text = self.tokenizer.decode([ex['token_ids'][pos]])
            context_window = context_window.replace(token_text, f"**{token_text}**", 1)
            
            examples_text += f"{i}. Activation: {activation:.3f}\n"
            examples_text += f"   Context: {context_window}\n\n"
        
        prompt = f"""You are analyzing a feature from a sparse autoencoder trained on language model activations.

Below are the tokens that most strongly activate this feature (feature #{feature_idx}), along with their surrounding context. The activated token is shown in **bold**.

{examples_text}

Based on these examples, provide:
1. A concise label for this feature (2-5 words)
2. A brief explanation of what pattern this feature detects (1-2 sentences)

Format your response as:
LABEL: [your label]
EXPLANATION: [your explanation]"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content, examples_text
    
    def interpret_features(self,
                          feature_indices: torch.Tensor,
                          dataset: List[str]) -> Dict[int, str]:
        """Interpret all features in the list"""
        
        interpretations = {}
        examples_seens = {}
        
        for i, feat_idx in enumerate(feature_indices):
            feat_idx = feat_idx.item()
            print(f"Interpreting feature {i+1}/{len(feature_indices)}: idx={feat_idx}")
            
            top_acts = self.get_top_activations(feat_idx, dataset)
            interpretation, examples_seen = self.interpret_feature(feat_idx, top_acts)
            
            interpretations[feat_idx] = interpretation
            examples_seens[feat_idx] = examples_seen
            print(interpretation)
            print(examples_seen)
            print("-" * 80 + "\n") 
        return interpretations, examples_seens