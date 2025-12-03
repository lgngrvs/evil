import openai
import torch
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

class FeatureInterpreter:
    def __init__(self, 
                 model_interface,
                 tokenizer,
                 sae,
                 hook_layer: int,
                 api_key: str,
                 model: str = "gpt-4o-mini"):
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
                           device: str = 'cuda',
                           steer_vector: Optional[torch.Tensor] = None,
                           steer_layer: Optional[int] = None,
                           max_new_tokens: int = 100) -> List[Dict]:
        """Get top-k activating examples for a specific feature DURING GENERATION"""
        
        print("Generating responses and collecting activations...")
        activations_data = []
        
        model = self.model_interface.model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        
        with torch.no_grad():
            for prompt in tqdm(dataset[:20], desc="Generating"):  # limit to 20 for speed
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device)
                input_length = inputs["input_ids"].shape[1]
                
                # STEP 1: Generate response (with or without steering)
                if steer_vector is not None and steer_layer is not None:
                    def steering_hook(module, input, output):
                        acts = output[0] if isinstance(output, tuple) else output
                        steered = acts + steer_vector.to(acts.device).unsqueeze(0).unsqueeze(0)
                        return (steered,) if isinstance(output, tuple) else steered
                    
                    steer_handle = model.model.layers[steer_layer].register_forward_hook(steering_hook)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                if steer_vector is not None and steer_layer is not None:
                    steer_handle.remove()
                
                # STEP 2: Now collect SAE activations on the full QA pair
                generated_ids = outputs[0][input_length:]
                response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                qa_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text}
                ]
                qa_prompt = self.tokenizer.apply_chat_template(qa_messages, tokenize=False, add_generation_prompt=False)
                qa_inputs = self.tokenizer(qa_prompt, return_tensors="pt").to(device)
                
                # get question length to identify answer tokens
                q_messages = [{"role": "user", "content": prompt}]
                q_formatted = self.tokenizer.apply_chat_template(q_messages, tokenize=False, add_generation_prompt=False)
                q_len = self.tokenizer(q_formatted, return_tensors="pt")["input_ids"].shape[1]
                
                # collect SAE activations
                activations = []
                def hook_fn(module, input, output):
                    acts = output[0] if isinstance(output, tuple) else output
                    encoded = self.sae.encode(acts.float())
                    activations.append(encoded.detach().cpu())
                
                handle = model.model.layers[self.hook_layer].register_forward_hook(hook_fn)
                _ = model(**qa_inputs)
                handle.remove()
                
                # extract answer token activations and tokens
                attention_mask = qa_inputs["attention_mask"][0]
                real_indices = torch.where(attention_mask == 1)[0]
                answer_indices = real_indices[q_len:]
                
                if len(answer_indices) > 0:
                    answer_acts = activations[0][0, answer_indices.cpu(), feature_idx]
                    answer_token_ids = qa_inputs["input_ids"][0, answer_indices.cpu()]
                    answer_tokens = self.tokenizer.convert_ids_to_tokens(answer_token_ids)
                    
                    for pos, (token, act_val, token_id) in enumerate(zip(answer_tokens, answer_acts, answer_token_ids)):
                        activations_data.append({
                            'token': token,
                            'activation': act_val.item(),
                            'context': prompt,
                            'position': pos,
                            'context_tokens': answer_tokens,
                            'token_ids': answer_token_ids
                        })
        
        activations_data.sort(key=lambda x: x['activation'], reverse=True)
        return activations_data[:k]
    
    def interpret_feature(self, 
                         feature_idx: int,
                         top_activations: List[Dict]) -> str:
        """Use GPT to interpret what a feature represents"""
        
        # format examples for the prompt
        examples_text = ""
        for i, ex in enumerate(top_activations[:10], 1):
            activation = ex['activation']
            pos = ex['position']
            
            # decode context window
            start = max(0, pos - 5)
            end = min(len(ex['token_ids']), pos + 6)
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
                          dataset: List[str],
                          steer_vector: Optional[torch.Tensor] = None,
                          steer_layer: Optional[int] = None) -> Dict[int, str]:
        """Interpret all features in the list"""
        
        interpretations = {}
        examples_seens = {}
        
        for i, feat_idx in enumerate(feature_indices):
            feat_idx = feat_idx.item()
            print(f"Interpreting feature {i+1}/{len(feature_indices)}: idx={feat_idx}")
            
            top_acts = self.get_top_activations(
                feat_idx, 
                dataset,
                steer_vector=steer_vector,
                steer_layer=steer_layer
            )
            interpretation, examples_seen = self.interpret_feature(feat_idx, top_acts)
            
            interpretations[feat_idx] = interpretation
            examples_seens[feat_idx] = examples_seen
            print(examples_seen)
            print(interpretation)
            print("-" * 80 + "\n")
            
        return interpretations, examples_seens