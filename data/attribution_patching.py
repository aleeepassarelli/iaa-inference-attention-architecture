"""
attribution_patching.py
Implementação simplificada via gradientes (2 forward passes apenas)
Método mais eficiente que activation patching tradicional
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class AttributionResult:
    """Resultado de attribution patching"""
    component_attributions: Dict[str, float]
    total_effect: float
    top_components: list

class AttributionPatcher:
    """
    Attribution patching via gradientes (método de Nanda et al.)
    Requer apenas 2 forward passes
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.n_layers = len(self.model.transformer.h)
        self.activations = {}
        
    def _register_hooks_for_gradients(self):
        """Registra hooks para capturar ativações e gradientes"""
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                
                # Requer grad para backward
                act = act.requires_grad_(True)
                act.retain_grad()
                self.activations[name] = act
                
                # Return modified output
                if isinstance(output, tuple):
                    return (act,) + output[1:]
                return act
            return hook
        
        # Registrar em todas as layers
        for idx in range(self.n_layers):
            # Attention
            hook = self.model.transformer.h[idx].attn.register_forward_hook(
                make_hook(f'L{idx}_attn')
            )
            hooks.append(hook)
            
            # MLP
            hook = self.model.transformer.h[idx].mlp.register_forward_hook(
                make_hook(f'L{idx}_mlp')
            )
            hooks.append(hook)
        
        return hooks
    
    def attribute(self,
                  clean_input: torch.Tensor,
                  corrupt_input: torch.Tensor,
                  target_id: int,
                  distractor_id: Optional[int] = None) -> AttributionResult:
        """
        Executa attribution patching
        
        Apenas 2 forward passes:
        1. Clean (com gradientes)
        2. Corrupt (sem gradientes, apenas ativações)
        
        Attribution = gradient × (corrupt_act - clean_act)
        """
        
        # === PASS 1: Clean com gradientes ===
        self.model.zero_grad()
        self.activations = {}
        
        hooks = self._register_hooks_for_gradients()
        
        # Forward
        outputs_clean = self.model(clean_input)
        
        # Loss = negative log prob of target
        logits = outputs_clean.logits[0, -1]
        
        if distractor_id is not None:
            # Logit difference
            loss = -(logits[target_id] - logits[distractor_id])
        else:
            loss = -F.log_softmax(logits, dim=-1)[target_id]
        
        # Backward para obter gradientes
        loss.backward()
        
        # Armazenar gradientes
        clean_acts = {}
        clean_grads = {}
        
        for name, act in self.activations.items():
            clean_acts[name] = act.detach().clone()
            if act.grad is not None:
                clean_grads[name] = act.grad.detach().clone()
        
        # Limpar hooks
        for hook in hooks:
            hook.remove()
        
        # === PASS 2: Corrupt (sem gradientes) ===
        self.activations = {}
        hooks = self._register_hooks_for_gradients()
        
        with torch.no_grad():
            outputs_corrupt = self.model(corrupt_input)
        
        corrupt_acts = {}
        for name, act in self.activations.items():
            corrupt_acts[name] = act.detach().clone()
        
        for hook in hooks:
            hook.remove()
        
        # === Compute attributions ===
        attributions = {}
        
        for name in clean_acts.keys():
            if name in clean_grads:
                # Attribution = gradient · (corrupt - clean)
                delta_act = corrupt_acts[name] - clean_acts[name]
                grad = clean_grads[name]
                
                # Dot product summed over all dimensions except batch
                attribution = (grad * delta_act).sum().item()
                attributions[name] = attribution
        
        # Total effect = sum of attributions
        total_effect = sum(attributions.values())
        
        # Top components
        sorted_comps = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_components = sorted_comps[:20]
        
        return AttributionResult(
            component_attributions=attributions,
            total_effect=total_effect,
            top_components=top_components
        )
    
    def visualize_attributions(self, result: AttributionResult, 
                              save_path: Optional[str] = None):
        """Visualiza attributions"""
        
        # Separar por layer e tipo
        attn_attrs = []
        mlp_attrs = []
        layer_labels = []
        
        for layer in range(self.n_layers):
            attn_key = f'L{layer}_attn'
            mlp_key = f'L{layer}_mlp'
            
            attn_val = result.component_attributions.get(attn_key, 0)
            mlp_val = result.component_attributions.get(mlp_key, 0)
            
            attn_attrs.append(attn_val)
            mlp_attrs.append(mlp_val)
            layer_labels.append(f'L{layer}')
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Stacked bar por layer
        x = np.arange(len(layer_labels))
        width = 0.35
        
        ax1.bar(x - width/2, attn_attrs, width, label='Attention', alpha=0.8)
        ax1.bar(x + width/2, mlp_attrs, width, label='MLP', alpha=0.8)
        
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Attribution', fontsize=12)
        ax1.set_title('Attribution by Layer and Component Type', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_labels)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 2: Top components
        top_names = [name for name, _ in result.top_components[:15]]
        top_values = [val for _, val in result.top_components[:15]]
        
        colors = ['blue' if 'attn' in name else 'red' for name in top_names]
        ax2.barh(range(len(top_names)), top_values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_names)))
        ax2.set_yticklabels(top_names, fontsize=10)
        ax2.set_xlabel('Attribution', fontsize=12)
        ax2.set_title('Top-15 Components by Attribution', fontsize=14)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"ATTRIBUTION PATCHING SUMMARY")
        print(f"{'='*70}")
        print(f"Total effect: {result.total_effect:.6f}")
        print(f"\nTop-10 components:")
        for i, (name, val) in enumerate(result.top_components[:10], 1):
            print(f"  {i:2d}. {name:15s}: {val:+.6f}")
        print(f"{'='*70}\n")


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    attributor = AttributionPatcher(model_name="gpt2", 
                                    device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Inputs
    clean_prompt = "The Eiffel Tower is located in Paris"
    corrupt_prompt = "The Eiffel Tower is located in London"
    
    clean_tokens = attributor.tokenizer(clean_prompt, return_tensors="pt").input_ids.to(attributor.device)
    corrupt_tokens = attributor.tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(attributor.device)
    
    target_id = attributor.tokenizer.encode(" Paris", add_special_tokens=False)[0]
    distractor_id = attributor.tokenizer.encode(" London", add_special_tokens=False)[0]
    
    # Atribuir
    result = attributor.attribute(clean_tokens, corrupt_tokens, target_id, distractor_id)
    
    # Visualizar
    attributor.visualize_attributions(result, save_path="attribution_results.png")
"""
attribution_patching.py
Implementação simplificada via gradientes (2 forward passes apenas)
Método mais eficiente que activation patching tradicional
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class AttributionResult:
    """Resultado de attribution patching"""
    component_attributions: Dict[str, float]
    total_effect: float
    top_components: list

class AttributionPatcher:
    """
    Attribution patching via gradientes (método de Nanda et al.)
    Requer apenas 2 forward passes
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.n_layers = len(self.model.transformer.h)
        self.activations = {}
        
    def _register_hooks_for_gradients(self):
        """Registra hooks para capturar ativações e gradientes"""
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                
                # Requer grad para backward
                act = act.requires_grad_(True)
                act.retain_grad()
                self.activations[name] = act
                
                # Return modified output
                if isinstance(output, tuple):
                    return (act,) + output[1:]
                return act
            return hook
        
        # Registrar em todas as layers
        for idx in range(self.n_layers):
            # Attention
            hook = self.model.transformer.h[idx].attn.register_forward_hook(
                make_hook(f'L{idx}_attn')
            )
            hooks.append(hook)
            
            # MLP
            hook = self.model.transformer.h[idx].mlp.register_forward_hook(
                make_hook(f'L{idx}_mlp')
            )
            hooks.append(hook)
        
        return hooks
    
    def attribute(self,
                  clean_input: torch.Tensor,
                  corrupt_input: torch.Tensor,
                  target_id: int,
                  distractor_id: Optional[int] = None) -> AttributionResult:
        """
        Executa attribution patching
        
        Apenas 2 forward passes:
        1. Clean (com gradientes)
        2. Corrupt (sem gradientes, apenas ativações)
        
        Attribution = gradient × (corrupt_act - clean_act)
        """
        
        # === PASS 1: Clean com gradientes ===
        self.model.zero_grad()
        self.activations = {}
        
        hooks = self._register_hooks_for_gradients()
        
        # Forward
        outputs_clean = self.model(clean_input)
        
        # Loss = negative log prob of target
        logits = outputs_clean.logits[0, -1]
        
        if distractor_id is not None:
            # Logit difference
            loss = -(logits[target_id] - logits[distractor_id])
        else:
            loss = -F.log_softmax(logits, dim=-1)[target_id]
        
        # Backward para obter gradientes
        loss.backward()
        
        # Armazenar gradientes
        clean_acts = {}
        clean_grads = {}
        
        for name, act in self.activations.items():
            clean_acts[name] = act.detach().clone()
            if act.grad is not None:
                clean_grads[name] = act.grad.detach().clone()
        
        # Limpar hooks
        for hook in hooks:
            hook.remove()
        
        # === PASS 2: Corrupt (sem gradientes) ===
        self.activations = {}
        hooks = self._register_hooks_for_gradients()
        
        with torch.no_grad():
            outputs_corrupt = self.model(corrupt_input)
        
        corrupt_acts = {}
        for name, act in self.activations.items():
            corrupt_acts[name] = act.detach().clone()
        
        for hook in hooks:
            hook.remove()
        
        # === Compute attributions ===
        attributions = {}
        
        for name in clean_acts.keys():
            if name in clean_grads:
                # Attribution = gradient · (corrupt - clean)
                delta_act = corrupt_acts[name] - clean_acts[name]
                grad = clean_grads[name]
                
                # Dot product summed over all dimensions except batch
                attribution = (grad * delta_act).sum().item()
                attributions[name] = attribution
        
        # Total effect = sum of attributions
        total_effect = sum(attributions.values())
        
        # Top components
        sorted_comps = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_components = sorted_comps[:20]
        
        return AttributionResult(
            component_attributions=attributions,
            total_effect=total_effect,
            top_components=top_components
        )
    
    def visualize_attributions(self, result: AttributionResult, 
                              save_path: Optional[str] = None):
        """Visualiza attributions"""
        
        # Separar por layer e tipo
        attn_attrs = []
        mlp_attrs = []
        layer_labels = []
        
        for layer in range(self.n_layers):
            attn_key = f'L{layer}_attn'
            mlp_key = f'L{layer}_mlp'
            
            attn_val = result.component_attributions.get(attn_key, 0)
            mlp_val = result.component_attributions.get(mlp_key, 0)
            
            attn_attrs.append(attn_val)
            mlp_attrs.append(mlp_val)
            layer_labels.append(f'L{layer}')
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Stacked bar por layer
        x = np.arange(len(layer_labels))
        width = 0.35
        
        ax1.bar(x - width/2, attn_attrs, width, label='Attention', alpha=0.8)
        ax1.bar(x + width/2, mlp_attrs, width, label='MLP', alpha=0.8)
        
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Attribution', fontsize=12)
        ax1.set_title('Attribution by Layer and Component Type', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_labels)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 2: Top components
        top_names = [name for name, _ in result.top_components[:15]]
        top_values = [val for _, val in result.top_components[:15]]
        
        colors = ['blue' if 'attn' in name else 'red' for name in top_names]
        ax2.barh(range(len(top_names)), top_values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_names)))
        ax2.set_yticklabels(top_names, fontsize=10)
        ax2.set_xlabel('Attribution', fontsize=12)
        ax2.set_title('Top-15 Components by Attribution', fontsize=14)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"ATTRIBUTION PATCHING SUMMARY")
        print(f"{'='*70}")
        print(f"Total effect: {result.total_effect:.6f}")
        print(f"\nTop-10 components:")
        for i, (name, val) in enumerate(result.top_components[:10], 1):
            print(f"  {i:2d}. {name:15s}: {val:+.6f}")
        print(f"{'='*70}\n")


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    attributor = AttributionPatcher(model_name="gpt2", 
                                    device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Inputs
    clean_prompt = "The Eiffel Tower is located in Paris"
    corrupt_prompt = "The Eiffel Tower is located in London"
    
    clean_tokens = attributor.tokenizer(clean_prompt, return_tensors="pt").input_ids.to(attributor.device)
    corrupt_tokens = attributor.tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(attributor.device)
    
    target_id = attributor.tokenizer.encode(" Paris", add_special_tokens=False)[0]
    distractor_id = attributor.tokenizer.encode(" London", add_special_tokens=False)[0]
    
    # Atribuir
    result = attributor.attribute(clean_tokens, corrupt_tokens, target_id, distractor_id)
    
    # Visualizar
    attributor.visualize_attributions(result, save_path="attribution_results.png")
