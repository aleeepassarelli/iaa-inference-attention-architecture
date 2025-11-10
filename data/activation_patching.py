"""
activation_patching.py
Implementação de Activation Patching para análise de componentes causais
Suporta patching de attention heads, MLPs, e residual stream
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class PatchingResult:
    """Resultado de um experimento de patching"""
    component_name: str
    clean_output: float
    corrupt_output: float
    patched_output: float
    causal_effect: float  # patched - corrupt
    necessity: float      # clean - patched
    
class ActivationPatcher:
    """
    Implementa activation patching para identificar componentes causais
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        self.n_layers = len(self.model.transformer.h)
        self.n_heads = self.model.config.n_head
        self.d_model = self.model.config.n_embd
        self.d_head = self.d_model // self.n_heads
        
    def _get_logit_diff(self, logits: torch.Tensor, target_id: int, 
                       distractor_id: Optional[int] = None) -> float:
        """
        Calcula diferença de logits (target vs distractor ou max de outros)
        """
        probs = F.softmax(logits[0, -1], dim=-1)
        target_prob = probs[target_id].item()
        
        if distractor_id is not None:
            distractor_prob = probs[distractor_id].item()
            return target_prob - distractor_prob
        else:
            return target_prob
    
    def patch_attention_head(self,
                            clean_input: torch.Tensor,
                            corrupt_input: torch.Tensor,
                            layer_idx: int,
                            head_idx: int,
                            target_id: int,
                            distractor_id: Optional[int] = None) -> PatchingResult:
        """
        Patch um attention head específico
        
        Args:
            clean_input: input limpo [1, seq_len]
            corrupt_input: input corrompido [1, seq_len]
            layer_idx: índice da layer
            head_idx: índice do head
            target_id: token alvo
            distractor_id: token distrator (opcional)
        """
        
        # === Clean run ===
        with torch.no_grad():
            clean_cache = {}
            
            # Hook para armazenar attention output do head específico
            def cache_hook(module, input, output):
                # output[0] é o attention output [batch, seq, n_heads, d_head]
                # Extrair apenas o head específico
                attn_output = output[0]  # Tuple[Tensor, ...]
                clean_cache['head_output'] = attn_output.clone()
            
            hook = self.model.transformer.h[layer_idx].attn.register_forward_hook(cache_hook)
            outputs_clean = self.model(clean_input)
            clean_logit = self._get_logit_diff(outputs_clean.logits, target_id, distractor_id)
            hook.remove()
        
        # === Corrupt run ===
        with torch.no_grad():
            outputs_corrupt = self.model(corrupt_input)
            corrupt_logit = self._get_logit_diff(outputs_corrupt.logits, target_id, distractor_id)
        
        # === Patched run (corruption + patch este head com clean) ===
        def patch_hook(module, input, output):
            # Substituir output do head específico com versão clean
            attn_output = output[0]
            
            # attn_output shape: [batch, seq, n_heads, d_head]
            # Patch apenas head_idx
            attn_output[:, :, head_idx, :] = clean_cache['head_output'][:, :, head_idx, :]
            
            return (attn_output,) + output[1:]
        
        with torch.no_grad():
            hook = self.model.transformer.h[layer_idx].attn.register_forward_hook(patch_hook)
            outputs_patched = self.model(corrupt_input)
            patched_logit = self._get_logit_diff(outputs_patched.logits, target_id, distractor_id)
            hook.remove()
        
        causal_effect = patched_logit - corrupt_logit
        necessity = clean_logit - patched_logit
        
        return PatchingResult(
            component_name=f"L{layer_idx}H{head_idx}",
            clean_output=clean_logit,
            corrupt_output=corrupt_logit,
            patched_output=patched_logit,
            causal_effect=causal_effect,
            necessity=necessity
        )
    
    def patch_mlp(self,
                  clean_input: torch.Tensor,
                  corrupt_input: torch.Tensor,
                  layer_idx: int,
                  target_id: int,
                  distractor_id: Optional[int] = None) -> PatchingResult:
        """Patch MLP de uma layer específica"""
        
        # Clean run
        with torch.no_grad():
            clean_cache = {}
            
            def cache_hook(module, input, output):
                clean_cache['mlp_output'] = output.clone()
            
            hook = self.model.transformer.h[layer_idx].mlp.register_forward_hook(cache_hook)
            outputs_clean = self.model(clean_input)
            clean_logit = self._get_logit_diff(outputs_clean.logits, target_id, distractor_id)
            hook.remove()
        
        # Corrupt run
        with torch.no_grad():
            outputs_corrupt = self.model(corrupt_input)
            corrupt_logit = self._get_logit_diff(outputs_corrupt.logits, target_id, distractor_id)
        
        # Patched run
        def patch_hook(module, input, output):
            return clean_cache['mlp_output']
        
        with torch.no_grad():
            hook = self.model.transformer.h[layer_idx].mlp.register_forward_hook(patch_hook)
            outputs_patched = self.model(corrupt_input)
            patched_logit = self._get_logit_diff(outputs_patched.logits, target_id, distractor_id)
            hook.remove()
        
        causal_effect = patched_logit - corrupt_logit
        necessity = clean_logit - patched_logit
        
        return PatchingResult(
            component_name=f"L{layer_idx}_MLP",
            clean_output=clean_logit,
            corrupt_output=corrupt_logit,
            patched_output=patched_logit,
            causal_effect=causal_effect,
            necessity=necessity
        )
    
    def comprehensive_patch_scan(self,
                                 clean_input: torch.Tensor,
                                 corrupt_input: torch.Tensor,
                                 target_id: int,
                                 distractor_id: Optional[int] = None,
                                 include_mlps: bool = True) -> List[PatchingResult]:
        """
        Scan completo: patch todos attention heads e MLPs
        
        Returns:
            Lista de PatchingResult ordenada por causal_effect
        """
        results = []
        
        # Patch todos os attention heads
        print("Patching attention heads...")
        for layer in tqdm(range(self.n_layers)):
            for head in range(self.n_heads):
                result = self.patch_attention_head(
                    clean_input, corrupt_input, layer, head, 
                    target_id, distractor_id
                )
                results.append(result)
        
        # Patch todos os MLPs
        if include_mlps:
            print("Patching MLPs...")
            for layer in tqdm(range(self.n_layers)):
                result = self.patch_mlp(
                    clean_input, corrupt_input, layer,
                    target_id, distractor_id
                )
                results.append(result)
        
        # Ordenar por causal effect (descendente)
        results.sort(key=lambda x: x.causal_effect, reverse=True)
        
        return results
    
    def visualize_patching_results(self, results: List[PatchingResult], 
                                   top_k: int = 20, save_path: Optional[str] = None):
        """Visualiza resultados de patching"""
        import matplotlib.pyplot as plt
        
        # Top-k componentes
        top_results = results[:top_k]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Causal Effect
        names = [r.component_name for r in top_results]
        effects = [r.causal_effect for r in top_results]
        
        colors = ['red' if 'MLP' in name else 'blue' for name in names]
        ax1.barh(range(len(names)), effects, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel('Causal Effect', fontsize=12)
        ax1.set_title(f'Top-{top_k} Components by Causal Effect', fontsize=14)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # Plot 2: Necessity vs Causal Effect
        causal_effs = [r.causal_effect for r in results]
        necessities = [r.necessity for r in results]
        component_types = ['MLP' if 'MLP' in r.component_name else 'Head' 
                          for r in results]
        
        for comp_type, color in [('Head', 'blue'), ('MLP', 'red')]:
            mask = [t == comp_type for t in component_types]
            ax2.scatter(
                [e for e, m in zip(causal_effs, mask) if m],
                [n for n, m in zip(necessities, mask) if m],
                alpha=0.5, label=comp_type, color=color, s=20
            )
        
        ax2.set_xlabel('Causal Effect', fontsize=12)
        ax2.set_ylabel('Necessity', fontsize=12)
        ax2.set_title('Necessity vs Causal Effect (all components)', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Quadrantes
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"ACTIVATION PATCHING SUMMARY")
        print(f"{'='*70}")
        print(f"Total components analyzed: {len(results)}")
        print(f"\nTop-10 by Causal Effect:")
        for i, r in enumerate(results[:10], 1):
            print(f"  {i:2d}. {r.component_name:12s} | Effect: {r.causal_effect:+.4f} | Necessity: {r.necessity:+.4f}")
        print(f"{'='*70}\n")


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    patcher = ActivationPatcher(model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Preparar inputs
    clean_prompt = "The Eiffel Tower is located in Paris"
    corrupt_prompt = "The Eiffel Tower is located in London"  # Counterfactual
    
    clean_tokens = patcher.tokenizer(clean_prompt, return_tensors="pt").input_ids.to(patcher.device)
    corrupt_tokens = patcher.tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(patcher.device)
    
    target_token = " Paris"
    distractor_token = " London"
    
    target_id = patcher.tokenizer.encode(target_token, add_special_tokens=False)[0]
    distractor_id = patcher.tokenizer.encode(distractor_token, add_special_tokens=False)[0]
    
    # Scan completo
    results = patcher.comprehensive_patch_scan(
        clean_tokens, corrupt_tokens,
        target_id, distractor_id,
        include_mlps=True
    )
    
    # Visualizar
    patcher.visualize_patching_results(results, top_k=20, save_path="patching_results.png")
