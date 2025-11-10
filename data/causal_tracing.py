"""
causal_tracing.py
Implementação de Causal Tracing baseado em ROME (Meng et al., 2022)
Identifica layers causalmente responsáveis por outputs específicos
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CausalTraceResult:
    """Container para resultados de causal tracing"""
    layer_effects: Dict[int, float]
    critical_layers: List[int]
    baseline_prob: float
    corrupted_prob: float
    trace_matrix: np.ndarray  # [layers × positions]

class CausalTracer:
    """
    Implementa causal tracing para identificar layers críticas
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # Storage para ativações
        self.activations = {}
        self.hooks = []
        
    def _register_hooks(self, layer_range: Optional[List[int]] = None):
        """Registra hooks para capturar ativações"""
        if layer_range is None:
            layer_range = range(len(self.model.transformer.h))
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Armazena hidden states
                if isinstance(output, tuple):
                    self.activations[f'layer_{layer_idx}'] = output[0].detach()
                else:
                    self.activations[f'layer_{layer_idx}'] = output.detach()
            return hook
        
        # Registrar hooks em todas as layers
        for idx in layer_range:
            hook = self.model.transformer.h[idx].register_forward_hook(
                make_hook(idx)
            )
            self.hooks.append(hook)
    
    def _clear_hooks(self):
        """Remove todos os hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def _corrupt_subject(self, 
                         tokens: torch.Tensor, 
                         subject_range: Tuple[int, int],
                         noise_level: float = 3.0) -> torch.Tensor:
        """
        Corrompe tokens do subject com ruído Gaussiano nos embeddings
        
        Args:
            tokens: tensor de tokens [1, seq_len]
            subject_range: (start, end) índices do subject
            noise_level: desvio padrão do ruído
        """
        with torch.no_grad():
            embeds = self.model.transformer.wte(tokens)
            
            # Adicionar ruído aos tokens do subject
            start, end = subject_range
            noise = torch.randn_like(embeds[:, start:end]) * noise_level
            embeds[:, start:end] += noise
            
            return embeds
    
    def trace(self,
              prompt: str,
              subject: str,
              target_token: str,
              noise_level: float = 3.0,
              restoration_type: str = "clean") -> CausalTraceResult:
        """
        Executa causal tracing completo
        
        Args:
            prompt: prompt completo (ex: "The Eiffel Tower is in")
            subject: subject a ser traced (ex: "Eiffel Tower")
            target_token: token esperado (ex: "Paris")
            noise_level: intensidade da corrupção
            restoration_type: "clean" ou "mean"
            
        Returns:
            CausalTraceResult com efeitos por layer
        """
        
        # Tokenizar
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        
        # Identificar range do subject
        subject_tokens = self.tokenizer.encode(subject, add_special_tokens=False)
        subject_str = self.tokenizer.decode(subject_tokens)
        
        # Encontrar posição do subject no prompt
        prompt_tokens = self.tokenizer.tokenize(prompt)
        subject_token_list = self.tokenizer.tokenize(subject)
        
        # Buscar subject_range (simplified - assume contiguous)
        for i in range(len(prompt_tokens) - len(subject_token_list) + 1):
            if prompt_tokens[i:i+len(subject_token_list)] == subject_token_list:
                subject_range = (i, i + len(subject_token_list))
                break
        else:
            raise ValueError(f"Subject '{subject}' não encontrado em prompt")
        
        # === PASSO 1: Clean run (baseline) ===
        self._register_hooks()
        with torch.no_grad():
            outputs_clean = self.model(tokens)
            baseline_prob = F.softmax(outputs_clean.logits[0, -1], dim=-1)[target_id].item()
            clean_acts = {k: v.clone() for k, v in self.activations.items()}
        self._clear_hooks()
        
        # === PASSO 2: Corrupted run ===
        self._register_hooks()
        with torch.no_grad():
            embeds_corrupt = self._corrupt_subject(tokens, subject_range, noise_level)
            outputs_corrupt = self.model(inputs_embeds=embeds_corrupt)
            corrupted_prob = F.softmax(outputs_corrupt.logits[0, -1], dim=-1)[target_id].item()
        self._clear_hooks()
        
        # === PASSO 3: Restoration per layer ===
        layer_effects = {}
        n_layers = len(self.model.transformer.h)
        
        for restore_layer in range(n_layers):
            # Registrar hook que restaura apenas esta layer
            def make_restore_hook(target_layer, clean_activation):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        # Substituir hidden state por clean
                        output = (clean_activation.clone(),) + output[1:]
                        return output
                    else:
                        return clean_activation.clone()
                return hook
            
            # Aplicar hook de restauração
            restore_hook = self.model.transformer.h[restore_layer].register_forward_hook(
                make_restore_hook(restore_layer, clean_acts[f'layer_{restore_layer}'])
            )
            
            # Forward com corrupção + restauração desta layer
            with torch.no_grad():
                embeds_corrupt = self._corrupt_subject(tokens, subject_range, noise_level)
                outputs_restored = self.model(inputs_embeds=embeds_corrupt)
                restored_prob = F.softmax(outputs_restored.logits[0, -1], dim=-1)[target_id].item()
            
            # Causal effect = quanto a restauração recuperou
            causal_effect = restored_prob - corrupted_prob
            layer_effects[restore_layer] = causal_effect
            
            restore_hook.remove()
        
        # Identificar layers críticas (top 20% effect)
        sorted_layers = sorted(layer_effects.items(), key=lambda x: x[1], reverse=True)
        threshold = sorted_layers[int(len(sorted_layers) * 0.2)][1]
        critical_layers = [l for l, eff in sorted_layers if eff >= threshold]
        
        # Criar matriz de trace (simplificada - apenas layers)
        trace_matrix = np.array([[layer_effects[i]] for i in range(n_layers)])
        
        return CausalTraceResult(
            layer_effects=layer_effects,
            critical_layers=critical_layers,
            baseline_prob=baseline_prob,
            corrupted_prob=corrupted_prob,
            trace_matrix=trace_matrix
        )
    
    def visualize_trace(self, result: CausalTraceResult, save_path: Optional[str] = None):
        """Visualiza resultados do causal tracing"""
        layers = list(result.layer_effects.keys())
        effects = list(result.layer_effects.values())
        
        plt.figure(figsize=(12, 6))
        
        # Bar plot dos efeitos causais
        colors = ['red' if l in result.critical_layers else 'blue' for l in layers]
        plt.bar(layers, effects, color=colors, alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Causal Effect (Δ Probability)', fontsize=12)
        plt.title(f'Causal Tracing Results\nBaseline: {result.baseline_prob:.3f} | Corrupted: {result.corrupted_prob:.3f}', 
                  fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        # Destacar layers críticas
        for layer in result.critical_layers:
            plt.axvspan(layer-0.5, layer+0.5, alpha=0.2, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"CAUSAL TRACING SUMMARY")
        print(f"{'='*60}")
        print(f"Baseline probability: {result.baseline_prob:.4f}")
        print(f"Corrupted probability: {result.corrupted_prob:.4f}")
        print(f"Corruption impact: {result.baseline_prob - result.corrupted_prob:.4f}")
        print(f"\nCritical layers (top 20%): {result.critical_layers}")
        print(f"\nTop 5 layers by causal effect:")
        sorted_effects = sorted(result.layer_effects.items(), key=lambda x: x[1], reverse=True)
        for layer, effect in sorted_effects[:5]:
            print(f"  Layer {layer:2d}: {effect:+.4f}")
        print(f"{'='*60}\n")


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    # Inicializar tracer
    tracer = CausalTracer(model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Exemplo 1: Factual knowledge
    result = tracer.trace(
        prompt="The Eiffel Tower is located in",
        subject="Eiffel Tower",
        target_token=" Paris",
        noise_level=3.0
    )
    
    tracer.visualize_trace(result, save_path="causal_trace_eiffel.png")
    
    # Exemplo 2: Outro fact
    result2 = tracer.trace(
        prompt="Apple Inc. was founded by",
        subject="Apple Inc.",
        target_token=" Steve",
        noise_level=3.0
    )
    
    tracer.visualize_trace(result2, save_path="causal_trace_apple.png")
