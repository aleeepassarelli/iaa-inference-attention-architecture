#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_density_llm.py
-----------------------
Calcula densidade semântica usando ativações de LLMs (GPT, LLaMA, etc)
Integra com os scripts de causal patching anteriores.

Versão: 2.0 (2025-11-09)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple
import argparse

class LLMSemanticDensity:
    """
    Calcula densidade semântica usando representações internas de LLMs.
    Extrai ativações de layers específicas para análise mais profunda.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        self.n_layers = len(self.model.transformer.h)
        print(f"✓ Loaded {model_name} ({self.n_layers} layers) on {self.device}")
    
    def extract_layer_activations(self, 
                                  text: str, 
                                  layer_idx: int = -1,
                                  pooling: str = 'mean') -> np.ndarray:
        """
        Extrai ativações de uma layer específica.
        
        Args:
            text: texto de entrada
            layer_idx: índice da layer (-1 = última)
            pooling: 'mean', 'max', 'cls', ou 'last'
        """
        tokens = self.tokenizer(text, return_tensors="pt", 
                               padding=True, truncation=True, 
                               max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]  # [batch, seq_len, dim]
        
        # Pooling
        if pooling == 'mean':
            activation = hidden_states.mean(dim=1)
        elif pooling == 'max':
            activation = hidden_states.max(dim=1)[0]
        elif pooling == 'cls':
            activation = hidden_states[:, 0, :]
        elif pooling == 'last':
            activation = hidden_states[:, -1, :]
        
        return activation.cpu().numpy()
    
    def compute_multilayer_density(self,
                                   text: str,
                                   layers: Optional[List[int]] = None,
                                   pooling: str = 'mean') -> Dict:
        """
        Calcula densidade semântica em múltiplas layers.
        Permite ver evolução da densidade através do modelo.
        """
        if layers is None:
            # Layers estratégicas: early, middle, late
            layers = [0, self.n_layers // 4, self.n_layers // 2, 
                     3 * self.n_layers // 4, self.n_layers - 1]
        
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 2]
        
        if len(sentences) < 2:
            raise ValueError("Texto deve ter ao menos 2 sentenças.")
        
        layer_results = {}
        
        for layer_idx in layers:
            # Extrair embeddings para cada sentença
            embeddings = []
            for sent in sentences:
                emb = self.extract_layer_activations(sent, layer_idx, pooling)
                embeddings.append(emb[0])  # Remove batch dim
            
            embeddings = np.array(embeddings)
            
            # Normalizar
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            
            # Calcular métricas
            metrics = self._compute_metrics_from_embeddings(embeddings_norm)
            
            layer_results[f'layer_{layer_idx}'] = metrics
        
        # Agregar cross-layer
        all_densities = [res['semantic_density'] for res in layer_results.values()]
        
        aggregate = {
            'layer_results': layer_results,
            'mean_density_across_layers': float(np.mean(all_densities)),
            'std_density_across_layers': float(np.std(all_densities)),
            'best_layer': f"layer_{layers[np.argmax(all_densities)]}",
            'worst_layer': f"layer_{layers[np.argmin(all_densities)]}",
            'n_layers_analyzed': len(layers)
        }
        
        return aggregate
    
    def _compute_metrics_from_embeddings(self, embeddings_norm: np.ndarray) -> Dict:
        """Calcula métricas a partir de embeddings normalizados"""
        
        # Similarity matrix
        sim_matrix = cosine_similarity(embeddings_norm)
        
        # Dispersão
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings_norm, metric='cosine')
        mean_dispersion = np.mean(distances)
        
        # Entropia
        probs = np.abs(sim_matrix) / (np.sum(np.abs(sim_matrix), axis=1, keepdims=True) + 1e-8)
        entropy_vals = [entropy(p + 1e-10) for p in probs]
        mean_entropy = np.mean(entropy_vals)
        
        # Coerência
        coherence_scores = []
        for i in range(len(embeddings_norm) - 1):
            coherence_scores.append(
                cosine_similarity([embeddings_norm[i]], [embeddings_norm[i+1]])[0][0]
            )
        mean_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        # Isotropy
        centroid = embeddings_norm.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        angles = np.arccos(np.clip(embeddings_norm @ centroid, -1, 1))
        isotropy_score = np.clip(1.0 - (np.std(angles) / 10), 0, 1)
        
        # SD composite
        dispersion_term = 1.0 - mean_dispersion
        entropy_term = 1.0 - (mean_entropy / np.log(len(embeddings_norm)))
        coherence_term = mean_coherence
        isotropy_term = isotropy_score
        
        sd_composite = (
            0.3 * dispersion_term +
            0.3 * entropy_term +
            0.25 * coherence_term +
            0.15 * isotropy_term
        )
        sd_composite = np.clip(sd_composite, 0, 1)
        
        return {
            'semantic_density': float(sd_composite),
            'mean_entropy': float(mean_entropy),
            'mean_dispersion': float(mean_dispersion),
            'coherence': float(mean_coherence),
            'isotropy': float(isotropy_score)
        }
    
    def compute_token_level_density(self, text: str, layer_idx: int = -1) -> Dict:
        """
        Calcula densidade semântica token-by-token (alta granularidade).
        Útil para identificar tokens que aumentam/diminuem densidade.
        """
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        token_strs = [self.tokenizer.decode([t]) for t in tokens.input_ids[0]]
        
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]  # [1, seq_len, dim]
        
        # Embeddings por token
        embeddings = hidden_states[0].cpu().numpy()  # [seq_len, dim]
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Densidade local: para cada token, calcular similaridade com vizinhos
        local_densities = []
        
        for i in range(len(embeddings_norm)):
            if i == 0 or i == len(embeddings_norm) - 1:
                local_densities.append(0.0)  # Borders
                continue
            
            # Janela: i-1, i, i+1
            window = embeddings_norm[max(0, i-1):min(len(embeddings_norm), i+2)]
            sim_matrix = cosine_similarity(window)
            
            # Densidade = média de similaridades
            local_dens = sim_matrix[1].mean()  # Token central (índice 1 na janela)
            local_densities.append(float(local_dens))
        
        # Identificar tokens de alta/baixa densidade
        densities_array = np.array(local_densities)
        top_k = min(5, len(token_strs))
        
        top_indices = np.argsort(densities_array)[-top_k:][::-1]
        low_indices = np.argsort(densities_array)[:top_k]
        
        return {
            'token_densities': local_densities,
            'tokens': token_strs,
            'mean_token_density': float(np.mean(densities_array)),
            'std_token_density': float(np.std(densities_array)),
            'high_density_tokens': [
                {'token': token_strs[i], 'density': local_densities[i], 'position': int(i)}
                for i in top_indices
            ],
            'low_density_tokens': [
                {'token': token_strs[i], 'density': local_densities[i], 'position': int(i)}
                for i in low_indices
            ]
        }


def main():
    parser = argparse.ArgumentParser(description="Semantic Density via LLM activations")
    parser.add_argument("input", type=str, help="Text or file path")
    parser.add_argument("--model", type=str, default="gpt2", help="LLM model name")
    parser.add_argument("--multilayer", action="store_true", 
                       help="Analyze density across multiple layers")
    parser.add_argument("--token-level", action="store_true",
                       help="Compute token-level density")
    parser.add_argument("--layer", type=int, default=-1,
                       help="Specific layer to analyze (default: -1 = last)")
    parser.add_argument("--pooling", type=str, default="mean",
                       choices=['mean', 'max', 'cls', 'last'],
                       help="Pooling strategy")
    
    args = parser.parse_args()
    
    # Load text
    try:
        if args.input.endswith(('.txt', '.md')):
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.input
    except Exception as e:
        print(f"❌ Error loading input: {e}")
        return 1
    
    # Initialize
    analyzer = LLMSemanticDensity(model_name=args.model)
    
    try:
        if args.multilayer:
            print(f"\n🔬 Analyzing density across layers...")
            result = analyzer.compute_multilayer_density(text, pooling=args.pooling)
            
            print(f"\n{'='*60}")
            print(f"MULTI-LAYER SEMANTIC DENSITY")
            print(f"{'='*60}")
            print(f"Layers analyzed: {result['n_layers_analyzed']}")
            print(f"Mean density:    {result['mean_density_across_layers']:.4f}")
            print(f"Std density:     {result['std_density_across_layers']:.4f}")
            print(f"Best layer:      {result['best_layer']}")
            print(f"Worst layer:     {result['worst_layer']}")
            
            print(f"\n--- Per-Layer Results ---")
            for layer_name, metrics in result['layer_results'].items():
                print(f"\n{layer_name}:")
                print(f"  SD:         {metrics['semantic_density']:.4f}")
                print(f"  Coherence:  {metrics['coherence']:.4f}")
                print(f"  Isotropy:   {metrics['isotropy']:.4f}")
        
        elif args.token_level:
            print(f"\n🔬 Computing token-level density...")
            result = analyzer.compute_token_level_density(text, layer_idx=args.layer)
            
            print(f"\n{'='*60}")
            print(f"TOKEN-LEVEL SEMANTIC DENSITY")
            print(f"{'='*60}")
            print(f"Tokens analyzed:     {len(result['tokens'])}")
            print(f"Mean token density:  {result['mean_token_density']:.4f}")
            print(f"Std token density:   {result['std_token_density']:.4f}")
            
            print(f"\n--- High Density Tokens ---")
            for item in result['high_density_tokens']:
                print(f"  '{item['token']:15s}' @ pos {item['position']:3d}: {item['density']:.4f}")
            
            print(f"\n--- Low Density Tokens ---")
            for item in result['low_density_tokens']:
                print(f"  '{item['token']:15s}' @ pos {item['position']:3d}: {item['density']:.4f}")
        
        else:
            # Single layer analysis
            print(f"\n🔬 Analyzing layer {args.layer}...")
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 2]
            embeddings = []
            
            for sent in sentences:
                emb = analyzer.extract_layer_activations(sent, args.layer, args.pooling)
                embeddings.append(emb[0])
            
            embeddings = np.array(embeddings)
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            
            result = analyzer._compute_metrics_from_embeddings(embeddings_norm)
            
            print(f"\n{'='*60}")
            print(f"SEMANTIC DENSITY (Layer {args.layer})")
            print(f"{'='*60}")
            print(f"SD Score:       {result['semantic_density']:.4f}")
            print(f"Coherence:      {result['coherence']:.4f}")
            print(f"Isotropy:       {result['isotropy']:.4f}")
            print(f"Mean Entropy:   {result['mean_entropy']:.4f}")
            print(f"Mean Dispersion:{result['mean_dispersion']:.4f}")
            print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
