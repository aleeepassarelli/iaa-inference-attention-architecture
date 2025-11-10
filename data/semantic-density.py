#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_density.py (Enhanced)
-------------------------------
Calcula densidade semântica (SD) com métricas adicionais do framework EAT.

Versão: 2.0 (2025-11-09)
Referência: EAT-Lab Investigation Reports #1-30
"""

import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 🔹 Métricas Core
# ============================================================

def compute_semantic_density(text: str, 
                            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                            verbose: bool = False) -> Dict:
    """
    Calcula densidade semântica (SD) expandida com métricas EAT.
    
    Returns:
        Dict com:
        - semantic_density (SD): densidade semântica normalizada
        - mean_entropy (S_H): entropia heurística média
        - mean_dispersion (ρ): dispersão média no espaço latente
        - coherence (μ): coerência inter-sentencial
        - isotropy: medida de uniformidade espacial
        - intrinsic_dim: dimensionalidade intrínseca
    """
    model = SentenceTransformer(model_name)
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 2]
    
    if len(sentences) < 2:
        raise ValueError("Texto muito curto. Forneça ao menos duas sentenças.")
    
    # Embeddings
    embeddings = model.encode(sentences, convert_to_tensor=False)
    embeddings = np.array(embeddings)
    
    # Normalizar para hiperesfera (como transformers fazem com LayerNorm)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # === MÉTRICAS CORE ===
    
    # 1. Similarity matrix
    sim_matrix = cosine_similarity(embeddings_norm)
    
    # 2. Dispersão média (distância cosine)
    distances = pdist(embeddings_norm, metric='cosine')
    mean_dispersion = np.mean(distances)
    std_dispersion = np.std(distances)
    
    # 3. Entropia Heurística (S_H)
    # Normalizar sim_matrix para probabilidades
    probs = np.abs(sim_matrix) / (np.sum(np.abs(sim_matrix), axis=1, keepdims=True) + 1e-8)
    entropy_vals = [entropy(p + 1e-10) for p in probs]
    mean_entropy = np.mean(entropy_vals)
    
    # 4. Coerência (μ) - similaridade entre sentenças adjacentes
    coherence_scores = []
    for i in range(len(embeddings_norm) - 1):
        coherence_scores.append(
            cosine_similarity([embeddings_norm[i]], [embeddings_norm[i+1]])[0][0]
        )
    mean_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
    
    # 5. Isotropy (uniformidade espacial) - EAT-REx #14
    # Medir concentração em cone: baixo std de ângulos = anisotropic
    centroid = embeddings_norm.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    
    angles = np.arccos(np.clip(embeddings_norm @ centroid, -1, 1))
    isotropy_score = 1.0 / (np.std(angles) + 1e-8)  # Alto std = isotropic
    isotropy_score = np.clip(1.0 - (isotropy_score / 10), 0, 1)  # Normalizar
    
    # 6. Dimensionalidade Intrínseca (via PCA) - EAT-REx #13, #17
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(embeddings_norm)
    
    # Dimensionalidade: quantos PCs explicam 95% da variância
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = np.argmax(cumsum >= 0.95) + 1
    
    # === DENSIDADE SEMÂNTICA COMPOSTA ===
    # Fórmula melhorada: combinar métricas com pesos
    
    # Componente 1: Baixa dispersão (alta coesão)
    dispersion_term = 1.0 - mean_dispersion
    
    # Componente 2: Baixa entropia (alta previsibilidade)
    entropy_term = 1.0 - (mean_entropy / np.log(len(sentences)))  # Normalizar por max entropy
    
    # Componente 3: Alta coerência sequencial
    coherence_term = mean_coherence
    
    # Componente 4: Isotropy (estrutura bem distribuída)
    isotropy_term = isotropy_score
    
    # SD composto (pesos inspirados em Score(P) do EAT)
    weights = {
        'dispersion': 0.3,
        'entropy': 0.3,
        'coherence': 0.25,
        'isotropy': 0.15
    }
    
    sd_composite = (
        weights['dispersion'] * dispersion_term +
        weights['entropy'] * entropy_term +
        weights['coherence'] * coherence_term +
        weights['isotropy'] * isotropy_term
    )
    
    sd_composite = np.clip(sd_composite, 0, 1)
    
    # === SCORE ORIGINAL (retrocompatibilidade) ===
    sd_original = 1 - (mean_entropy / (mean_dispersion + 1e-8))
    sd_original = np.clip(sd_original, 0, 1)
    
    results = {
        "semantic_density": float(sd_composite),
        "sd_original": float(sd_original),
        "mean_entropy": float(mean_entropy),
        "mean_dispersion": float(mean_dispersion),
        "std_dispersion": float(std_dispersion),
        "coherence": float(mean_coherence),
        "isotropy": float(isotropy_score),
        "intrinsic_dim": int(intrinsic_dim),
        "n_sentences": len(sentences),
        "embedding_dim": embeddings.shape[1],
        "components": {
            "dispersion_term": float(dispersion_term),
            "entropy_term": float(entropy_term),
            "coherence_term": float(coherence_term),
            "isotropy_term": float(isotropy_term)
        }
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SEMANTIC DENSITY ANALYSIS (Enhanced)")
        print(f"{'='*60}")
        print(f"Sentences analyzed:     {results['n_sentences']}")
        print(f"Embedding dimension:    {results['embedding_dim']}")
        print(f"Intrinsic dimension:    {results['intrinsic_dim']}")
        print(f"\n--- Core Metrics ---")
        print(f"SD (Composite):         {results['semantic_density']:.4f} ★")
        print(f"SD (Original):          {results['sd_original']:.4f}")
        print(f"Coherence (μ):          {results['coherence']:.4f}")
        print(f"Isotropy:               {results['isotropy']:.4f}")
        print(f"\n--- Underlying Factors ---")
        print(f"Mean Entropy (S_H):     {results['mean_entropy']:.4f}")
        print(f"Mean Dispersion (ρ):    {results['mean_dispersion']:.4f}")
        print(f"Std Dispersion:         {results['std_dispersion']:.4f}")
        print(f"\n--- Component Contributions ---")
        for comp, val in results['components'].items():
            print(f"{comp:20s}: {val:.4f}")
        print(f"{'='*60}\n")
    
    return results


def compute_density_evolution(text: str, 
                             window_size: int = 3,
                             model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
    """
    Calcula evolução da densidade semântica ao longo do texto (sliding window).
    
    Útil para identificar regiões de alta vs. baixa densidade.
    """
    model = SentenceTransformer(model_name)
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 2]
    
    if len(sentences) < window_size:
        raise ValueError(f"Texto precisa ter ao menos {window_size} sentenças.")
    
    embeddings = model.encode(sentences, convert_to_tensor=False)
    embeddings = np.array(embeddings)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    densities = []
    positions = []
    
    for i in range(len(sentences) - window_size + 1):
        window_emb = embeddings_norm[i:i+window_size]
        
        # Calcular SD para esta janela
        sim_matrix = cosine_similarity(window_emb)
        distances = pdist(window_emb, metric='cosine')
        mean_dispersion = np.mean(distances)
        
        probs = np.abs(sim_matrix) / (np.sum(np.abs(sim_matrix), axis=1, keepdims=True) + 1e-8)
        entropy_vals = [entropy(p + 1e-10) for p in probs]
        mean_entropy = np.mean(entropy_vals)
        
        # SD simplificado para janela
        sd_window = 1.0 - mean_dispersion - (mean_entropy / np.log(window_size))
        sd_window = np.clip(sd_window, 0, 1)
        
        densities.append(sd_window)
        positions.append(i + window_size // 2)  # Centro da janela
    
    return {
        "densities": densities,
        "positions": positions,
        "mean_density": float(np.mean(densities)),
        "std_density": float(np.std(densities)),
        "max_density": float(np.max(densities)),
        "min_density": float(np.min(densities))
    }


# ============================================================
# 🔹 CLI Interface (Enhanced)
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calcula densidade semântica (SD) expandida com métricas EAT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analisar texto direto
  python semantic_density.py "First sentence. Second sentence. Third sentence."
  
  # Analisar arquivo
  python semantic_density.py prompt.txt --verbose
  
  # Calcular evolução temporal
  python semantic_density.py paper.txt --evolution --window 5
  
  # Usar modelo diferente
  python semantic_density.py text.txt --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
        """
    )
    
    parser.add_argument("input", type=str, 
                       help="Texto de entrada ou caminho para arquivo .txt/.md")
    parser.add_argument("--model", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Modelo de embeddings")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Mostrar análise detalhada")
    parser.add_argument("--evolution", "-e", action="store_true",
                       help="Calcular evolução da densidade ao longo do texto")
    parser.add_argument("--window", type=int, default=3,
                       help="Tamanho da janela para --evolution (default: 3)")
    parser.add_argument("--output", "-o", type=str,
                       help="Salvar resultados em JSON")
    
    args = parser.parse_args()
    
    # Ler conteúdo
    try:
        if args.input.endswith(".txt") or args.input.endswith(".md"):
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = args.input
    except Exception as e:
        print(f"❌ Erro ao ler entrada: {e}")
        return 1
    
    # Análise principal
    try:
        result = compute_semantic_density(text, args.model, verbose=args.verbose)
        
        if not args.verbose:
            print(f"\n📊 Semantic Density Analysis")
            print(f"{'-'*40}")
            print(f"→ SD Score (Composite):  {result['semantic_density']:.4f}")
            print(f"→ SD Score (Original):   {result['sd_original']:.4f}")
            print(f"→ Coherence (μ):         {result['coherence']:.4f}")
            print(f"→ Isotropy:              {result['isotropy']:.4f}")
            print(f"→ Intrinsic Dimension:   {result['intrinsic_dim']}")
            print(f"→ Sentences analyzed:    {result['n_sentences']}\n")
        
        # Evolução temporal
        if args.evolution:
            print(f"\n📈 Computing density evolution (window={args.window})...")
            evolution = compute_density_evolution(text, args.window, args.model)
            
            print(f"\n--- Density Evolution ---")
            print(f"Mean density:  {evolution['mean_density']:.4f}")
            print(f"Std density:   {evolution['std_density']:.4f}")
            print(f"Max density:   {evolution['max_density']:.4f} (position {evolution['positions'][np.argmax(evolution['densities'])]})")
            print(f"Min density:   {evolution['min_density']:.4f} (position {evolution['positions'][np.argmin(evolution['densities'])]})")
            
            # Plot simples ASCII
            print(f"\nDensity across text:")
            densities_scaled = np.array(evolution['densities'])
            for i, (pos, dens) in enumerate(zip(evolution['positions'], densities_scaled)):
                bar_length = int(dens * 40)
                bar = "█" * bar_length
                print(f"  Position {pos:3d}: {bar} {dens:.3f}")
            
            result['evolution'] = evolution
        
        # Salvar JSON
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Results saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
