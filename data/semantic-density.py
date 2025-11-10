#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_density.py
--------------------
Calcula a densidade semântica (SD) de um texto, template ou prompt.

Definição:
  SD = 1 - (E[entropia vetorial] / dispersão média)
onde:
  - alta SD → coerência e convergência semântica
  - baixa SD → ruído e dispersão semântica

Referência conceitual: Estrutura de Atenção para Inferência (EIA)
"""

import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

# ============================================================
# 🔹 Funções principais
# ============================================================

def compute_semantic_density(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Calcula a densidade semântica de um texto ou prompt.
    Retorna um dicionário com SD, entropia média e dispersão.
    """
    model = SentenceTransformer(model_name)
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 2]
    if len(sentences) < 2:
        raise ValueError("Texto muito curto. Forneça ao menos duas sentenças.")

    embeddings = model.encode(sentences, convert_to_tensor=False)
    sim_matrix = cosine_similarity(embeddings)

    # Dispersão média
    distances = pdist(embeddings, metric='cosine')
    mean_dispersion = np.mean(distances)

    # Entropia média
    probs = sim_matrix / np.sum(sim_matrix, axis=1, keepdims=True)
    entropy_vals = [entropy(p) for p in probs]
    mean_entropy = np.mean(entropy_vals)

    # Densidade Semântica (normalizada)
    sd = 1 - (mean_entropy / (mean_dispersion + 1e-8))
    sd = np.clip(sd, 0, 1)

    return {
        "semantic_density": float(sd),
        "mean_entropy": float(mean_entropy),
        "mean_dispersion": float(mean_dispersion),
        "n_sentences": len(sentences)
    }

# ============================================================
# 🔹 CLI Interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Calcula a densidade semântica (SD) de um texto/prompt.")
    parser.add_argument("input", type=str, help="Texto de entrada ou caminho para arquivo .txt/.md")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Modelo de embeddings (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    # Ler conteúdo
    try:
        if args.input.endswith(".txt") or args.input.endswith(".md"):
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = args.input
    except Exception as e:
        print(f"Erro ao ler entrada: {e}")
        return

    result = compute_semantic_density(text, args.model)
    print(f"\n📊 Semantic Density Analysis\n{'-'*40}")
    print(f"→ SD Score:           {result['semantic_density']:.4f}")
    print(f"→ Mean Entropy (H):   {result['mean_entropy']:.4f}")
    print(f"→ Mean Dispersion:    {result['mean_dispersion']:.4f}")
    print(f"→ Sentences analyzed: {result['n_sentences']}\n")


if __name__ == "__main__":
    main()
