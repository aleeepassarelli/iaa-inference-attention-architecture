#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eat_score.py
============
Framework Completo EAT Score(P) - Engenharia de Atenção e Inferência

Implementa Score(P) = Σ ωᵢ · ρᵢ · κᵢ - β·S_H

Integra 30 EAT-REx de investigação sistemática:
- Atenção e heads (REx #1-10)
- Geometria latente (REx #11-20)
- Causal paths (REx #21-30)

Versão: 3.0 (2025-11-09)
Autor: EAT-Lab Framework
Licença: MIT
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 🔹 DATA STRUCTURES
# ============================================================

@dataclass
class TokenMetrics:
    """Métricas por token individual"""
    token: str
    position: int
    omega: float        # ωᵢ - peso atencional
    rho: float          # ρᵢ - densidade semântica
    kappa: float        # κᵢ - consistência
    contribution: float # ωᵢ · ρᵢ · κᵢ

@dataclass
class LayerMetrics:
    """Métricas por layer"""
    layer_idx: int
    mean_omega: float
    mean_rho: float
    mean_kappa: float
    semantic_density: float
    heuristic_entropy: float
    coherence: float
    isotropy: float
    causal_effect: float  # Do causal tracing

@dataclass
class ScoreP:
    """Score(P) completo com todos os componentes"""
    total_score: float
    raw_score: float           # Antes de penalidade
    entropy_penalty: float     # β·S_H
    
    # Componentes agregados
    mean_omega: float
    mean_rho: float
    mean_kappa: float
    mean_entropy: float
    
    # Métricas geométricas
    isotropy: float
    intrinsic_dim: int
    coherence: float
    
    # Per-token breakdown
    token_metrics: List[TokenMetrics]
    
    # Per-layer breakdown
    layer_metrics: List[LayerMetrics]
    
    # Metadados
    n_tokens: int
    n_layers: int
    model_name: str


# ============================================================
# 🔹 CORE: EAT SCORE CALCULATOR
# ============================================================

class EATScoreCalculator:
    """
    Calculador completo de Score(P) para prompts/textos.
    
    Integra:
    - Análise de atenção (ω)
    - Densidade semântica (ρ)
    - Consistência (κ)
    - Entropia heurística (S_H)
    - Geometria latente
    - Causal tracing
    """
    
    def __init__(self, 
                 model_name: str = "gpt2",
                 device: str = "cuda",
                 beta: float = 0.1):
        """
        Args:
            model_name: modelo transformer (GPT-2, LLaMA, etc)
            device: 'cuda' ou 'cpu'
            beta: peso da penalidade de entropia em Score(P)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        self.model_name = model_name
        self.beta = beta
        self.n_layers = len(self.model.transformer.h)
        
        print(f"✓ EAT Score Calculator initialized")
        print(f"  Model: {model_name}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Device: {self.device}")
        print(f"  Beta (entropy weight): {beta}\n")
    
    def compute_score_P(self, 
                       text: str,
                       detailed: bool = True,
                       include_causal: bool = False) -> ScoreP:
        """
        Calcula Score(P) completo para um texto/prompt.
        
        Score(P) = Σᵢ ωᵢ · ρᵢ · κᵢ - β · S_H
        
        Args:
            text: texto de entrada
            detailed: incluir métricas detalhadas por token/layer
            include_causal: executar causal tracing (mais lento)
        
        Returns:
            ScoreP com todos os componentes
        """
        
        # Tokenizar
        tokens = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512).to(self.device)
        token_strs = [self.tokenizer.decode([t]) for t in tokens.input_ids[0]]
        
        # Forward pass com todas as informações
        with torch.no_grad():
            outputs = self.model(
                **tokens,
                output_hidden_states=True,
                output_attentions=True
            )
        
        hidden_states = outputs.hidden_states  # Tuple[layers] of [batch, seq, dim]
        attentions = outputs.attentions        # Tuple[layers] of [batch, heads, seq, seq]
        
        # ===== COMPONENTE 1: ωᵢ (Attention Weights) =====
        omega_scores = self._compute_omega(attentions, token_strs)
        
        # ===== COMPONENTE 2: ρᵢ (Semantic Density) =====
        rho_scores = self._compute_rho(hidden_states, token_strs)
        
        # ===== COMPONENTE 3: κᵢ (Consistency) =====
        kappa_scores = self._compute_kappa(text, token_strs)
        
        # ===== COMPONENTE 4: S_H (Heuristic Entropy) =====
        S_H = self._compute_entropy(hidden_states, outputs.logits)
        
        # ===== MÉTRICAS GEOMÉTRICAS =====
        geometric_metrics = self._compute_geometric_metrics(hidden_states)
        
        # ===== SCORE(P) TOTAL =====
        # Σᵢ ωᵢ · ρᵢ · κᵢ
        token_contributions = []
        for i, (token, omega, rho, kappa) in enumerate(zip(token_strs, omega_scores, rho_scores, kappa_scores)):
            contribution = omega * rho * kappa
            token_contributions.append(contribution)
            
            if detailed:
                token_contributions.append(TokenMetrics(
                    token=token,
                    position=i,
                    omega=float(omega),
                    rho=float(rho),
                    kappa=float(kappa),
                    contribution=float(contribution)
                ))
        
        raw_score = sum([tm.contribution if isinstance(tm, TokenMetrics) else tm 
                        for tm in token_contributions])
        
        # Aplicar penalidade de entropia
        entropy_penalty = self.beta * S_H
        total_score = raw_score - entropy_penalty
        
        # ===== MÉTRICAS POR LAYER (se detailed) =====
        layer_metrics_list = []
        if detailed:
            layer_metrics_list = self._compute_layer_metrics(
                hidden_states, attentions, 
                include_causal=include_causal,
                text=text
            )
        
        # ===== CONSTRUIR OBJETO SCORE =====
        score_obj = ScoreP(
            total_score=float(total_score),
            raw_score=float(raw_score),
            entropy_penalty=float(entropy_penalty),
            mean_omega=float(np.mean(omega_scores)),
            mean_rho=float(np.mean(rho_scores)),
            mean_kappa=float(np.mean(kappa_scores)),
            mean_entropy=float(S_H),
            isotropy=geometric_metrics['isotropy'],
            intrinsic_dim=geometric_metrics['intrinsic_dim'],
            coherence=geometric_metrics['coherence'],
            token_metrics=token_contributions if detailed else [],
            layer_metrics=layer_metrics_list,
            n_tokens=len(token_strs),
            n_layers=self.n_layers,
            model_name=self.model_name
        )
        
        return score_obj
    
    def _compute_omega(self, attentions: Tuple, token_strs: List[str]) -> np.ndarray:
        """
        Calcula ωᵢ - pesos de atenção por token.
        
        Estratégia: agregar atenção recebida através de todas as layers/heads.
        """
        seq_len = len(token_strs)
        
        # Agregar atenção de todas as layers e heads
        # attentions: (layers, [batch, heads, seq, seq])
        total_attention = torch.zeros(seq_len, device=self.device)
        
        for layer_attn in attentions:
            # layer_attn: [batch, heads, seq, seq]
            # Somar atenção recebida por cada token (axis=-2)
            attn_received = layer_attn[0].sum(dim=0).sum(dim=0)  # [seq]
            total_attention += attn_received
        
        # Normalizar
        omega = total_attention / (total_attention.sum() + 1e-8)
        
        return omega.cpu().numpy()
    
    def _compute_rho(self, hidden_states: Tuple, token_strs: List[str]) -> np.ndarray:
        """
        Calcula ρᵢ - densidade semântica por token.
        
        Estratégia: norma do hidden state + variância local.
        """
        # Usar última layer
        final_hidden = hidden_states[-1][0]  # [seq, dim]
        
        # Norma L2 por token (magnitude da representação)
        norms = torch.norm(final_hidden, dim=-1)  # [seq]
        
        # Densidade local: similaridade com vizinhos
        final_hidden_norm = F.normalize(final_hidden, dim=-1)
        
        rho_scores = []
        for i in range(len(token_strs)):
            # Norma base
            base_rho = norms[i].item()
            
            # Similaridade com vizinhos (±1 token)
            neighbor_sim = 0.0
            count = 0
            
            if i > 0:
                neighbor_sim += (final_hidden_norm[i] @ final_hidden_norm[i-1]).item()
                count += 1
            if i < len(token_strs) - 1:
                neighbor_sim += (final_hidden_norm[i] @ final_hidden_norm[i+1]).item()
                count += 1
            
            if count > 0:
                neighbor_sim /= count
            
            # Combinar: norma + similaridade local
            rho = 0.7 * base_rho + 0.3 * neighbor_sim
            rho_scores.append(rho)
        
        # Normalizar para [0, 1]
        rho_array = np.array(rho_scores)
        rho_array = (rho_array - rho_array.min()) / (rho_array.max() - rho_array.min() + 1e-8)
        
        return rho_array
    
    def _compute_kappa(self, text: str, token_strs: List[str]) -> np.ndarray:
        """
        Calcula κᵢ - consistência por token.
        
        Estratégia: executar múltiplas inferências com dropout e medir variância.
        Para eficiência, usa aproximação via entropy da distribuição de atenção.
        """
        # Aproximação rápida: usar entropy inversa da atenção recebida
        # Alta entropy = baixa consistência, baixa entropy = alta consistência
        
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**tokens, output_attentions=True)
            attentions = outputs.attentions
        
        kappa_scores = []
        
        for i in range(len(token_strs)):
            # Para cada token, calcular entropy da atenção recebida
            attention_received = []
            
            for layer_attn in attentions:
                # layer_attn: [batch, heads, seq, seq]
                # Atenção recebida por token i de todos os outros tokens
                attn_to_i = layer_attn[0, :, :, i]  # [heads, seq]
                attention_received.append(attn_to_i.mean(dim=0).cpu().numpy())  # [seq]
            
            # Stack e calcular entropy
            attn_dist = np.concatenate(attention_received)
            attn_dist = attn_dist / (attn_dist.sum() + 1e-8)
            
            token_entropy = entropy(attn_dist + 1e-10)
            
            # Consistência = inverso da entropy (normalizado)
            kappa = 1.0 / (1.0 + token_entropy)
            kappa_scores.append(kappa)
        
        return np.array(kappa_scores)
    
    def _compute_entropy(self, hidden_states: Tuple, logits: torch.Tensor) -> float:
        """
        Calcula S_H - entropia heurística.
        
        Combina:
        - Entropy da predição de próximo token
        - Variância das representações
        """
        # Entropy da predição (última posição)
        final_logits = logits[0, -1, :]  # [vocab_size]
        probs = F.softmax(final_logits, dim=-1)
        pred_entropy = entropy(probs.cpu().numpy())
        
        # Variância das hidden states (dispersão representacional)
        final_hidden = hidden_states[-1][0]  # [seq, dim]
        hidden_variance = torch.var(final_hidden).item()
        
        # Combinar
        S_H = 0.7 * pred_entropy + 0.3 * hidden_variance
        
        return S_H
    
    def _compute_geometric_metrics(self, hidden_states: Tuple) -> Dict:
        """
        Calcula métricas geométricas do espaço latente.
        """
        # Usar última layer
        final_hidden = hidden_states[-1][0].cpu().numpy()  # [seq, dim]
        
        # Normalizar para hiperesfera
        hidden_norm = final_hidden / (np.linalg.norm(final_hidden, axis=1, keepdims=True) + 1e-8)
        
        # 1. Isotropy
        centroid = hidden_norm.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        angles = np.arccos(np.clip(hidden_norm @ centroid, -1, 1))
        isotropy = 1.0 / (np.std(angles) + 1e-8)
        isotropy = np.clip(1.0 - (isotropy / 10), 0, 1)
        
        # 2. Intrinsic Dimensionality
        pca = PCA()
        pca.fit(hidden_norm)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = int(np.argmax(cumsum >= 0.95) + 1)
        
        # 3. Coherence (similaridade sequencial)
        coherence_scores = []
        for i in range(len(hidden_norm) - 1):
            sim = np.dot(hidden_norm[i], hidden_norm[i+1])
            coherence_scores.append(sim)
        
        mean_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return {
            'isotropy': float(isotropy),
            'intrinsic_dim': intrinsic_dim,
            'coherence': float(mean_coherence)
        }
    
    def _compute_layer_metrics(self, 
                               hidden_states: Tuple,
                               attentions: Tuple,
                               include_causal: bool,
                               text: str) -> List[LayerMetrics]:
        """
        Calcula métricas detalhadas por layer.
        """
        layer_metrics = []
        
        for layer_idx in range(self.n_layers):
            # Hidden state desta layer
            hidden = hidden_states[layer_idx][0].cpu().numpy()  # [seq, dim]
            hidden_norm = hidden / (np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-8)
            
            # Atenção desta layer
            attn = attentions[layer_idx][0].cpu().numpy()  # [heads, seq, seq]
            
            # Métricas básicas
            mean_omega = np.mean(attn)
            
            # Densidade semântica (dispersão)
            if len(hidden_norm) > 1:
                distances = pdist(hidden_norm, metric='cosine')
                mean_rho = 1.0 - np.mean(distances)
            else:
                mean_rho = 1.0
            
            # Consistência (via entropy da atenção)
            attn_flat = attn.reshape(-1)
            attn_probs = attn_flat / (attn_flat.sum() + 1e-8)
            attn_entropy = entropy(attn_probs + 1e-10)
            mean_kappa = 1.0 / (1.0 + attn_entropy)
            
            # SD layer-wise
            sim_matrix = cosine_similarity(hidden_norm)
            probs_sim = np.abs(sim_matrix) / (np.abs(sim_matrix).sum(axis=1, keepdims=True) + 1e-8)
            entropy_vals = [entropy(p + 1e-10) for p in probs_sim]
            mean_entropy = np.mean(entropy_vals)
            
            sd = 1.0 - mean_entropy / np.log(len(hidden_norm) + 1)
            
            # Coherence
            coherence_scores = []
            for i in range(len(hidden_norm) - 1):
                coherence_scores.append(np.dot(hidden_norm[i], hidden_norm[i+1]))
            mean_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
            
            # Isotropy
            centroid = hidden_norm.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            angles = np.arccos(np.clip(hidden_norm @ centroid, -1, 1))
            isotropy = np.clip(1.0 - (np.std(angles) / 10), 0, 1)
            
            # Causal effect (placeholder - requires full causal tracing)
            causal_effect = 0.0  # TODO: integrate with causal_tracing.py
            
            layer_metrics.append(LayerMetrics(
                layer_idx=layer_idx,
                mean_omega=float(mean_omega),
                mean_rho=float(mean_rho),
                mean_kappa=float(mean_kappa),
                semantic_density=float(sd),
                heuristic_entropy=float(mean_entropy),
                coherence=float(mean_coherence),
                isotropy=float(isotropy),
                causal_effect=float(causal_effect)
            ))
        
        return layer_metrics
    
    def compare_prompts(self, prompts: List[str], labels: Optional[List[str]] = None) -> Dict:
        """
        Compara Score(P) de múltiplos prompts.
        
        Útil para:
        - A/B testing de prompts
        - Identificar qual formulação tem maior densidade semântica
        - Otimização de templates
        """
        if labels is None:
            labels = [f"Prompt {i+1}" for i in range(len(prompts))]
        
        results = []
        
        print(f"Comparing {len(prompts)} prompts...")
        for prompt, label in zip(tqdm(prompts), labels):
            score = self.compute_score_P(prompt, detailed=False)
            results.append({
                'label': label,
                'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                'score': score
            })
        
        # Ordenar por score total
        results_sorted = sorted(results, key=lambda x: x['score'].total_score, reverse=True)
        
        return {
            'results': results_sorted,
            'best': results_sorted[0],
            'worst': results_sorted[-1],
            'mean_score': np.mean([r['score'].total_score for r in results]),
            'std_score': np.std([r['score'].total_score for r in results])
        }


# ============================================================
# 🔹 VISUALIZATION
# ============================================================

class EATVisualizer:
    """Visualizações avançadas para Score(P)"""
    
    @staticmethod
    def plot_score_breakdown(score: ScoreP, save_path: Optional[str] = None):
        """Dashboard completo de Score(P)"""
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # ===== Plot 1: Score Components =====
        ax1 = fig.add_subplot(gs[0, 0])
        components = ['Raw Score', 'Entropy\nPenalty', 'Total Score']
        values = [score.raw_score, -score.entropy_penalty, score.total_score]
        colors = ['green', 'red', 'blue']
        
        ax1.bar(components, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Score Value')
        ax1.set_title('Score(P) Breakdown', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        
        # ===== Plot 2: ω, ρ, κ Components =====
        ax2 = fig.add_subplot(gs[0, 1])
        comp_names = ['ω (Attention)', 'ρ (Density)', 'κ (Consistency)']
        comp_values = [score.mean_omega, score.mean_rho, score.mean_kappa]
        
        ax2.bar(comp_names, comp_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax2.set_ylabel('Mean Value')
        ax2.set_title('Component Metrics', fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        # ===== Plot 3: Geometric Properties =====
        ax3 = fig.add_subplot(gs[0, 2])
        geo_names = ['Isotropy', 'Coherence', '1 - Entropy\n(norm)']
        geo_values = [
            score.isotropy, 
            score.coherence,
            1.0 - (score.mean_entropy / 10)  # Normalize
        ]
        
        ax3.bar(geo_names, geo_values, color=['#95E1D3', '#F38181', '#FFBE7A'], alpha=0.8)
        ax3.set_ylabel('Value')
        ax3.set_title('Geometric Properties', fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(axis='y', alpha=0.3)
        
        # ===== Plot 4: Token Contributions =====
        if score.token_metrics:
            ax4 = fig.add_subplot(gs[1, :])
            
            tokens = [tm.token.replace('\n', '\\n')[:15] for tm in score.token_metrics]
            contributions = [tm.contribution for tm in score.token_metrics]
            
            colors_token = ['green' if c > np.mean(contributions) else 'red' 
                           for c in contributions]
            
            ax4.bar(range(len(tokens)), contributions, color=colors_token, alpha=0.7)
            ax4.set_xticks(range(len(tokens)))
            ax4.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax4.set_ylabel('Contribution (ω·ρ·κ)')
            ax4.set_title('Token-Level Contributions to Score(P)', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            ax4.axhline(y=np.mean(contributions), color='blue', linestyle='--', 
                       label='Mean', linewidth=2)
            ax4.legend()
        
        # ===== Plot 5: Layer Evolution =====
        if score.layer_metrics:
            ax5 = fig.add_subplot(gs[2, :2])
            
            layers = [lm.layer_idx for lm in score.layer_metrics]
            sd_values = [lm.semantic_density for lm in score.layer_metrics]
            coherence_values = [lm.coherence for lm in score.layer_metrics]
            isotropy_values = [lm.isotropy for lm in score.layer_metrics]
            
            ax5.plot(layers, sd_values, marker='o', label='Semantic Density', linewidth=2)
            ax5.plot(layers, coherence_values, marker='s', label='Coherence', linewidth=2)
            ax5.plot(layers, isotropy_values, marker='^', label='Isotropy', linewidth=2)
            
            ax5.set_xlabel('Layer')
            ax5.set_ylabel('Metric Value')
            ax5.set_title('Layer-wise Metric Evolution', fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)
        
        # ===== Plot 6: Summary Stats =====
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_text = f"""
SCORE(P) SUMMARY
{'='*25}

Total Score: {score.total_score:.4f}
Raw Score:   {score.raw_score:.4f}
Entropy Pen: {score.entropy_penalty:.4f}

--- Components ---
ω (Attention):   {score.mean_omega:.3f}
ρ (Density):     {score.mean_rho:.3f}
κ (Consistency): {score.mean_kappa:.3f}

--- Geometry ---
Isotropy:        {score.isotropy:.3f}
Coherence:       {score.coherence:.3f}
Intrinsic Dim:   {score.intrinsic_dim}

--- Meta ---
Tokens:  {score.n_tokens}
Layers:  {score.n_layers}
Model:   {score.model_name}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('EAT Score(P) Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_comparison(comparison: Dict, save_path: Optional[str] = None):
        """Visualiza comparação entre múltiplos prompts"""
        
        results = comparison['results']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Total Scores
        labels = [r['label'] for r in results]
        scores = [r['score'].total_score for r in results]
        
        axes[0, 0].barh(range(len(labels)), scores, color='steelblue', alpha=0.7)
        axes[0, 0].set_yticks(range(len(labels)))
        axes[0, 0].set_yticklabels(labels)
        axes[0, 0].set_xlabel('Total Score(P)')
        axes[0, 0].set_title('Score Comparison', fontweight='bold')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Component Breakdown
        omegas = [r['score'].mean_omega for r in results]
        rhos = [r['score'].mean_rho for r in results]
        kappas = [r['score'].mean_kappa for r in results]
        
        x = np.arange(len(labels))
        width = 0.25
        
        axes[0, 1].bar(x - width, omegas, width, label='ω', alpha=0.8)
        axes[0, 1].bar(x, rhos, width, label='ρ', alpha=0.8)
        axes[0, 1].bar(x + width, kappas, width, label='κ', alpha=0.8)
        
        axes[0, 1].set_xlabel('Prompts')
        axes[0, 1].set_ylabel('Component Value')
        axes[0, 1].set_title('Component Comparison', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Geometric Properties
        isotropies = [r['score'].isotropy for r in results]
        coherences = [r['score'].coherence for r in results]
        
        axes[1, 0].plot(range(len(labels)), isotropies, marker='o', label='Isotropy', linewidth=2)
        axes[1, 0].plot(range(len(labels)), coherences, marker='s', label='Coherence', linewidth=2)
        
        axes[1, 0].set_xlabel('Prompt Index')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Geometric Properties', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_xticks(range(len(labels)))
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
        
        # Plot 4: Summary Table
        axes[1, 1].axis('off')
        
        table_data = []
        for r in results:
            table_data.append([
                r['label'][:20],
                f"{r['score'].total_score:.3f}",
                f"{r['score'].mean_omega:.2f}",
                f"{r['score'].mean_rho:.2f}",
                f"{r['score'].mean_kappa:.2f}"
            ])
        
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Prompt', 'Score', 'ω', 'ρ', 'κ'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Highlight best/worst
        for i, cell in table.get_celld().items():
            if i[0] == 1:  # Best (first data row)
                cell.set_facecolor('#90EE90')
            elif i[0] == len(results):  # Worst (last data row)
                cell.set_facecolor('#FFB6C1')
        
        plt.suptitle('Prompt Comparison Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison saved to {save_path}")
        
        plt.show()


# ============================================================
# 🔹 CLI INTERFACE
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EAT Score(P) Calculator - Complete Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic score calculation
  python eat_score.py "Your prompt text here" --verbose
  
  # Analyze from file
  python eat_score.py prompt.txt --model gpt2 --detailed
  
  # Compare multiple prompts
  python eat_score.py prompts.json --compare --visualize
  
  # Export results
  python eat_score.py "Text" --output results.json --plot dashboard.png
        """
    )
    
    parser.add_argument("input", type=str,
                       help="Text, file path, or JSON with prompts")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model name (default: gpt2)")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Entropy penalty weight (default: 0.1)")
    parser.add_argument("--detailed", action="store_true",
                       help="Include per-token and per-layer metrics")
    parser.add_argument("--compare", action="store_true",
                       help="Compare
