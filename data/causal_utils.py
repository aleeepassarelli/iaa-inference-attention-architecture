"""
causal_utils.py
Utilitários para análise causal: metrics, visualization, data handling
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json

class CausalMetrics:
    """Métricas para avaliar análises causais"""
    
    @staticmethod
    def compute_necessity(clean_output: float, ablated_output: float) -> float:
        """
        Necessity: quanto o componente é necessário
        Necessity = P(Y|X) - P(Y|X\C)
        """
        return clean_output - ablated_output
    
    @staticmethod
    def compute_sufficiency(isolated_output: float, baseline: float) -> float:
        """
        Sufficiency: quanto o componente sozinho é suficiente
        Sufficiency = P(Y|C) - P(Y|baseline)
        """
        return isolated_output - baseline
    
    @staticmethod
    def compute_fidelity(circuit_output: float, full_model_output: float) -> float:
        """
        Fidelity: quão bem o circuito aproxima o modelo completo
        """
        return circuit_output / (full_model_output + 1e-10)
    
    @staticmethod
    def compute_compression_ratio(circuit_size: int, full_graph_size: int) -> float:
        """Taxa de compressão do circuito"""
        return 1.0 - (circuit_size / full_graph_size)
    
    @staticmethod
    def compute_intervention_effect(pre_intervention: float, 
                                   post_intervention: float) -> float:
        """Efeito de uma intervenção"""
        return post_intervention - pre_intervention


class VisualizationUtils:
    """Utilitários para visualização"""
    
    @staticmethod
    def plot_heatmap(matrix: np.ndarray, 
                    xlabel: str = "Position",
                    ylabel: str = "Layer",
                    title: str = "Activation Heatmap",
                    save_path: Optional[str] = None):
        """Plot heatmap genérico"""
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, cmap='RdYlBu_r', center=0, 
                   cbar_kws={'label': 'Activation'})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_attention_pattern(attention_weights: torch.Tensor,
                              tokens: List[str],
                              layer: int,
                              head: int,
                              save_path: Optional[str] = None):
        """Visualiza padrão de atenção de um head"""
        
        # attention_weights: [n_heads, seq_len, seq_len]
        attn = attention_weights[head].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                   cmap='viridis', cbar_kws={'label': 'Attention Weight'})
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(f'Attention Pattern - Layer {layer}, Head {head}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_comparison(values1: List[float], 
                       values2: List[float],
                       labels: List[str],
                       title: str = "Comparison",
                       legend: Tuple[str, str] = ("Method 1", "Method 2"),
                       save_path: Optional[str] = None):
        """Compara duas séries de valores"""
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, values1, width, label=legend[0], alpha=0.8)
        ax.bar(x + width/2, values2, width, label=legend[1], alpha=0.8)
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Value')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class DataUtils:
    """Utilitários para manipulação de dados"""
    
    @staticmethod
    def create_counterfactual_dataset(prompts: List[str],
                                     subjects: List[str],
                                     replacements: List[str]) -> List[Tuple[str, str]]:
        """
        Cria dataset de contrafactuais
        
        Returns: List[(clean_prompt, corrupt_prompt)]
        """
        dataset = []
        
        for prompt, subject, replacement in zip(prompts, subjects, replacements):
            corrupt = prompt.replace(subject, replacement)
            dataset.append((prompt, corrupt))
        
        return dataset
    
    @staticmethod
    def batch_tokenize(tokenizer, texts: List[str], device: str = "cuda") -> torch.Tensor:
        """Tokeniza batch de textos"""
        return tokenizer(texts, return_tensors="pt", padding=True).input_ids.to(device)
    
    @staticmethod
    def save_results(results: Dict, filepath: str):
        """Salva resultados em JSON"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    @staticmethod
    def load_results(filepath: str) -> Dict:
        """Carrega resultados de JSON"""
        with open(filepath, 'r') as f:
            results = json.load(f)
        return results


class BenchmarkUtils:
    """Utilitários para benchmarking"""
    
    @staticmethod
    def benchmark_patching_methods(pipeline,
                                   test_cases: List[Dict],
                                   methods: List[str] = ['tracing', 'patching', 'attribution']):
        """
        Benchmark diferentes métodos de análise causal
        
        test_cases: List[{'prompt': str, 'subject': str, 'target': str}]
        """
        import time
        
        results = {method: {'times': [], 'accuracies': []} for method in methods}
        
        for case in test_cases:
            print(f"\nTesting: {case['prompt'][:50]}...")
            
            # Preparar
            clean_tokens = pipeline.tokenizer(case['prompt'], return_tensors="pt").input_ids.to(pipeline.device)
            target_id = pipeline.tokenizer.encode(case['target'], add_special_tokens=False)[0]
            
            if 'tracing' in methods:
                start = time.time()
                trace_result = pipeline.tracer.trace(
                    case['prompt'], case['subject'], case['target']
                )
                elapsed = time.time() - start
                results['tracing']['times'].append(elapsed)
                
                # Accuracy: % of critical layers identified
                accuracy = len(trace_result.critical_layers) / pipeline.tracer.model.config.n_layer
                results['tracing']['accuracies'].append(accuracy)
            
            # ... Similar para outros métodos
        
        # Summarize
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS")
        print(f"{'='*70}")
        
        for method in methods:
            avg_time = np.mean(results[method]['times'])
            avg_acc = np.mean(results[method]['accuracies'])
            print(f"{method:15s} | Avg Time: {avg_time:6.3f}s | Avg Accuracy: {avg_acc:.3f}")
        
        return results


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    # Exemplo de uso dos utils
    
    # 1. Métricas
    metrics = CausalMetrics()
    necessity = metrics.compute_necessity(clean_output=0.8, ablated_output=0.3)
    print(f"Necessity: {necessity:.4f}")
    
    # 2. Visualização
    viz = VisualizationUtils()
    
    # Criar dummy heatmap
    dummy_matrix = np.random.randn(12, 50)  # 12 layers, 50 positions
    viz.plot_heatmap(dummy_matrix, xlabel="Position", ylabel="Layer",
                     title="Example Activation Heatmap")
    
    # 3. Data utils
    data = DataUtils()
    
    counterfactuals = data.create_counterfactual_dataset(
        prompts=["The Eiffel Tower is in Paris", "Apple was founded by Steve Jobs"],
        subjects=["Eiffel Tower", "Apple"],
        replacements=["Big Ben", "Microsoft"]
    )
    
    print("\nCounterfactuals created:")
    for clean, corrupt in counterfactuals:
        print(f"  Clean: {clean}")
        print(f"  Corrupt: {corrupt}\n")
