"""
integrated_pipeline.py
Pipeline unificado combinando todos os métodos de análise causal
Workflow completo: Trace -> Patch -> Attribute -> Circuit -> Score
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import numpy as np

# Importar classes dos scripts anteriores
from causal_tracing import CausalTracer, CausalTraceResult
from activation_patching import ActivationPatcher, PatchingResult
from attribution_patching import AttributionPatcher, AttributionResult
from path_patching import PathPatcher, PathPatchingResult

@dataclass
class ComprehensiveAnalysis:
    """Container para análise completa"""
    trace_result: CausalTraceResult
    patching_results: List[PatchingResult]
    attribution_result: AttributionResult
    circuit_result: PathPatchingResult
    score_evolution: List[float]
    summary: Dict

class CausalAnalysisPipeline:
    """
    Pipeline completo para análise causal de prompts
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        
        # Inicializar todos os analyzers
        print("Initializing pipeline components...")
        self.tracer = CausalTracer(model_name, device)
        self.patcher = ActivationPatcher(model_name, device)
        self.attributor = AttributionPatcher(model_name, device)
        self.path_patcher = PathPatcher(model_name, device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Pipeline ready!\n")
    
    def analyze_prompt(self,
                      prompt: str,
                      subject: str,
                      target_token: str,
                      corrupt_prompt: Optional[str] = None,
                      run_full_scan: bool = False) -> ComprehensiveAnalysis:
        """
        Análise completa de um prompt
        
        Args:
            prompt: prompt original (clean)
            subject: parte do prompt a ser traced
            target_token: token esperado
            corrupt_prompt: versão corrompida (se None, gera automaticamente)
            run_full_scan: se True, roda scan completo de patching (lento)
        """
        
        print(f"{'='*70}")
        print(f"COMPREHENSIVE CAUSAL ANALYSIS")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"Subject: {subject}")
        print(f"Target: {target_token}")
        print(f"{'='*70}\n")
        
        # Preparar inputs
        clean_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        
        # Gerar corrupt prompt se não fornecido
        if corrupt_prompt is None:
            # Simple corruption: substituir subject por random token
            corrupt_prompt = prompt.replace(subject, "something")
        
        corrupt_tokens = self.tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(self.device)
        
        # ========== 1. CAUSAL TRACING ==========
        print("Step 1/5: Running causal tracing...")
        trace_result = self.tracer.trace(
            prompt=prompt,
            subject=subject,
            target_token=target_token,
            noise_level=3.0
        )
        print(f"  → Critical layers identified: {trace_result.critical_layers}\n")
        
        # ========== 2. ACTIVATION PATCHING ==========
        print("Step 2/5: Running activation patching...")
        
        if run_full_scan:
            patching_results = self.patcher.comprehensive_patch_scan(
                clean_tokens, corrupt_tokens, target_id
            )
            print(f"  → Analyzed all components (heads + MLPs)\n")
        else:
            # Patch apenas critical layers identificadas
            patching_results = []
            for layer in trace_result.critical_layers[:5]:  # Top 5
                # Patch MLP desta layer
                result = self.patcher.patch_mlp(
                    clean_tokens, corrupt_tokens, layer, target_id
                )
                patching_results.append(result)
            print(f"  → Patched {len(patching_results)} critical MLPs\n")
        
        # ========== 3. ATTRIBUTION PATCHING ==========
        print("Step 3/5: Running attribution patching (gradient-based)...")
        attribution_result = self.attributor.attribute(
            clean_tokens, corrupt_tokens, target_id
        )
        print(f"  → Total effect attributed: {attribution_result.total_effect:.4f}\n")
        
        # ========== 4. CIRCUIT DISCOVERY ==========
        print("Step 4/5: Discovering causal circuit...")
        circuit_result = self.path_patcher.discover_circuit(
            clean_tokens, corrupt_tokens, target_id,
            threshold=0.005,
            max_edges=40
        )
        print(f"  → Circuit: {len(circuit_result.circuit_graph.nodes())} nodes, "
              f"{len(circuit_result.circuit_graph.edges())} edges\n")
        
        # ========== 5. SCORE EVOLUTION ==========
        print("Step 5/5: Computing Score(P) evolution across layers...")
        score_evolution = self._compute_score_evolution(clean_tokens)
        print(f"  → Score evolution computed\n")
        
        # ========== SUMMARY ==========
        summary = self._generate_summary(
            trace_result, patching_results, 
            attribution_result, circuit_result
        )
        
        analysis = ComprehensiveAnalysis(
            trace_result=trace_result,
            patching_results=patching_results,
            attribution_result=attribution_result,
            circuit_result=circuit_result,
            score_evolution=score_evolution,
            summary=summary
        )
        
        return analysis
    
    def _compute_score_evolution(self, tokens: torch.Tensor) -> List[float]:
        """
        Computa Score(P) em cada layer usando logit lens
        """
        model = self.tracer.model
        scores = []
        
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
            
            for hidden in outputs.hidden_states:
                # Project to vocab
                logits = hidden @ model.lm_head.weight.T
                
                # Score = entropy of prediction (inverse)
                probs = torch.softmax(logits[0, -1], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                
                # Score inversely proportional to entropy
                score = 1.0 / (1.0 + entropy)
                scores.append(score)
        
        return scores
    
    def _generate_summary(self, trace_result, patching_results, 
                         attribution_result, circuit_result) -> Dict:
        """Gera sumário executivo"""
        
        # Top components de patching
        if patching_results:
            top_patching = sorted(patching_results, 
                                 key=lambda x: x.causal_effect, 
                                 reverse=True)[:5]
            top_patch_names = [r.component_name for r in top_patching]
        else:
            top_patch_names = []
        
        summary = {
            'critical_layers': trace_result.critical_layers,
            'top_patching_components': top_patch_names,
            'top_attribution_components': [name for name, _ in attribution_result.top_components[:5]],
            'circuit_size': {
                'nodes': len(circuit_result.circuit_graph.nodes()),
                'edges': len(circuit_result.circuit_graph.edges())
            },
            'baseline_prob': trace_result.baseline_prob,
            'corrupted_prob': trace_result.corrupted_prob,
            'total_attribution': attribution_result.total_effect
        }
        
        return summary
    
    def visualize_comprehensive(self, analysis: ComprehensiveAnalysis,
                               save_dir: Optional[str] = "./results"):
        """
        Cria visualização completa em um dashboard
        """
        import os
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ===== Plot 1: Causal Tracing =====
        ax1 = fig.add_subplot(gs[0, :2])
        layers = list(analysis.trace_result.layer_effects.keys())
        effects = list(analysis.trace_result.layer_effects.values())
        colors = ['red' if l in analysis.trace_result.critical_layers else 'blue' 
                 for l in layers]
        ax1.bar(layers, effects, color=colors, alpha=0.7)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Causal Effect')
        ax1.set_title('Causal Tracing Results')
        ax1.grid(axis='y', alpha=0.3)
        
        # ===== Plot 2: Top Patching Components =====
        ax2 = fig.add_subplot(gs[1, :2])
        if analysis.patching_results:
            top_results = sorted(analysis.patching_results, 
                               key=lambda x: x.causal_effect, reverse=True)[:10]
            names = [r.component_name for r in top_results]
            effects = [r.causal_effect for r in top_results]
            ax2.barh(range(len(names)), effects, alpha=0.7)
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names)
            ax2.set_xlabel('Causal Effect')
            ax2.set_title('Top Patching Components')
            ax2.invert_yaxis()
        
        # ===== Plot 3: Attribution by Layer =====
        ax3 = fig.add_subplot(gs[2, :2])
        n_layers = len([k for k in analysis.attribution_result.component_attributions.keys() 
                       if 'attn' in k])
        attn_attrs = [analysis.attribution_result.component_attributions.get(f'L{i}_attn', 0) 
                     for i in range(n_layers)]
        mlp_attrs = [analysis.attribution_result.component_attributions.get(f'L{i}_mlp', 0) 
                    for i in range(n_layers)]
        
        x = np.arange(n_layers)
        width = 0.35
        ax3.bar(x - width/2, attn_attrs, width, label='Attention', alpha=0.8)
        ax3.bar(x + width/2, mlp_attrs, width, label='MLP', alpha=0.8)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Attribution')
        ax3.set_title('Attribution by Component Type')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # ===== Plot 4: Score Evolution =====
        ax4 = fig.add_subplot(gs[0, 2])
        ax4.plot(analysis.score_evolution, marker='o', linewidth=2)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Score(P)')
        ax4.set_title('Score Evolution')
        ax4.grid(alpha=0.3)
        
        # ===== Plot 5: Summary Stats =====
        ax5 = fig.add_subplot(gs[1:, 2])
        ax5.axis('off')
        
        summary_text = f"""
ANALYSIS SUMMARY
{'='*30}

Critical Layers:
{', '.join(map(str, analysis.summary['critical_layers']))}

Baseline Prob: {analysis.summary['baseline_prob']:.4f}
Corrupted Prob: {analysis.summary['corrupted_prob']:.4f}

Circuit Size:
  Nodes: {analysis.summary['circuit_size']['nodes']}
  Edges: {analysis.summary['circuit_size']['edges']}

Total Attribution:
  {analysis.summary['total_attribution']:.4f}

Top Patching Components:
"""
        for comp in analysis.summary['top_patching_components']:
            summary_text += f"  • {comp}\n"
        
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Comprehensive Causal Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_dir:
            plt.savefig(f"{save_dir}/comprehensive_analysis.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # Salvar summary como JSON
        if save_dir:
            with open(f"{save_dir}/summary.json", 'w') as f:
                json.dump(analysis.summary, f, indent=2)
    
    def export_results(self, analysis: ComprehensiveAnalysis, 
                      output_path: str = "analysis_results.json"):
        """Exporta todos os resultados para JSON"""
        
        # Converter para formato serializável
        results = {
            'trace': {
                'layer_effects': {str(k): v for k, v in analysis.trace_result.layer_effects.items()},
                'critical_layers': analysis.trace_result.critical_layers,
                'baseline_prob': analysis.trace_result.baseline_prob,
                'corrupted_prob': analysis.trace_result.corrupted_prob
            },
            'patching': [
                {
                    'component': r.component_name,
                    'causal_effect': r.causal_effect,
                    'necessity': r.necessity
                }
                for r in analysis.patching_results[:20]  # Top 20
            ],
            'attribution': {
                'total_effect': analysis.attribution_result.total_effect,
                'top_components': [
                    {'name': name, 'value': val}
                    for name, val in analysis.attribution_result.top_components[:20]
                ]
            },
            'circuit': {
                'n_nodes': len(analysis.circuit_result.circuit_graph.nodes()),
                'n_edges': len(analysis.circuit_result.circuit_graph.edges()),
                'critical_paths': analysis.circuit_result.critical_paths
            },
            'score_evolution': analysis.score_evolution,
            'summary': analysis.summary
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults exported to {output_path}")


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    # Inicializar pipeline
    pipeline = CausalAnalysisPipeline(
        model_name="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Análise completa
    analysis = pipeline.analyze_prompt(
        prompt="The Eiffel Tower is located in Paris",
        subject="Eiffel Tower",
        target_token=" Paris",
        corrupt_prompt="The Eiffel Tower is located in London",
        run_full_scan=False  # True para análise completa (mais lento)
    )
    
    # Visualizar
    pipeline.visualize_comprehensive(analysis, save_dir="./analysis_results")
    
    # Exportar
    pipeline.export_results(analysis, "eiffel_analysis.json")
    
    print("\n✓ Analysis complete!")
