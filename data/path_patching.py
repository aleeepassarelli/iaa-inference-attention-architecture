"""
path_patching.py
Implementação de Path Patching para análise de fluxo causal edge-by-edge
Identifica caminhos específicos de informação através da rede
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Edge:
    """Representa uma edge no computational graph"""
    source: str  # Ex: "L3_attn_h2"
    target: str  # Ex: "L4_mlp"
    source_layer: int
    target_layer: int
    
@dataclass
class PathPatchingResult:
    """Resultado de path patching"""
    edge_effects: Dict[Edge, float]
    critical_paths: List[List[Edge]]
    circuit_graph: nx.DiGraph
    
class PathPatcher:
    """
    Path Patching: análise edge-by-edge do fluxo causal
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        self.n_layers = len(self.model.transformer.h)
        self.n_heads = self.model.config.n_head
        
    def _get_component_output(self, layer_idx: int, component: str, 
                             input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extrai output de um componente específico
        
        component: 'attn' ou 'mlp' ou 'attn_h{i}' para head específico
        """
        activations = {}
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations['output'] = output[0].detach().clone()
            else:
                activations['output'] = output.detach().clone()
        
        # Registrar hook apropriado
        if component == 'attn':
            hook = self.model.transformer.h[layer_idx].attn.register_forward_hook(hook_fn)
        elif component == 'mlp':
            hook = self.model.transformer.h[layer_idx].mlp.register_forward_hook(hook_fn)
        elif component.startswith('attn_h'):
            head_idx = int(component.split('_h')[1])
            def head_hook_fn(module, input, output):
                attn_out = output[0]  # [batch, seq, n_heads, d_head]
                activations['output'] = attn_out[:, :, head_idx, :].detach().clone()
            hook = self.model.transformer.h[layer_idx].attn.register_forward_hook(head_hook_fn)
        
        # Forward pass
        with torch.no_grad():
            self.model(input_ids)
        
        hook.remove()
        return activations['output']
    
    def patch_edge(self,
                   clean_input: torch.Tensor,
                   corrupt_input: torch.Tensor,
                   source_layer: int,
                   source_component: str,
                   target_layer: int,
                   target_component: str,
                   target_token_id: int) -> float:
        """
        Patch uma edge específica: source -> target
        
        Procedimento:
        1. Run corrupted até source_layer
        2. Patch output de source_component com clean version
        3. Continue corrupted até target_layer
        4. Patch input de target_component com resultado do passo 2
        5. Continue até final
        6. Medir efeito
        """
        
        # Baseline: fully corrupted
        with torch.no_grad():
            outputs_corrupt = self.model(corrupt_input)
            corrupt_logit = F.softmax(outputs_corrupt.logits[0, -1], dim=-1)[target_token_id].item()
        
        # Get clean activation at source
        clean_source_act = self._get_component_output(source_layer, source_component, clean_input)
        
        # Path-patched run
        patched_activation = {}
        
        def source_patch_hook(module, input, output):
            # Substituir output do source com clean
            return clean_source_act
        
        def target_receive_hook(module, input, output):
            # Target recebe a ativação patched
            # (simplificado - assume que fluxo residual passa informação)
            patched_activation['received'] = True
            return output
        
        # Registrar hooks
        if source_component == 'attn':
            hook1 = self.model.transformer.h[source_layer].attn.register_forward_hook(source_patch_hook)
        elif source_component == 'mlp':
            hook1 = self.model.transformer.h[source_layer].mlp.register_forward_hook(source_patch_hook)
        
        if target_component == 'attn':
            hook2 = self.model.transformer.h[target_layer].attn.register_forward_hook(target_receive_hook)
        elif target_component == 'mlp':
            hook2 = self.model.transformer.h[target_layer].mlp.register_forward_hook(target_receive_hook)
        
        # Forward com patch
        with torch.no_grad():
            outputs_patched = self.model(corrupt_input)
            patched_logit = F.softmax(outputs_patched.logits[0, -1], dim=-1)[target_token_id].item()
        
        hook1.remove()
        hook2.remove()
        
        # Edge effect
        effect = patched_logit - corrupt_logit
        
        return effect
    
    def discover_circuit(self,
                        clean_input: torch.Tensor,
                        corrupt_input: torch.Tensor,
                        target_token_id: int,
                        threshold: float = 0.01,
                        max_edges: int = 100) -> PathPatchingResult:
        """
        Descobre circuito causal via path patching iterativo
        
        Args:
            threshold: edge com efeito < threshold são descartadas
            max_edges: máximo de edges a testar (por eficiência)
        """
        
        # Definir componentes possíveis
        components = []
        for layer in range(self.n_layers):
            components.append((layer, 'attn'))
            components.append((layer, 'mlp'))
        
        # Testar todas as edges possíveis (source -> target com target_layer > source_layer)
        edge_effects = {}
        edges_tested = 0
        
        print(f"Testing edges (max: {max_edges})...")
        
        for i, (src_layer, src_comp) in enumerate(components):
            for j, (tgt_layer, tgt_comp) in enumerate(components):
                if tgt_layer <= src_layer:
                    continue  # Edge deve ir para frente
                
                if edges_tested >= max_edges:
                    break
                
                edge = Edge(
                    source=f"L{src_layer}_{src_comp}",
                    target=f"L{tgt_layer}_{tgt_comp}",
                    source_layer=src_layer,
                    target_layer=tgt_layer
                )
                
                effect = self.patch_edge(
                    clean_input, corrupt_input,
                    src_layer, src_comp,
                    tgt_layer, tgt_comp,
                    target_token_id
                )
                
                edge_effects[edge] = effect
                edges_tested += 1
                
                if edges_tested % 10 == 0:
                    print(f"  Tested {edges_tested} edges...")
        
        # Filtrar edges acima do threshold
        critical_edges = {e: eff for e, eff in edge_effects.items() 
                         if abs(eff) >= threshold}
        
        print(f"\nFound {len(critical_edges)} critical edges (threshold: {threshold})")
        
        # Construir grafo do circuito
        circuit_graph = nx.DiGraph()
        
        for edge, effect in critical_edges.items():
            circuit_graph.add_edge(
                edge.source,
                edge.target,
                weight=effect,
                effect=effect
            )
        
        # Identificar caminhos críticos (source -> sink paths)
        # Source: layers iniciais, Sink: layers finais
        sources = [node for node in circuit_graph.nodes() 
                  if node.startswith('L0') or node.startswith('L1')]
        sinks = [node for node in circuit_graph.nodes() 
                if node.startswith(f'L{self.n_layers-1}') or node.startswith(f'L{self.n_layers-2}')]
        
        critical_paths = []
        for source in sources:
            for sink in sinks:
                if nx.has_path(circuit_graph, source, sink):
                    paths = list(nx.all_simple_paths(circuit_graph, source, sink, cutoff=5))
                    critical_paths.extend(paths[:3])  # Top 3 paths per source-sink pair
        
        return PathPatchingResult(
            edge_effects=edge_effects,
            critical_paths=critical_paths[:10],  # Top 10 paths
            circuit_graph=circuit_graph
        )
    
    def visualize_circuit(self, result: PathPatchingResult, 
                         save_path: Optional[str] = None):
        """Visualiza circuito descoberto"""
        
        G = result.circuit_graph
        
        if len(G.nodes()) == 0:
            print("No circuit found!")
            return
        
        # Layout hierárquico por layer
        pos = {}
        layer_counts = defaultdict(int)
        
        for node in G.nodes():
            layer = int(node.split('_')[0][1:])  # Extract layer number
            y_pos = -layer  # Negative so layer 0 is at top
            x_pos = layer_counts[layer]
            layer_counts[layer] += 1
            pos[node] = (x_pos, y_pos)
        
        # Centralizar cada layer
        for layer in layer_counts:
            nodes_in_layer = [n for n in G.nodes() if int(n.split('_')[0][1:]) == layer]
            count = len(nodes_in_layer)
            offset = -count / 2
            for i, node in enumerate(sorted(nodes_in_layer)):
                x, y = pos[node]
                pos[node] = (offset + i, y)
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        # Edges coloridas por effect
        edges = G.edges()
        weights = [G[u][v]['effect'] for u, v in edges]
        
        # Normalizar weights para colormap
        weights_array = np.array(weights)
        norm_weights = (weights_array - weights_array.min()) / (weights_array.max() - weights_array.min() + 1e-8)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=800, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Draw edges with varying widths and colors
        for (u, v), weight, norm_w in zip(edges, weights, norm_weights):
            color = plt.cm.RdYlGn(norm_w)
            width = 1 + 5 * norm_w  # Width proportional to effect
            nx.draw_networkx_edges(G, pos, [(u, v)], 
                                  edge_color=[color], width=width,
                                  alpha=0.7, arrows=True, 
                                  arrowsize=15, arrowstyle='->')
        
        plt.title("Causal Circuit (Path Patching)", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"CIRCUIT DISCOVERY SUMMARY")
        print(f"{'='*70}")
        print(f"Total nodes: {len(G.nodes())}")
        print(f"Total edges: {len(G.edges())}")
        print(f"Critical paths found: {len(result.critical_paths)}")
        
        # Top edges
        sorted_edges = sorted(result.edge_effects.items(), 
                            key=lambda x: abs(x[1]), reverse=True)
        print(f"\nTop-10 edges by effect:")
        for i, (edge, effect) in enumerate(sorted_edges[:10], 1):
            print(f"  {i:2d}. {edge.source:15s} -> {edge.target:15s} | Effect: {effect:+.4f}")
        
        # Sample path
        if result.critical_paths:
            print(f"\nSample critical path:")
            path = result.critical_paths[0]
            print("  " + " -> ".join(path))
        
        print(f"{'='*70}\n")


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    patcher = PathPatcher(model_name="gpt2", 
                         device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Inputs
    clean_prompt = "The Eiffel Tower is in Paris"
    corrupt_prompt = "The Eiffel Tower is in London"
    
    clean_tokens = patcher.tokenizer(clean_prompt, return_tensors="pt").input_ids.to(patcher.device)
    corrupt_tokens = patcher.tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(patcher.device)
    
    target_id = patcher.tokenizer.encode(" Paris", add_special_tokens=False)[0]
    
    # Descobrir circuito
    result = patcher.discover_circuit(
        clean_tokens, corrupt_tokens,
        target_id,
        threshold=0.005,  # Threshold para edge significance
        max_edges=50      # Limitar para eficiência
    )
    
    # Visualizar
    patcher.visualize_circuit(result, save_path="circuit_diagram.png")
