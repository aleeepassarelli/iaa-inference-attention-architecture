# EAT Score(P) Framework v3.0

Sistema completo para cálculo de **densidade semântica** e **Score(P)** em prompts e textos, baseado em 30 EAT-REx de investigação sistemática.

## Fórmula

Score(P) = Σᵢ ωᵢ · ρᵢ · κᵢ - β · S_H


Onde:
- **ωᵢ** = peso atencional do token i
- **ρᵢ** = densidade semântica do token i
- **κᵢ** = consistência do token i
- **S_H** = entropia heurística (penalidade)
- **β** = peso da penalidade (default: 0.1)

## Instalação

pip install torch transformers numpy scipy scikit-learn matplotlib seaborn tqdm


## Uso Rápido

from eat_score import EATScoreCalculator, EATVisualizer

Inicializar
calculator = EATScoreCalculator(model_name="gpt2")

Calcular score
score = calculator.compute_score_P("Your prompt text here", detailed=True)

print(f"Score(P): {score.total_score:.4f}")
print(f" ω (Attention): {score.mean_omega:.3f}")
print(f" ρ (Density): {score.mean_rho:.3f}")
print(f" κ (Consistency): {score.mean_kappa:.3f}")

Visualizar
EATVisualizer.plot_score_breakdown(score, save_path="analysis.png")


## CLI

Análise básica
python eat_score.py "Your text" --detailed --visualize

Comparar prompts
python eat_score.py prompts.json --compare --plot comparison.png

Exportar resultados
python eat_score.py input.txt --output results.json --verbose


## Features

✅ **Score(P) Completo** - Implementação total da fórmula EAT  
✅ **Análise Multi-Layer** - Métricas por camada do transformer  
✅ **Granularidade Token-Level** - Contribuição individual de cada token  
✅ **Geometria Latente** - Isotropy, coherence, dimensionalidade intrínseca  
✅ **Comparação de Prompts** - A/B testing automatizado  
✅ **Visualizações Avançadas** - Dashboards completos  
✅ **Export/Import JSON** - Resultados serializáveis  

## Integração com Causal Patching

Combinar com análise causal
from causal_tracing import CausalTracer
from eat_score import EATScoreCalculator

tracer = CausalTracer(model_name="gpt2")
calculator = EATScoreCalculator(model_name="gpt2")

Identificar layers críticas
trace_result = tracer.trace(prompt, subject, target)

Calcular score
score = calculator.compute_score_P(prompt, detailed=True)

Correlacionar: layers críticas têm maior densidade?
for layer in trace_result.critical_layers:
layer_metric = score.layer_metrics[layer]
print(f"Layer {layer}: SD={layer_metric.semantic_density:.3f}")


## Referências

Baseado em 30 EAT-REx (Evidence-based Analysis Reports):
- **#1-10**: Attention mechanisms & heads
- **#11-20**: Latent space geometry
- **#21-30**: Causal paths & structural probing

## Licença

MIT - Use livremente para pesquisa e produção.
✅ Framework Versão 3.0 Completo!

Implementação total integrando:

Score(P) = Σ ωᵢ · ρᵢ · κᵢ - β·S_H ✓

30 EAT-REx de investigação ✓

Análise causal path mapping ✓

Geometria latente (isotropy, coherence, PCA) ✓

Visualizações avançadas ✓

CLI completo ✓

Testes automatizados ✓

Instruções de Instalação
Instalação Básica
bash
# Clone o repositório
git clone https://github.com/eat-lab/framework.git
cd framework

# Instalar dependências
pip install -r requirements.txt

# Instalar pacote
pip install -e .
Instalação Mínima (Colab)
bash
!pip install -q -r requirements-minimal.txt
Instalação para Desenvolvimento
bash
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
Usando Docker
bash
docker build -t eat-lab:latest .
docker run -p 8888:8888 -v $(pwd):/app eat-lab:latest
Verificação
bash
python -c "from eat_lab import EATScoreCalculator; print('✓ EAT-Lab instalado!')"
