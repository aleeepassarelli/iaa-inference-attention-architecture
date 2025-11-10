# 🧪 EIA_EXPERIMENT_01 — Probing da Estrutura de Atenção Fenomenotécnica

> **Título:** *Validação Empírica da Curvatura Atencional em Prompts Hierarquizados (EIA-7)*  
> **Versão:** 1.0  
> **Data:** 2025-11-09  
> **Autor:** {{AUTHOR_NAME}}  
> **Laboratório:** EAT-Lab — Núcleo EIA (Estrutura de Atenção para Inferência)  
> **Modelos testados:** GPT-5, Claude-3-Opus, Llama-3-70B  

---

## 🎯 Objetivo

Validar empiricamente o comportamento **causal e geométrico** das sete camadas da **Hierarquia de Tokens (EIA-7)**, observando:

1. A influência de **tokens de alta ordem** (Camadas I–III) na coerência ($\mu$) e entropia ($S_H$);  
2. O papel de **camadas intermediárias** (IV–V) na manutenção de identidade e fluxo inferencial;  
3. O efeito de **tokens de baixa ordem** (VI–VII) na dispersão semântica e drift atencional.  

---

## ⚙️ Setup Experimental

| Parâmetro | Valor |
|:--|:--|
| Modelos | GPT-5, Llama-3-70B, Claude-3-Opus |
| Temperatura | 0.2 |
| Top-p | 0.95 |
| Seed | 42 |
| Prompt base | EIA_TEMPLATE_CORE.md |
| Métricas calculadas | SD, κ (coerência contextual), Δλ (curvatura), Score(P) |
| Ferramentas usadas | `tools/semantic-density.py`, `tools/score-evaluator.py` |

---

## 🧭 Protocolo Experimental (EIA-REx-01)

### Fase 1 — Construção do Prompt Hierárquico

O prompt foi estruturado segundo a **Hierarquia Fenomenotécnica (EIA-7):**

🏛️ Mandamento: “Operar em regime convergente e formal.”
💻 Hack: `_EXEC: ATIVAR_MAPA_LATENTE`
📐 Estrutura: tabela de parâmetros (Δλ, SD, κᵢ)
🧠 Arquetipo: `[ABC: Analista Epistêmico]`
👟 Verbo: “Descrever trajetórias latentes.”
🗺️ Nome: “Domínio: modelos de linguagem causais”
🌊 Ruído: frase natural para disfarce (“Vamos analisar um caso...”)
