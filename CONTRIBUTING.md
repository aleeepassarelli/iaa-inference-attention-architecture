# 🤝 Guia de Contribuição — Estrutura de Atenção para Inferência (EIA)

> **Versão:** 1.0.0  
> **Projeto:** EIA — *Estrutura de Atenção para Inferência*  
> **Licença:** MIT  
> **Idioma base:** Português / English / 中文  

---

## 🧭 1. Princípios de Contribuição

Toda contribuição ao projeto **EIA** deve respeitar três princípios fundamentais:

1. **Reprodutibilidade** — Cada resultado, métrica ou código deve poder ser reproduzido.
2. **Rastreabilidade** — Cada modificação deve possuir commit rastreável e metadados verificáveis.
3. **Coerência semântica** — A contribuição deve preservar a integridade heurística e estrutural do framework.

> 🔍 *O EIA é um projeto científico de Engenharia Semântica. Cada contribuição deve ter embasamento empírico ou código-fonte de validação.*

---

## ⚙️ 2. Estrutura do Repositório

```

EIA/
├── README.md
├── CONTRIBUTING.md          # (este arquivo)
├── LICENSE
├── CHANGELOG.md
├── docs/
├── tools/
├── templates/
├── examples/
└── research/

````

- **docs/** — Documentação técnica e teórica (EIA_THEORY.md, MANUAL_COMPLETO.md, etc.)  
- **tools/** — Scripts de análise e validação (SD, Score(P), probing, etc.)  
- **templates/** — Modelos de prompts estruturados segundo a EIA-7  
- **examples/** — Estudos de caso e validações empíricas  
- **research/** — Implementações experimentais e papers derivados  

---

## 🧠 3. Tipos de Contribuição

| Tipo | Descrição | Exemplo |
|:--|:--|:--|
| **📄 Documentação** | Revisão ou expansão dos manuais e docs teóricos. | `docs/EIA_THEORY.md` |
| **🧩 Template** | Novo modelo de prompt com métricas validadas. | `templates/prompt-experimental.md` |
| **🧰 Ferramenta** | Scripts, notebooks ou validadores SD/κ/μ. | `tools/score-evaluator.py` |
| **🔬 Pesquisa** | Experimentos empíricos com datasets e papers. | `research/latent-steering.ipynb` |
| **⚗️ POC** | Provas de conceito (mini frameworks, testes). | `examples/case-study-xx.md` |

---

## 🧾 4. Checklist de Submissão

Antes de abrir um Pull Request, verifique:

| Item | Requisito | Ferramenta de Validação |
|:--|:--|:--|
| [ ] | SD ≥ **0.80** | `tools/semantic-density.py` |
| [ ] | Tokens ≤ **1500** | `tools/token-counter.py` |
| [ ] | 5–7 exemplos *baseshot* | `tools/baseshot-validator.sh` |
| [ ] | Testado em ≥ 2 modelos | CLI ou API wrapper |
| [ ] | Score(P) calculado e registrado | `tools/score-evaluator.py` |
| [ ] | Comentários documentados | `CONTRIBUTING.md` guidelines |
| [ ] | DOI / commit rastreável | `ACC_TRACK` logs |

---

## 🧪 5. Fluxo de Contribuição

1. **Fork** o repositório  
   ```bash
   git clone https://github.com/{{your_username}}/EIA.git
   cd EIA


2. **Crie uma branch dedicada**

   ```bash
   git checkout -b feature/{{feature_name}}
   ```

3. **Implemente e valide localmente**
   Execute as ferramentas internas de verificação:

   ```bash
   python tools/semantic-density.py templates/novo_prompt.md
   python tools/score-evaluator.py templates/novo_prompt.md
   bash tools/baseshot-validator.sh templates/novo_prompt.md
   ```

4. **Atualize documentação e exemplos**
   Inclua no `CHANGELOG.md` e `examples/` um caso ilustrativo.

5. **Abra um Pull Request (PR)**
   No título:

   ```
   feat(template): novo modelo SD>0.82 com validação causal κ=0.91
   ```

   E inclua a checklist preenchida.

---

## 📈 6. Padrão de Métricas e Logs

Cada nova contribuição (template, modelo, ferramenta) deve conter **metadados incorporados**:

```yaml
semantic_density: 0.84
entropy: 0.09
coherence_mu: 0.88
curvature_kappa: 0.91
scoreP: 0.87
version: 1.0.0
validated_on:
  - llama-3-70b
  - mistral-8x22b
commit: a81e52c
date: 2025-11-09
```

E opcionalmente um identificador de rastreio:

```
ACC_TRACK: session_id: "2025-11-09T21:15Z" model: "Gemma-2-27B" contributor: "@username"
```

---

## 🧩 7. Estilo e Convenções

### Linguagem

* Use português técnico claro.
* Evite metáforas ou figuras poéticas em docs técnicos (permitido apenas em prefácios).
* Prefira termos com equivalentes diretos em inglês e chinês.

### Código

* Use **PEP8** (Python) e tipagem explícita.
* Documente cada função com docstring científica (parâmetros e métricas).
* Inclua seed e versionamento de ambiente (`requirements.txt`).

### Nomenclatura

* Todos os nomes devem ser **semânticos e invariantes**.
* Prefira nomes curtos e explicativos, ex:
  `attention-prober.py`, `semantic-density.py`, `latent-field.md`.

---

## 🔬 8. Validação Científica

Antes da integração de qualquer novo método ou métrica:

1. Cite referências empíricas (**DOI, arXiv, GitHub commit**).
2. Descreva metodologia e resultados no formato:

   ```
   Método: Attention Patch (Olsson, 2022)
   Métrica: Δκᵢ = 0.12 (↓)
   Reprodutibilidade: +93% (n=10 runs)
   ```
3. Inclua links ou hashes dos datasets utilizados.
4. Preferência para código **open source e reprodutível**.

---

## 📜 9. Revisão e Aprovação

As contribuições são avaliadas por 2 revisores:

| Etapa              | Responsável   | Critério                           |
| :----------------- | :------------ | :--------------------------------- |
| Revisão Técnica    | Core Engineer | Padrão de código, métricas, logs   |
| Revisão Epistêmica | Research Lead | Coerência semântica e metodológica |

PRs são mesclados apenas após **dupla aprovação** e **validação cruzada**.

---

## 🧠 10. Recomendações de Pesquisa

Para extensão do EIA, priorizar estudos sobre:

* **Head Attribution & Concept Vectors**
* **Structural Probing & Manifold Geometry**
* **Causal Mediation & Latent Steering**
* **Semantic Drift Detection**
* **Skill Fusion em agentes compostos**

---

## 👥 11. Comunidade e Créditos

| Função             | Nome / Handle    |
| :----------------- | :--------------- |
| Autor Principal    | {{AUTHOR_NAME}}  |
| Mantenedor Técnico | {{MAINTAINER}}   |
| Contribuidores     | {{CONTRIBUTORS}} |

Participe das discussões em:

* **Discussions:** {{DISCUSSIONS_URL}}
* **Issues:** {{ISSUES_URL}}

---

## 📄 12. Licença

Este repositório está sob a licença **MIT**, conforme arquivo `LICENSE`.

Toda contribuição implica concordância com:

> “A preservação da coerência e rastreabilidade é condição para a evolução semântica coletiva.”

---

