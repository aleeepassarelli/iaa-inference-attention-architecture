"""
test_eat_score.py
=================
Test suite e exemplos de uso do framework EAT Score(P)
"""

from eat_score import EATScoreCalculator, EATVisualizer, ScoreP
import json

def test_basic_score():
    """Teste básico de cálculo de score"""
    print("\n" + "="*70)
    print("TEST 1: Basic Score Calculation")
    print("="*70)
    
    calculator = EATScoreCalculator(model_name="gpt2")
    
    text = """
    The Eiffel Tower is located in Paris, France. 
    It was constructed in 1889 for the World's Fair. 
    Today it remains one of the most iconic landmarks in the world.
    """
    
    score = calculator.compute_score_P(text, detailed=True)
    
    print(f"✓ Score calculated: {score.total_score:.4f}")
    print(f"  Components: ω={score.mean_omega:.3f}, ρ={score.mean_rho:.3f}, κ={score.mean_kappa:.3f}")
    
    assert score.total_score > 0, "Score should be positive"
    assert len(score.token_metrics) > 0, "Should have token metrics"
    
    print("✓ Test passed!")
    return score

def test_prompt_comparison():
    """Teste de comparação de prompts"""
    print("\n" + "="*70)
    print("TEST 2: Prompt Comparison")
    print("="*70)
    
    calculator = EATScoreCalculator(model_name="gpt2")
    
    prompts = [
        "Write a short story about a robot.",
        "Please write a brief narrative about an artificial being.",
        "Robot story short write.",  # Poor grammar
        "Compose a concise tale featuring a mechanical automaton."
    ]
    
    labels = ["Simple", "Verbose", "Poor", "Complex"]
    
    comparison = calculator.compare_prompts(prompts, labels)
    
    print(f"✓ Compared {len(prompts)} prompts")
    print(f"  Best: {comparison['best']['label']} (Score: {comparison['best']['score'].total_score:.4f})")
    print(f"  Worst: {comparison['worst']['label']} (Score: {comparison['worst']['score'].total_score:.4f})")
    
    # Visualize
    EATVisualizer.plot_comparison(comparison)
    
    print("✓ Test passed!")
    return comparison

def test_layer_evolution():
    """Teste de evolução por layer"""
    print("\n" + "="*70)
    print("TEST 3: Layer Evolution Analysis")
    print("="*70)
    
    calculator = EATScoreCalculator(model_name="gpt2")
    
    text = "Artificial intelligence is transforming society through automation and prediction."
    
    score = calculator.compute_score_P(text, detailed=True)
    
    # Analisar evolução
    if score.layer_metrics:
        densities = [lm.semantic_density for lm in score.layer_metrics]
        print(f"✓ Layer metrics computed for {len(densities)} layers")
        print(f"  Density range: [{min(densities):.3f}, {max(densities):.3f}]")
        
        # Check evolution trend
        early_density = densities[:len(densities)//3]
        late_density = densities[-len(densities)//3:]
        
        print(f"  Early layers mean: {sum(early_density)/len(early_density):.3f}")
        print(f"  Late layers mean:  {sum(late_density)/len(late_density):.3f}")
    
    print("✓ Test passed!")
    return score

def example_optimization():
    """Exemplo: otimizar prompt via iteração"""
    print("\n" + "="*70)
    print("EXAMPLE: Prompt Optimization via Score(P)")
    print("="*70)
    
    calculator = EATScoreCalculator(model_name="gpt2")
    
    # Versões progressivamente melhoradas
    versions = {
        "v1_basic": "Write story.",
        "v2_expanded": "Write a short story.",
        "v3_detailed": "Write a short story about a character facing a challenge.",
        "v4_optimized": "Compose a concise narrative describing how a protagonist overcomes adversity."
    }
    
    results = []
    for name, prompt in versions.items():
        score = calculator.compute_score_P(prompt, detailed=False)
        results.append({
            'version': name,
            'prompt': prompt,
            'score': score.total_score,
            'omega': score.mean_omega,
            'rho': score.mean_rho,
            'kappa': score.mean_kappa
        })
        print(f"{name:15s}: Score = {score.total_score:.4f}")
    
    # Identificar melhor
    best = max(results, key=lambda x: x['score'])
    print(f"\n✓ Best version: {best['version']} (Score: {best['score']:.4f})")
    print(f"  Prompt: {best['prompt']}")
    
    return results

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EAT SCORE(P) FRAMEWORK - TEST SUITE")
    print("="*70)
    
    # Run tests
    score1 = test_basic_score()
    comparison = test_prompt_comparison()
    score2 = test_layer_evolution()
    results = example_optimization()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
