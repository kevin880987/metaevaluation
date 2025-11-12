"""
Simple Framework Comparison: Original vs CAE-M
==============================================

This script compares:
1. Original multi-indicator framework 
2. CAE-M single method approach
3. Combined approach

Focus on end-to-end performance evaluation for practical decision making.
"""

import os
import pandas as pd
import numpy as np
import time
from configuration import Configuration
from data_holder import DataHolder
from health_indicator_extraction import extract
from scoring import score, evaluate_mece
from data_helpers import save_pkl, load_pkl

def run_comparison_experiment():
    """
    Run the main comparison experiment
    """
    print("Framework Comparison: Original vs CAE-M")
    print("=" * 60)
    
    # Define comparison scenarios
    scenarios = {
        "original": {
            "name": "Original Multi-Indicator Framework",
            "model": "cvae",
            "health_indicators": [
                "PC1", 
                "ICA MD", "ICA HT2", 
                "REC MD", "REC HT2", 
                "LS MD", "LS HT2", 
                "RECLS MD", "RECLS HT2"
            ]
        },
        "cae_m": {
            "name": "CAE-M Single Method",
            "model": "cae_m",
            "health_indicators": ["CAE_ANOMALY", "CAE_LATENT_MD"]
        }
    }
    
    results = {}
    
    # Run each scenario
    for scenario_key, scenario_config in scenarios.items():
        print(f"\n{'='*40}")
        print(f"Running: {scenario_config['name']}")
        print(f"{'='*40}")
        
        try:
            result = run_single_scenario(scenario_key, scenario_config)
            results[scenario_key] = result
            print(f"✓ {scenario_config['name']} completed successfully")
            
        except Exception as e:
            print(f"✗ Error in {scenario_config['name']}: {e}")
            results[scenario_key] = {"status": "failed", "error": str(e)}
    
    # Compare results
    compare_results(results)
    
    return results

def run_single_scenario(scenario_key, scenario_config, checkpoint=True):
    """
    Run a single scenario and return results
    """
    # Setup configuration
    config = Configuration()
    config.model = scenario_config['model']
    config.health_indicatores = scenario_config['health_indicators']
    
    # Update paths to avoid conflicts
    base_result_dir = os.path.join("..", "result", "comparison", scenario_key)
    config.result = base_result_dir + os.sep + config.model + os.sep
    config.training_dir = config.result + "training" + os.sep
    config.reconstruction_dir = config.result + "reconstruction" + os.sep
    config.health_indicator_dir = config.result + "health indicator" + os.sep
    config.deterioration_modeling_dir = config.result + "deterioration modeling" + os.sep + config.simulation + os.sep
    
    # Create directories
    config._make_directories([
        config.training_dir,
        config.reconstruction_dir,
        config.health_indicator_dir,
        config.deterioration_modeling_dir
    ])
    
    print(f"Model: {config.model}")
    print(f"Health Indicators: {config.health_indicatores}")
    print(f"Results directory: {config.result}")
    
    start_time = time.time()
    
    # 1. Load and prepare data
    print("\n1. Loading data...")
    holder = DataHolder()
    holder.read_files(config.data_dir)
    holder.prepare(config.test_ratio, 
                  getattr(config, 'input_sequence', [0, 1, 2, 3]), 
                  getattr(config, 'output_sequence', [0]))
    
    print(f"   Domains: {holder.domains}")
    
    # 2. Extract health indicators
    print("2. Extracting health indicators...")
    hi_dict = extract(config, holder, checkopint=checkpoint)
    
    # Save health indicators
    hi_file = os.path.join(config.health_indicator_dir, "hi_dict.pkl")
    save_pkl(hi_dict, hi_file)
    
    # 3. Score health indicators
    print("3. Scoring health indicators...")
    rul_dict = {}
    for d in holder.domains:
        rul_dict[d] = dict(holder.get("Y", domain=d, output_sequence=getattr(config, 'output_sequence', [0])))
    
    score_df = score(hi_dict, rul_dict)
    score_file = os.path.join(config.health_indicator_dir, "score.csv")
    score_df.to_csv(score_file)
    
    # 4. MECE evaluation
    print("4. MECE evaluation...")
    mece_df = evaluate_mece(score_df)
    mece_file = os.path.join(config.health_indicator_dir, "mece.csv")
    mece_df.to_csv(mece_file)
    
    execution_time = time.time() - start_time
    
    # Collect key metrics
    results = {
        "status": "completed",
        "execution_time": execution_time,
        "model": config.model,
        "health_indicators": config.health_indicatores,
        "score_df": score_df,
        "mece_df": mece_df,
        "mean_score": score_df.mean().mean(),
        "max_score": score_df.max().max(),
        "std_score": score_df.mean().std(),
        "n_indicators": len(config.health_indicatores),
        "result_dir": config.result
    }
    
    print(f"   Execution time: {execution_time:.2f} seconds")
    print(f"   Mean score: {results['mean_score']:.4f}")
    print(f"   Max score: {results['max_score']:.4f}")
    
    return results

def compare_results(results):
    """
    Compare and display results
    """
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    successful_results = {k: v for k, v in results.items() if v.get("status") == "completed"}
    
    if len(successful_results) < 2:
        print("Need at least 2 successful results for comparison")
        return
    
    # Create comparison table
    comparison_data = []
    for scenario_key, result in successful_results.items():
        comparison_data.append({
            'Approach': scenario_key.upper(),
            'Model': result['model'],
            'Indicators': result['n_indicators'],
            'Mean_Score': f"{result['mean_score']:.4f}",
            'Max_Score': f"{result['max_score']:.4f}",
            'Std_Score': f"{result['std_score']:.4f}",
            'Time_sec': f"{result['execution_time']:.2f}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nCOMPARISON TABLE:")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    
    # Detailed analysis
    print(f"\n{'='*60}")
    print("DETAILED ANALYSIS")
    print(f"{'='*60}")
    
    original = successful_results.get('original')
    cae_m = successful_results.get('cae_m')
    
    if original and cae_m:
        print("\nOriginal Framework vs CAE-M Comparison:")
        print("-" * 40)
        
        print(f"\nPERFORMANCE:")
        print(f"Original Mean Score: {original['mean_score']:.4f}")
        print(f"CAE-M Mean Score:    {cae_m['mean_score']:.4f}")
        improvement = ((cae_m['mean_score'] - original['mean_score']) / original['mean_score']) * 100
        print(f"Performance Change:  {improvement:+.2f}%")
        
        print(f"\nEFFICIENCY:")
        print(f"Original Indicators: {original['n_indicators']}")
        print(f"CAE-M Indicators:    {cae_m['n_indicators']}")
        reduction = ((original['n_indicators'] - cae_m['n_indicators']) / original['n_indicators']) * 100
        print(f"Indicator Reduction: {reduction:.1f}%")
        
        print(f"\nSPEED:")
        print(f"Original Time:       {original['execution_time']:.2f}s")
        print(f"CAE-M Time:          {cae_m['execution_time']:.2f}s")
        speedup = original['execution_time'] / cae_m['execution_time']
        print(f"Speed Factor:        {speedup:.2f}x")
        
        # Recommendation
        print(f"\n{'='*40}")
        print("RECOMMENDATION")
        print(f"{'='*40}")
        
        if cae_m['mean_score'] > original['mean_score']:
            print("✓ CAE-M outperforms the original framework")
            print(f"  - Better performance: {improvement:+.2f}%")
            print(f"  - Fewer indicators: {reduction:.1f}% reduction")
            print(f"  - Faster execution: {speedup:.2f}x speedup")
            print("\n→ RECOMMEND: Switch to CAE-M approach")
        else:
            print("✓ Original framework outperforms CAE-M")
            print(f"  - Better performance: {-improvement:+.2f}%")
            print(f"  - More comprehensive: {original['n_indicators']} indicators")
            if speedup < 1:
                print(f"  - Faster execution: {1/speedup:.2f}x speedup")
            print("\n→ RECOMMEND: Keep original framework")
        
        # Trade-off analysis
        print(f"\nTRADE-OFF ANALYSIS:")
        print("-" * 20)
        if cae_m['n_indicators'] < original['n_indicators']:
            print("✓ CAE-M is simpler (fewer indicators to manage)")
        if cae_m['execution_time'] < original['execution_time']:
            print("✓ CAE-M is faster to execute")
        if abs(improvement) < 5:  # Less than 5% difference
            print("✓ Performance difference is minimal")
            print("  → Consider CAE-M for simplicity if performance is acceptable")

def main():
    """
    Main execution function
    """
    print("Starting Framework Comparison Experiment...")
    
    results = run_comparison_experiment()
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    
    # Save summary
    summary_dir = os.path.join("..", "result", "comparison")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_file = os.path.join(summary_dir, "comparison_summary.txt")
    
    # You can add code here to save detailed summary to file
    print(f"\nResults can be found in individual scenario directories")
    print(f"Base comparison directory: {summary_dir}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 Comparison completed successfully!")
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        print("\nPlease ensure:")
        print("1. All required dependencies are installed")
        print("2. Data files are available in the data directory")
        print("3. CAE-M model implementation is working correctly")
