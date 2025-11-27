"""
Hyperparameter tuning for MLP acoustic feature classifier
Tests different combinations of learning rate, hidden layer sizes, dropout, and batch size
"""

import os
import json
import itertools
from mlp_features_model import train_5fold_cv

# Hyperparameter search space
HYPERPARAMETER_GRID = {
    'learning_rate': [0.0001, 0.0005, 0.001],  # Lower learning rates
    'hidden_sizes': [
        [128, 64, 32],       # Small (best F1 before)
        [256, 128, 64],      # Medium
    ],
    'dropout': [0.5, 0.7],  # High dropout only
    'batch_size': [64],  # Fixed batch size
}

# Fixed parameters
FEATURES_CSV = "/mloscratch/users/gnahas/data/features/acoustic_features.csv"
LABELS_CSV = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
TRAIN_FOLDS_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
VAL_TEST_FOLDS_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"
BASE_OUTPUT_DIR = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/MLP_Tuning/V2"
NUM_EPOCHS = 100


def run_hyperparameter_search():
    """
    Run grid search over hyperparameters
    """
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Generate all combinations
    keys = list(HYPERPARAMETER_GRID.keys())
    values = list(HYPERPARAMETER_GRID.values())
    combinations = list(itertools.product(*values))
    
    print("="*80)
    print(f"HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Total combinations to test: {len(combinations)}")
    print(f"Parameters being tuned:")
    for key, value_list in HYPERPARAMETER_GRID.items():
        print(f"  - {key}: {value_list}")
    print("="*80 + "\n")
    
    all_results = []
    
    for idx, combo in enumerate(combinations):
        # Unpack combination
        params = dict(zip(keys, combo))
        
        print("\n" + "="*80)
        print(f"CONFIGURATION {idx+1}/{len(combinations)}")
        print("="*80)
        print(f"Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print("="*80 + "\n")
        
        # Create config name (no subfolder, just for identification)
        config_name = f"lr{params['learning_rate']}_hs{'-'.join(map(str, params['hidden_sizes']))}_drop{params['dropout']}_bs{params['batch_size']}"
        
        # Use a temporary directory that we'll delete after extracting results
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='mlp_tuning_')
        
        try:
            # Train model with this configuration
            summary = train_5fold_cv(
                features_csv=FEATURES_CSV,
                labels_csv=LABELS_CSV,
                train_folds_csv=TRAIN_FOLDS_CSV,
                val_test_folds_csv=VAL_TEST_FOLDS_CSV,
                output_dir=temp_dir,
                batch_size=params['batch_size'],
                num_epochs=NUM_EPOCHS,
                learning_rate=params['learning_rate'],
                hidden_sizes=params['hidden_sizes'],
                dropout=params['dropout']
            )
            
            # Store results with hyperparameters
            result = {
                'config_id': idx + 1,
                'config_name': config_name,
                'hyperparameters': params,
                'test_patient_majority': summary['test_patient_majority'],
                'test_patient_average': summary['test_patient_average'],
                'test_recording_level': summary['test_recording_level'],
            }
            all_results.append(result)
            
            print(f"\n✓ Configuration {idx+1} completed successfully")
            print(f"  Test Patient (Majority) - Acc: {summary['test_patient_majority']['avg_accuracy']:.2f}%, "
                  f"AUC: {summary['test_patient_majority']['avg_auc']:.4f}, "
                  f"F1: {summary['test_patient_majority']['avg_f1']:.4f}")
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"\n✗ Configuration {idx+1} failed with error: {e}")
            result = {
                'config_id': idx + 1,
                'config_name': config_name,
                'hyperparameters': params,
                'error': str(e)
            }
            all_results.append(result)
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Save intermediate results
        intermediate_path = os.path.join(BASE_OUTPUT_DIR, 'tuning_results_partial.json')
        with open(intermediate_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Save final results
    final_path = os.path.join(BASE_OUTPUT_DIR, 'tuning_results_final.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Find best configuration
    successful_results = [r for r in all_results if 'error' not in r]
    
    if successful_results:
        # Best by patient-level majority voting
        best_maj_accuracy = max(successful_results, 
                               key=lambda x: x['test_patient_majority']['avg_accuracy'])
        best_maj_auc = max(successful_results, 
                         key=lambda x: x['test_patient_majority']['avg_auc'])
        best_maj_f1 = max(successful_results, 
                        key=lambda x: x['test_patient_majority']['avg_f1'])
        
        # Best by patient-level average probability
        best_avg_accuracy = max(successful_results, 
                               key=lambda x: x['test_patient_average']['avg_accuracy'])
        best_avg_auc = max(successful_results, 
                         key=lambda x: x['test_patient_average']['avg_auc'])
        best_avg_f1 = max(successful_results, 
                        key=lambda x: x['test_patient_average']['avg_f1'])
        
        # Best by recording-level
        best_rec_accuracy = max(successful_results, 
                               key=lambda x: x['test_recording_level']['avg_accuracy'])
        best_rec_auc = max(successful_results, 
                         key=lambda x: x['test_recording_level']['avg_auc'])
        best_rec_f1 = max(successful_results, 
                        key=lambda x: x['test_recording_level']['avg_f1'])
        
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING COMPLETE")
        print("="*80)
        print(f"Total configurations tested: {len(combinations)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(all_results) - len(successful_results)}")
        
        print("\n" + "="*80)
        print("BEST CONFIGURATIONS - PATIENT-LEVEL (MAJORITY VOTING)")
        print("="*80)
        
        print("\n--- Best by Accuracy ---")
        print(f"Config: {best_maj_accuracy['config_name']}")
        print(f"Hyperparameters: {best_maj_accuracy['hyperparameters']}")
        print(f"Accuracy: {best_maj_accuracy['test_patient_majority']['avg_accuracy']:.2f}% "
              f"± {best_maj_accuracy['test_patient_majority']['std_accuracy']:.2f}%")
        print(f"AUC: {best_maj_accuracy['test_patient_majority']['avg_auc']:.4f} "
              f"± {best_maj_accuracy['test_patient_majority']['std_auc']:.4f}")
        print(f"F1: {best_maj_accuracy['test_patient_majority']['avg_f1']:.4f} "
              f"± {best_maj_accuracy['test_patient_majority']['std_f1']:.4f}")
        
        print("\n--- Best by AUC ---")
        print(f"Config: {best_maj_auc['config_name']}")
        print(f"Hyperparameters: {best_maj_auc['hyperparameters']}")
        print(f"AUC: {best_maj_auc['test_patient_majority']['avg_auc']:.4f} "
              f"± {best_maj_auc['test_patient_majority']['std_auc']:.4f}")
        print(f"Accuracy: {best_maj_auc['test_patient_majority']['avg_accuracy']:.2f}% "
              f"± {best_maj_auc['test_patient_majority']['std_accuracy']:.2f}%")
        print(f"F1: {best_maj_auc['test_patient_majority']['avg_f1']:.4f} "
              f"± {best_maj_auc['test_patient_majority']['std_f1']:.4f}")
        
        print("\n--- Best by F1 Score ---")
        print(f"Config: {best_maj_f1['config_name']}")
        print(f"Hyperparameters: {best_maj_f1['hyperparameters']}")
        print(f"F1: {best_maj_f1['test_patient_majority']['avg_f1']:.4f} "
              f"± {best_maj_f1['test_patient_majority']['std_f1']:.4f}")
        print(f"Accuracy: {best_maj_f1['test_patient_majority']['avg_accuracy']:.2f}% "
              f"± {best_maj_f1['test_patient_majority']['std_accuracy']:.2f}%")
        print(f"AUC: {best_maj_f1['test_patient_majority']['avg_auc']:.4f} "
              f"± {best_maj_f1['test_patient_majority']['std_auc']:.4f}")
        
        print("\n" + "="*80)
        print("BEST CONFIGURATIONS - PATIENT-LEVEL (AVERAGE PROBABILITY)")
        print("="*80)
        
        print("\n--- Best by Accuracy ---")
        print(f"Config: {best_avg_accuracy['config_name']}")
        print(f"Hyperparameters: {best_avg_accuracy['hyperparameters']}")
        print(f"Accuracy: {best_avg_accuracy['test_patient_average']['avg_accuracy']:.2f}% "
              f"± {best_avg_accuracy['test_patient_average']['std_accuracy']:.2f}%")
        print(f"AUC: {best_avg_accuracy['test_patient_average']['avg_auc']:.4f} "
              f"± {best_avg_accuracy['test_patient_average']['std_auc']:.4f}")
        print(f"F1: {best_avg_accuracy['test_patient_average']['avg_f1']:.4f} "
              f"± {best_avg_accuracy['test_patient_average']['std_f1']:.4f}")
        
        print("\n--- Best by AUC ---")
        print(f"Config: {best_avg_auc['config_name']}")
        print(f"Hyperparameters: {best_avg_auc['hyperparameters']}")
        print(f"AUC: {best_avg_auc['test_patient_average']['avg_auc']:.4f} "
              f"± {best_avg_auc['test_patient_average']['std_auc']:.4f}")
        print(f"Accuracy: {best_avg_auc['test_patient_average']['avg_accuracy']:.2f}% "
              f"± {best_avg_auc['test_patient_average']['std_accuracy']:.2f}%")
        print(f"F1: {best_avg_auc['test_patient_average']['avg_f1']:.4f} "
              f"± {best_avg_auc['test_patient_average']['std_f1']:.4f}")
        
        print("\n--- Best by F1 Score ---")
        print(f"Config: {best_avg_f1['config_name']}")
        print(f"Hyperparameters: {best_avg_f1['hyperparameters']}")
        print(f"F1: {best_avg_f1['test_patient_average']['avg_f1']:.4f} "
              f"± {best_avg_f1['test_patient_average']['std_f1']:.4f}")
        print(f"Accuracy: {best_avg_f1['test_patient_average']['avg_accuracy']:.2f}% "
              f"± {best_avg_f1['test_patient_average']['std_accuracy']:.2f}%")
        print(f"AUC: {best_avg_f1['test_patient_average']['avg_auc']:.4f} "
              f"± {best_avg_f1['test_patient_average']['std_auc']:.4f}")
        
        print("\n" + "="*80)
        print("BEST CONFIGURATIONS - RECORDING-LEVEL")
        print("="*80)
        
        print("\n--- Best by Accuracy ---")
        print(f"Config: {best_rec_accuracy['config_name']}")
        print(f"Hyperparameters: {best_rec_accuracy['hyperparameters']}")
        print(f"Accuracy: {best_rec_accuracy['test_recording_level']['avg_accuracy']:.2f}% "
              f"± {best_rec_accuracy['test_recording_level']['std_accuracy']:.2f}%")
        print(f"AUC: {best_rec_accuracy['test_recording_level']['avg_auc']:.4f} "
              f"± {best_rec_accuracy['test_recording_level']['std_auc']:.4f}")
        print(f"F1: {best_rec_accuracy['test_recording_level']['avg_f1']:.4f} "
              f"± {best_rec_accuracy['test_recording_level']['std_f1']:.4f}")
        
        print("\n--- Best by AUC ---")
        print(f"Config: {best_rec_auc['config_name']}")
        print(f"Hyperparameters: {best_rec_auc['hyperparameters']}")
        print(f"AUC: {best_rec_auc['test_recording_level']['avg_auc']:.4f} "
              f"± {best_rec_auc['test_recording_level']['std_auc']:.4f}")
        print(f"Accuracy: {best_rec_auc['test_recording_level']['avg_accuracy']:.2f}% "
              f"± {best_rec_auc['test_recording_level']['std_accuracy']:.2f}%")
        print(f"F1: {best_rec_auc['test_recording_level']['avg_f1']:.4f} "
              f"± {best_rec_auc['test_recording_level']['std_f1']:.4f}")
        
        print("\n--- Best by F1 Score ---")
        print(f"Config: {best_rec_f1['config_name']}")
        print(f"Hyperparameters: {best_rec_f1['hyperparameters']}")
        print(f"F1: {best_rec_f1['test_recording_level']['avg_f1']:.4f} "
              f"± {best_rec_f1['test_recording_level']['std_f1']:.4f}")
        print(f"Accuracy: {best_rec_f1['test_recording_level']['avg_accuracy']:.2f}% "
              f"± {best_rec_f1['test_recording_level']['std_accuracy']:.2f}%")
        print(f"AUC: {best_rec_f1['test_recording_level']['avg_auc']:.4f} "
              f"± {best_rec_f1['test_recording_level']['std_auc']:.4f}")
        
        print(f"\nResults saved to: {final_path}")
        print("="*80)
        
        # Save best configurations summary
        best_configs = {
            'patient_majority_voting': {
                'best_by_accuracy': best_maj_accuracy,
                'best_by_auc': best_maj_auc,
                'best_by_f1': best_maj_f1
            },
            'patient_average_probability': {
                'best_by_accuracy': best_avg_accuracy,
                'best_by_auc': best_avg_auc,
                'best_by_f1': best_avg_f1
            },
            'recording_level': {
                'best_by_accuracy': best_rec_accuracy,
                'best_by_auc': best_rec_auc,
                'best_by_f1': best_rec_f1
            }
        }
        best_path = os.path.join(BASE_OUTPUT_DIR, 'best_configurations.json')
        with open(best_path, 'w') as f:
            json.dump(best_configs, f, indent=2)
        
    return all_results


if __name__ == "__main__":
    results = run_hyperparameter_search()
