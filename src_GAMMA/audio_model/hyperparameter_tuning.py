"""
Hyperparameter tuning for MLP acoustic feature classifier
Tests different combinations of learning rate, hidden layer sizes, dropout, and batch size
"""

import os
import json
import itertools
from mlp_features_model import train_5fold_cv

# Hyperparameter search space - V6 with class weights (NO undersampling)
# Data is ~77% PD / 23% Control at recording level
# Using class weights to handle imbalance while keeping natural distribution
HYPERPARAMETER_GRID = {
    'learning_rate': [0.001],  # Best from previous results
    'hidden_sizes': [
        [128, 64],             # 2 hidden layers (simpler, less overfitting)
        [256, 128, 64],        # 3 hidden layers (Best from V7)
    ],
    'dropout': [0.7, 0.8, 0.9],  # Higher dropout to combat overfitting
    'batch_size': [64],  # Fixed batch size
    'class_weight': [3, 3.5],  # Best from V7 (optimal for 77/23 imbalance)
    'undersample': [False],  # NO undersampling - use class weights instead
}

# Fixed parameters
FEATURES_CSV = "/mloscratch/users/gnahas/data/features/acoustic_features_vf.csv"
LABELS_CSV = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
TRAIN_FOLDS_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
VAL_TEST_FOLDS_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"
BASE_OUTPUT_DIR = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/MLP_Tuning/V9_wo_preprocessing"
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
        config_name = f"lr{params['learning_rate']}_hs{'-'.join(map(str, params['hidden_sizes']))}_drop{params['dropout']}_bs{params['batch_size']}_cw{params['class_weight']}_us{params['undersample']}"
        
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
                dropout=params['dropout'],
                class_weight=params['class_weight'],
                undersample=params['undersample']
            )
            
            # Store results with hyperparameters
            result = {
                'config_id': idx + 1,
                'config_name': config_name,
                'hyperparameters': params,
                'test_patient_threshold_tuned': summary['test_patient_threshold_tuned'],
                'test_patient_majority': summary['test_patient_majority'],
                'test_patient_average': summary['test_patient_average'],
                'test_recording_level': summary['test_recording_level'],
            }
            all_results.append(result)
            
            print(f"\n✓ Configuration {idx+1} completed successfully")
            print(f"  Patient (Threshold-Tuned) - Acc: {summary['test_patient_threshold_tuned']['avg_accuracy']:.2f}%, "
                  f"AUC: {summary['test_patient_threshold_tuned']['avg_auc']:.4f}, "
                  f"F1: {summary['test_patient_threshold_tuned']['avg_f1']:.4f}, "
                  f"Sens: {summary['test_patient_threshold_tuned']['avg_sensitivity']:.3f}, "
                  f"Spec: {summary['test_patient_threshold_tuned']['avg_specificity']:.3f}")
            print(f"  Patient (Majority)        - Acc: {summary['test_patient_majority']['avg_accuracy']:.2f}%, "
                  f"AUC: {summary['test_patient_majority']['avg_auc']:.4f}, "
                  f"Sens: {summary['test_patient_majority']['avg_sensitivity']:.3f}, "
                  f"Spec: {summary['test_patient_majority']['avg_specificity']:.3f}")
            print(f"  Recording-level           - Acc: {summary['test_recording_level']['avg_accuracy']:.2f}%, "
                  f"AUC: {summary['test_recording_level']['avg_auc']:.4f}, "
                  f"Sens: {summary['test_recording_level']['avg_sensitivity']:.3f}, "
                  f"Spec: {summary['test_recording_level']['avg_specificity']:.3f}")
            
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
        # Print summary table of ALL configurations
        print("\n" + "="*80)
        print("SUMMARY TABLE - ALL CONFIGURATIONS")
        print("="*80)
        print(f"\n{'ID':<4} {'Config Name':<45} {'AUC':<7} {'Sens':<7} {'Spec':<7} {'F1':<7}")
        print("-" * 80)
        for r in successful_results:
            config_id = r['config_id']
            config_name = r['config_name'][:45]  # Truncate if too long
            auc = r['test_patient_threshold_tuned']['avg_auc']
            sens = r['test_patient_threshold_tuned']['avg_sensitivity']
            spec = r['test_patient_threshold_tuned']['avg_specificity']
            f1 = r['test_patient_threshold_tuned']['avg_f1']
            print(f"{config_id:<4} {config_name:<45} {auc:<7.4f} {sens:<7.3f} {spec:<7.3f} {f1:<7.4f}")
        print("="*80)
        

        # Best by patient-level threshold-tuned (PRIMARY)
        best_thr_accuracy = max(successful_results, 
                               key=lambda x: x['test_patient_threshold_tuned']['avg_accuracy'])
        best_thr_auc = max(successful_results, 
                         key=lambda x: x['test_patient_threshold_tuned']['avg_auc'])
        best_thr_f1 = max(successful_results, 
                        key=lambda x: x['test_patient_threshold_tuned']['avg_f1'])
        
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
        print("BEST CONFIGURATIONS - PATIENT-LEVEL (THRESHOLD-TUNED) [PRIMARY]")
        print("="*80)
        
        print("\n--- Best by Accuracy ---")
        print(f"Config: {best_thr_accuracy['config_name']}")
        print(f"Hyperparameters: {best_thr_accuracy['hyperparameters']}")
        print(f"Accuracy: {best_thr_accuracy['test_patient_threshold_tuned']['avg_accuracy']:.2f}% "
              f"± {best_thr_accuracy['test_patient_threshold_tuned']['std_accuracy']:.2f}%")
        print(f"AUC: {best_thr_accuracy['test_patient_threshold_tuned']['avg_auc']:.4f} "
              f"± {best_thr_accuracy['test_patient_threshold_tuned']['std_auc']:.4f}")
        print(f"F1: {best_thr_accuracy['test_patient_threshold_tuned']['avg_f1']:.4f} "
              f"± {best_thr_accuracy['test_patient_threshold_tuned']['std_f1']:.4f}")
        print(f"Sensitivity: {best_thr_accuracy['test_patient_threshold_tuned']['avg_sensitivity']:.4f} "
              f"± {best_thr_accuracy['test_patient_threshold_tuned']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_thr_accuracy['test_patient_threshold_tuned']['avg_specificity']:.4f} "
              f"± {best_thr_accuracy['test_patient_threshold_tuned']['std_specificity']:.4f}")
        
        print("\n--- Best by AUC ---")
        print(f"Config: {best_thr_auc['config_name']}")
        print(f"Hyperparameters: {best_thr_auc['hyperparameters']}")
        print(f"AUC: {best_thr_auc['test_patient_threshold_tuned']['avg_auc']:.4f} "
              f"± {best_thr_auc['test_patient_threshold_tuned']['std_auc']:.4f}")
        print(f"Accuracy: {best_thr_auc['test_patient_threshold_tuned']['avg_accuracy']:.2f}% "
              f"± {best_thr_auc['test_patient_threshold_tuned']['std_accuracy']:.2f}%")
        print(f"F1: {best_thr_auc['test_patient_threshold_tuned']['avg_f1']:.4f} "
              f"± {best_thr_auc['test_patient_threshold_tuned']['std_f1']:.4f}")
        print(f"Sensitivity: {best_thr_auc['test_patient_threshold_tuned']['avg_sensitivity']:.4f} "
              f"± {best_thr_auc['test_patient_threshold_tuned']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_thr_auc['test_patient_threshold_tuned']['avg_specificity']:.4f} "
              f"± {best_thr_auc['test_patient_threshold_tuned']['std_specificity']:.4f}")
        
        print("\n--- Best by F1 Score ---")
        print(f"Config: {best_thr_f1['config_name']}")
        print(f"Hyperparameters: {best_thr_f1['hyperparameters']}")
        print(f"F1: {best_thr_f1['test_patient_threshold_tuned']['avg_f1']:.4f} "
              f"± {best_thr_f1['test_patient_threshold_tuned']['std_f1']:.4f}")
        print(f"Accuracy: {best_thr_f1['test_patient_threshold_tuned']['avg_accuracy']:.2f}% "
              f"± {best_thr_f1['test_patient_threshold_tuned']['std_accuracy']:.2f}%")
        print(f"AUC: {best_thr_f1['test_patient_threshold_tuned']['avg_auc']:.4f} "
              f"± {best_thr_f1['test_patient_threshold_tuned']['std_auc']:.4f}")
        print(f"Sensitivity: {best_thr_f1['test_patient_threshold_tuned']['avg_sensitivity']:.4f} "
              f"± {best_thr_f1['test_patient_threshold_tuned']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_thr_f1['test_patient_threshold_tuned']['avg_specificity']:.4f} "
              f"± {best_thr_f1['test_patient_threshold_tuned']['std_specificity']:.4f}")
        
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
        print(f"Sensitivity: {best_maj_accuracy['test_patient_majority']['avg_sensitivity']:.4f} "
              f"± {best_maj_accuracy['test_patient_majority']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_maj_accuracy['test_patient_majority']['avg_specificity']:.4f} "
              f"± {best_maj_accuracy['test_patient_majority']['std_specificity']:.4f}")
        
        print("\n--- Best by AUC ---")
        print(f"Config: {best_maj_auc['config_name']}")
        print(f"Hyperparameters: {best_maj_auc['hyperparameters']}")
        print(f"AUC: {best_maj_auc['test_patient_majority']['avg_auc']:.4f} "
              f"± {best_maj_auc['test_patient_majority']['std_auc']:.4f}")
        print(f"Accuracy: {best_maj_auc['test_patient_majority']['avg_accuracy']:.2f}% "
              f"± {best_maj_auc['test_patient_majority']['std_accuracy']:.2f}%")
        print(f"F1: {best_maj_auc['test_patient_majority']['avg_f1']:.4f} "
              f"± {best_maj_auc['test_patient_majority']['std_f1']:.4f}")
        print(f"Sensitivity: {best_maj_auc['test_patient_majority']['avg_sensitivity']:.4f} "
              f"± {best_maj_auc['test_patient_majority']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_maj_auc['test_patient_majority']['avg_specificity']:.4f} "
              f"± {best_maj_auc['test_patient_majority']['std_specificity']:.4f}")
        
        print("\n--- Best by F1 Score ---")
        print(f"Config: {best_maj_f1['config_name']}")
        print(f"Hyperparameters: {best_maj_f1['hyperparameters']}")
        print(f"F1: {best_maj_f1['test_patient_majority']['avg_f1']:.4f} "
              f"± {best_maj_f1['test_patient_majority']['std_f1']:.4f}")
        print(f"Accuracy: {best_maj_f1['test_patient_majority']['avg_accuracy']:.2f}% "
              f"± {best_maj_f1['test_patient_majority']['std_accuracy']:.2f}%")
        print(f"AUC: {best_maj_f1['test_patient_majority']['avg_auc']:.4f} "
              f"± {best_maj_f1['test_patient_majority']['std_auc']:.4f}")
        print(f"Sensitivity: {best_maj_f1['test_patient_majority']['avg_sensitivity']:.4f} "
              f"± {best_maj_f1['test_patient_majority']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_maj_f1['test_patient_majority']['avg_specificity']:.4f} "
              f"± {best_maj_f1['test_patient_majority']['std_specificity']:.4f}")
        
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
        print(f"Sensitivity: {best_avg_accuracy['test_patient_average']['avg_sensitivity']:.4f} "
              f"± {best_avg_accuracy['test_patient_average']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_avg_accuracy['test_patient_average']['avg_specificity']:.4f} "
              f"± {best_avg_accuracy['test_patient_average']['std_specificity']:.4f}")
        
        print("\n--- Best by AUC ---")
        print(f"Config: {best_avg_auc['config_name']}")
        print(f"Hyperparameters: {best_avg_auc['hyperparameters']}")
        print(f"AUC: {best_avg_auc['test_patient_average']['avg_auc']:.4f} "
              f"± {best_avg_auc['test_patient_average']['std_auc']:.4f}")
        print(f"Accuracy: {best_avg_auc['test_patient_average']['avg_accuracy']:.2f}% "
              f"± {best_avg_auc['test_patient_average']['std_accuracy']:.2f}%")
        print(f"F1: {best_avg_auc['test_patient_average']['avg_f1']:.4f} "
              f"± {best_avg_auc['test_patient_average']['std_f1']:.4f}")
        print(f"Sensitivity: {best_avg_auc['test_patient_average']['avg_sensitivity']:.4f} "
              f"± {best_avg_auc['test_patient_average']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_avg_auc['test_patient_average']['avg_specificity']:.4f} "
              f"± {best_avg_auc['test_patient_average']['std_specificity']:.4f}")
        
        print("\n--- Best by F1 Score ---")
        print(f"Config: {best_avg_f1['config_name']}")
        print(f"Hyperparameters: {best_avg_f1['hyperparameters']}")
        print(f"F1: {best_avg_f1['test_patient_average']['avg_f1']:.4f} "
              f"± {best_avg_f1['test_patient_average']['std_f1']:.4f}")
        print(f"Accuracy: {best_avg_f1['test_patient_average']['avg_accuracy']:.2f}% "
              f"± {best_avg_f1['test_patient_average']['std_accuracy']:.2f}%")
        print(f"AUC: {best_avg_f1['test_patient_average']['avg_auc']:.4f} "
              f"± {best_avg_f1['test_patient_average']['std_auc']:.4f}")
        print(f"Sensitivity: {best_avg_f1['test_patient_average']['avg_sensitivity']:.4f} "
              f"± {best_avg_f1['test_patient_average']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_avg_f1['test_patient_average']['avg_specificity']:.4f} "
              f"± {best_avg_f1['test_patient_average']['std_specificity']:.4f}")
        
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
        print(f"Sensitivity: {best_rec_accuracy['test_recording_level']['avg_sensitivity']:.4f} "
              f"± {best_rec_accuracy['test_recording_level']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_rec_accuracy['test_recording_level']['avg_specificity']:.4f} "
              f"± {best_rec_accuracy['test_recording_level']['std_specificity']:.4f}")
        
        print("\n--- Best by AUC ---")
        print(f"Config: {best_rec_auc['config_name']}")
        print(f"Hyperparameters: {best_rec_auc['hyperparameters']}")
        print(f"AUC: {best_rec_auc['test_recording_level']['avg_auc']:.4f} "
              f"± {best_rec_auc['test_recording_level']['std_auc']:.4f}")
        print(f"Accuracy: {best_rec_auc['test_recording_level']['avg_accuracy']:.2f}% "
              f"± {best_rec_auc['test_recording_level']['std_accuracy']:.2f}%")
        print(f"F1: {best_rec_auc['test_recording_level']['avg_f1']:.4f} "
              f"± {best_rec_auc['test_recording_level']['std_f1']:.4f}")
        print(f"Sensitivity: {best_rec_auc['test_recording_level']['avg_sensitivity']:.4f} "
              f"± {best_rec_auc['test_recording_level']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_rec_auc['test_recording_level']['avg_specificity']:.4f} "
              f"± {best_rec_auc['test_recording_level']['std_specificity']:.4f}")
        
        print("\n--- Best by F1 Score ---")
        print(f"Config: {best_rec_f1['config_name']}")
        print(f"Hyperparameters: {best_rec_f1['hyperparameters']}")
        print(f"F1: {best_rec_f1['test_recording_level']['avg_f1']:.4f} "
              f"± {best_rec_f1['test_recording_level']['std_f1']:.4f}")
        print(f"Accuracy: {best_rec_f1['test_recording_level']['avg_accuracy']:.2f}% "
              f"± {best_rec_f1['test_recording_level']['std_accuracy']:.2f}%")
        print(f"AUC: {best_rec_f1['test_recording_level']['avg_auc']:.4f} "
              f"± {best_rec_f1['test_recording_level']['std_auc']:.4f}")
        print(f"Sensitivity: {best_rec_f1['test_recording_level']['avg_sensitivity']:.4f} "
              f"± {best_rec_f1['test_recording_level']['std_sensitivity']:.4f}")
        print(f"Specificity: {best_rec_f1['test_recording_level']['avg_specificity']:.4f} "
              f"± {best_rec_f1['test_recording_level']['std_specificity']:.4f}")
        
        print(f"\nResults saved to: {final_path}")
        print("="*80)
        
        # Save best configurations summary
        best_configs = {
            'patient_threshold_tuned': {
                'best_by_accuracy': best_thr_accuracy,
                'best_by_auc': best_thr_auc,
                'best_by_f1': best_thr_f1
            },
            'patient_majority_voting': {
                'best_by_accuracy': best_maj_accuracy,
                'best_by_auc': best_maj_auc,
                'best_by_f1': best_maj_f1
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
        
        # Quick diagnostic summary
        print("\n" + "="*80)
        print("QUICK DIAGNOSTIC SUMMARY - AVERAGE METRICS ACROSS ALL CONFIGURATIONS")
        print("="*80)
        
        avg_sensitivity_thr = sum(r['test_patient_threshold_tuned']['avg_sensitivity'] for r in successful_results) / len(successful_results)
        avg_specificity_thr = sum(r['test_patient_threshold_tuned']['avg_specificity'] for r in successful_results) / len(successful_results)
        avg_auc_thr = sum(r['test_patient_threshold_tuned']['avg_auc'] for r in successful_results) / len(successful_results)
        avg_accuracy_thr = sum(r['test_patient_threshold_tuned']['avg_accuracy'] for r in successful_results) / len(successful_results)
        avg_f1_thr = sum(r['test_patient_threshold_tuned']['avg_f1'] for r in successful_results) / len(successful_results)
        
        print(f"\nPatient-Level (Threshold-Tuned) - Averaged across {len(successful_results)} configs:")
        print(f"  Average Sensitivity: {avg_sensitivity_thr:.3f} ({avg_sensitivity_thr*100:.1f}%)")
        print(f"  Average Specificity: {avg_specificity_thr:.3f} ({avg_specificity_thr*100:.1f}%)")
        print(f"  Average AUC:         {avg_auc_thr:.4f}")
        print(f"  Average F1:          {avg_f1_thr:.4f}")
        print(f"  Average Accuracy:    {avg_accuracy_thr:.2f}%")
        
        # Breakdown by class weight
        print(f"\nBreakdown by Class Weight:")
        class_weights = sorted(set(r['hyperparameters']['class_weight'] for r in successful_results))
        for cw in class_weights:
            cw_results = [r for r in successful_results if r['hyperparameters']['class_weight'] == cw]
            cw_sens = sum(r['test_patient_threshold_tuned']['avg_sensitivity'] for r in cw_results) / len(cw_results)
            cw_spec = sum(r['test_patient_threshold_tuned']['avg_specificity'] for r in cw_results) / len(cw_results)
            cw_auc = sum(r['test_patient_threshold_tuned']['avg_auc'] for r in cw_results) / len(cw_results)
            cw_f1 = sum(r['test_patient_threshold_tuned']['avg_f1'] for r in cw_results) / len(cw_results)
            print(f"  Class Weight {cw}: Sens={cw_sens:.3f}, Spec={cw_spec:.3f}, AUC={cw_auc:.4f}, F1={cw_f1:.4f}")
        
        print("="*80)
        
    return all_results


if __name__ == "__main__":
    results = run_hyperparameter_search()
