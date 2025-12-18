import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

class BranchMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64, hidden_dims: list = None, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        
        # Build MLP layers
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer (two nodes for binary classification with CrossEntropyLoss)
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.mlp(x)
    
class IntermediateFusionMLP(nn.Module):
    def __init__(self, hidden_dims: list, dropout: float = 0.5, 
                 audio_branch: dict = None, 
                 tapping_branch: dict = None):
        super().__init__()
        
        # Set default values if not provided
        if audio_branch is None:
            audio_branch = {"input_dim": 128, "output_dim": 64, "hidden_dims": [256, 128]}
        if tapping_branch is None:
            tapping_branch = {"input_dim": 64, "output_dim": 64, "hidden_dims": [128, 64]}
        
        # Initialize branches
        self.audio_branch = BranchMLP(
            input_dim=audio_branch["input_dim"], 
            output_dim=audio_branch["output_dim"],
            hidden_dims=audio_branch["hidden_dims"], 
            dropout=dropout
        )
        
        self.tapping_branch = BranchMLP(
            input_dim=tapping_branch["input_dim"], 
            output_dim=tapping_branch["output_dim"],
            hidden_dims=tapping_branch["hidden_dims"], 
            dropout=dropout
        )
        
        # Fusion MLP
        fusion_input_dim = audio_branch["output_dim"] + tapping_branch["output_dim"]
        
        fusion_layers = []
        in_dim = fusion_input_dim
        
        for hidden_dim in hidden_dims:
            fusion_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        fusion_layers.append(nn.Linear(in_dim, 2))  # Output layer
        
        self.fusion_mlp = nn.Sequential(*fusion_layers)

        # Training history (populated by fit)
        self.train_loss_history = []
        self.train_acc_history = []
        
        # Test metrics (populated by test)
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None
        self.test_loss = None
        self.test_acc = None
        self.test_auc = None
        self.test_f1 = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, audio_x, tapping_x):
        audio_out = self.audio_branch(audio_x)
        tapping_out = self.tapping_branch(tapping_x)
        
        # Concatenate branch outputs
        fused_input = torch.cat((audio_out, tapping_out), dim=1)
        
        # Pass through fusion MLP
        output = self.fusion_mlp(fused_input)
        
        return output

    def predict(self, audio_features: torch.Tensor, tapping_features: torch.Tensor) -> torch.Tensor:
        """Get class predictions (0 or 1)."""
        self.eval()
        output = self.forward(audio_features, tapping_features)
        return torch.argmax(output, dim=1)
    
    def predict_proba(self, audio_features: torch.Tensor, tapping_features: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        self.eval()
        output = self.forward(audio_features, tapping_features)
        probs = torch.softmax(output, dim=1)
        return probs
    
    def fit(self, train_audio_features: torch.Tensor, train_tapping_features: torch.Tensor, 
            train_labels: torch.Tensor,
            val_audio_features: Optional[torch.Tensor] = None, 
            val_tapping_features: Optional[torch.Tensor] = None,
            val_labels: Optional[torch.Tensor] = None,
            epochs: int = 50, batch_size: int = 32, class_weight: float = 2.0, 
            lr: float = 1e-3, weight_decay: float = 0, verbose: bool = True):
        """
        Fit the model on training data with early stopping and learning rate reduction based on ROC AUC.
        
        Args:
            train_audio_features: Training audio features (N, audio_input_dim)
            train_tapping_features: Training tapping features (N, tapping_input_dim)
            train_labels: Training labels (N,)
            val_audio_features: Validation audio features (N_val, audio_input_dim) (optional)
            val_tapping_features: Validation tapping features (N_val, tapping_input_dim) (optional)
            val_labels: Validation labels (N_val,) (optional)
            epochs: Maximum number of training epochs (default: 50)
            batch_size: Batch size (default: 32)
            class_weight: Weight for class 0 (controls) to handle imbalance (default: 2)
            lr: Initial learning rate (default: 1e-3)
            weight_decay: L2 penalization coefficient (default: 0)
            verbose: Whether the method should output verbal execution tracing (default: True)
            
        Returns:
            val_loss_history: List of validation losses per epoch (if validation data provided), else None
            
        Note:
            - Early stopping: Training stops if validation ROC AUC doesn't increase for 40 consecutive epochs
            - LR scheduling: Learning rate is halved if validation ROC AUC doesn't increase for 10 consecutive epochs
        """
        # Reset history
        self.train_loss_history = []
        self.train_acc_history = []
        val_loss_history = []
        
        # Create data loader
        dataset = TensorDataset(train_audio_features, train_tapping_features, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        class_weights = torch.FloatTensor([class_weight, 1.0]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Early stopping and LR scheduling parameters based on ROC AUC
        best_val_auc = -float('inf')
        epochs_no_improve = 0
        epochs_no_improve_lr = 0
        early_stopping_patience = 40
        lr_reduction_patience = 10
        current_lr = lr
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            all_preds = []
            all_labels = []
            
            for audio_feat, tapping_feat, lbl in loader:
                audio_feat = audio_feat.to(self.device)
                tapping_feat = tapping_feat.to(self.device)
                lbl = lbl.to(self.device)
                
                optimizer.zero_grad()
                logits = self.forward(audio_feat, tapping_feat)
                loss = criterion(logits, lbl)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(lbl)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())
            
            epoch_loss /= len(dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_acc)
            
            # Validation at the end of each epoch
            if val_audio_features is not None and val_tapping_features is not None and val_labels is not None:
                self.eval()
                with torch.no_grad():
                    val_audio_device = val_audio_features.to(self.device)
                    val_tapping_device = val_tapping_features.to(self.device)
                    val_labels_device = val_labels.to(self.device)
                    
                    val_logits = self.forward(val_audio_device, val_tapping_device)
                    val_loss = criterion(val_logits, val_labels_device).item()
                    val_loss_history.append(val_loss)
                    
                    # Calculate ROC AUC for early stopping and LR scheduling
                    val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
                    val_labels_np = val_labels.cpu().numpy()
                    val_auc = roc_auc_score(val_labels_np, val_probs)
                    
                    # Check for improvement based on ROC AUC
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        epochs_no_improve = 0
                        epochs_no_improve_lr = 0
                    else:
                        epochs_no_improve += 1
                        epochs_no_improve_lr += 1
                        
                        # Learning rate reduction
                        if epochs_no_improve_lr >= lr_reduction_patience:
                            current_lr = current_lr / 2
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = current_lr
                            if verbose:
                                print(f"Learning rate reduced to {current_lr:.6f}")
                            epochs_no_improve_lr = 0
                        
                        # Early stopping
                        if epochs_no_improve >= early_stopping_patience:
                            if verbose:
                                print(f"Early stopping triggered at epoch {epoch+1}. "
                                      f"No improvement in ROC AUC for {early_stopping_patience} epochs.")
                            break
                    
                    if verbose and (epoch + 1) % 10 == 0:
                        val_preds = self.predict(val_audio_device, val_tapping_device).cpu().numpy()
                        val_acc = accuracy_score(val_labels.cpu().numpy(), val_preds)
                        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
                              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.6f} | "
                              f"Best AUC: {best_val_auc:.4f} | No improve: {epochs_no_improve}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        if verbose:
            print(f"Training complete. Final - Loss: {self.train_loss_history[-1]:.4f}, "
                  f"Acc: {self.train_acc_history[-1]:.4f}")
        
        return val_loss_history if val_audio_features is not None else None

    def test_with_majority_counting(self, audio_df, tapping_df, labels_df):
        """
        Evaluate the model on test data with majority voting aggregation per patient.
        
        This method handles multiple trials per patient by applying majority voting to get
        patient-level predictions and metrics.
        
        Args:
            audio_df: Pandas DataFrame containing audio feature columns plus 'healthCode'.
                     Each row represents one trial. Multiple trials may exist per patient (healthCode).
            tapping_df: Pandas DataFrame containing tapping feature columns plus 'healthCode'.
                       Each row represents one trial. Must have same length and healthCode ordering as audio_df.
            labels_df: Pandas DataFrame containing 'healthCode' and 'label_PD' columns.
                      One row per unique patient with their ground truth label.
                    
        Returns:
            Dictionary containing:
                - test_acc: Accuracy at patient level (after majority voting)
                - test_f1: F1 score at patient level
                - test_conf_mat: Confusion matrix at patient level
                - test_roc_auc: ROC AUC score at patient level
                - test_roc_curve: ROC curve (FPR, TPR, thresholds) at patient level
                - patient_predictions: DataFrame with per-patient aggregated predictions
                - num_patients: Total number of unique patients
                - num_trials: Total number of trials
                
        Note:
            The 'healthCode' column is used to match trials with patient labels.
            Majority voting is applied across all trials for each healthCode to determine the
            final patient-level prediction. Instance variables (TP, TN, FP, FN, test_acc, 
            test_f1, test_auc) are updated with patient-level metrics.
        """
        
        # Make copies to avoid modifying the original dataframes
        audio_df_copy = audio_df.copy()
        tapping_df_copy = tapping_df.copy()
        labels_df_copy = labels_df.copy()
        
        # Extract healthCode from audio_df
        healthcodes = audio_df_copy['healthCode'].values
        
        # Create a mapping from healthCode to label_PD
        label_map = dict(zip(labels_df_copy['healthCode'], labels_df_copy['label_PD']))
        
        # Map healthcodes to their true labels
        true_labels = np.array([label_map[hc] for hc in healthcodes])
        
        # Drop healthCode, label_PD, and any other metadata columns to get only numeric features
        # Metadata columns commonly include: filename, record_id, healthCode, label_PD
        metadata_columns = ['healthCode', 'label_PD', 'filename', 'record_id', 'healthcode', 'row_id', 'trial_id',
                          'filename_file1', 'filename_file2', 'record_id_file1', 'record_id_file2']
        
        audio_feature_columns = [col for col in audio_df_copy.columns if col not in metadata_columns]
        tapping_feature_columns = [col for col in tapping_df_copy.columns if col not in metadata_columns]
        
        audio_features = audio_df_copy[audio_feature_columns].values.astype(np.float32)
        tapping_features = tapping_df_copy[tapping_feature_columns].values.astype(np.float32)
        
        # Convert to torch tensors
        audio_features_tensor = torch.FloatTensor(audio_features).to(self.device)
        tapping_features_tensor = torch.FloatTensor(tapping_features).to(self.device)
        
        # Get probabilities for each trial
        self.eval()
        with torch.no_grad():
            probs = self.predict_proba(audio_features_tensor, tapping_features_tensor).cpu().numpy()
        
        # Get predicted class for each trial (0 or 1)
        trial_predictions = np.argmax(probs, axis=1)
        
        # Create a dataframe with healthCode, true_label, and predictions
        results_df = pd.DataFrame({
            'healthCode': healthcodes,
            'label_PD': true_labels,
            'prediction': trial_predictions,
            'prob_class_0': probs[:, 0],
            'prob_class_1': probs[:, 1]
        })
        
        # Group by healthCode and apply majority voting
        # For each patient, take the most common prediction across all trials
        patient_predictions = results_df.groupby('healthCode').agg({
            'prediction': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Majority vote
            'label_PD': 'first',  # Ground truth (same for all trials of a patient)
            'prob_class_0': 'mean',  # Average probability across trials
            'prob_class_1': 'mean'
        }).reset_index()
        
        # Extract aggregated predictions and true labels
        y_true = patient_predictions['label_PD'].values.astype(int)
        y_pred = patient_predictions['prediction'].values.astype(int)
        y_probs_class1 = patient_predictions['prob_class_1'].values
        
        # Calculate metrics
        conf_mat = confusion_matrix(y_true, y_pred)
        
        result_metrics = {
            "test_acc": accuracy_score(y_true, y_pred),
            "test_f1": f1_score(y_true, y_pred),
            "test_conf_mat": conf_mat,
            "test_roc_auc": roc_auc_score(y_true, y_probs_class1),
            "test_roc_curve": roc_curve(y_true, y_probs_class1),
            "patient_predictions": patient_predictions,  # Include detailed results
            "num_patients": len(patient_predictions),
            "num_trials": len(audio_df_copy)
        }
        
        # Store confusion matrix components
        self.TN, self.FP, self.FN, self.TP = conf_mat.ravel()
        self.test_acc = result_metrics["test_acc"]
        self.test_f1 = result_metrics["test_f1"]
        self.test_auc = result_metrics["test_roc_auc"]
        
        return result_metrics


    def test_by_averaging(self, audio_df, tapping_df, labels_df, fold_column='fold'):
        """
        Evaluate the model by averaging class 1 probabilities per healthCode and computing ROC AUC per fold.
        
        This method:
        1. Predicts probabilities for all trials
        2. Averages class 1 probabilities per healthCode
        3. Computes ROC AUC per fold
        4. Returns mean and standard deviation of ROC AUC across folds
        
        Args:
            audio_df: Pandas DataFrame containing audio feature columns plus 'healthCode' and optionally fold_column.
                     Each row represents one trial. Multiple trials may exist per patient (healthCode).
            tapping_df: Pandas DataFrame containing tapping feature columns plus 'healthCode' and optionally fold_column.
                       Each row represents one trial. Must have same length and healthCode ordering as audio_df.
            labels_df: Pandas DataFrame containing 'healthCode' and 'label_PD' columns.
                      One row per unique patient with their ground truth label.
            fold_column: Name of the column containing fold assignments (default: 'fold')
                    
        Returns:
            Dictionary containing:
                - mean_roc_auc: Mean ROC AUC across folds
                - std_roc_auc: Standard deviation of ROC AUC across folds
                - fold_roc_aucs: List of ROC AUC scores per fold
                - averaged_predictions: DataFrame with averaged predictions per healthCode
        """
        # Make copies to avoid modifying the original dataframes
        audio_df_copy = audio_df.copy()
        tapping_df_copy = tapping_df.copy()
        labels_df_copy = labels_df.copy()
        
        # Extract healthCode from audio_df
        healthcodes = audio_df_copy['healthCode'].values
        folds = audio_df_copy[fold_column].values if fold_column in audio_df_copy.columns else None
        
        # Create a mapping from healthCode to label_PD
        label_map = dict(zip(labels_df_copy['healthCode'], labels_df_copy['label_PD']))
        
        # Map healthcodes to their true labels
        true_labels = np.array([label_map[hc] for hc in healthcodes])
        
        # Drop metadata columns to get only numeric features
        metadata_columns = ['healthCode', 'label_PD', 'filename', 'record_id', 'healthcode', 'row_id', 'trial_id',
                          'filename_file1', 'filename_file2', 'record_id_file1', 'record_id_file2',
                          fold_column, 'fold']
        
        audio_feature_columns = [col for col in audio_df_copy.columns if col not in metadata_columns]
        tapping_feature_columns = [col for col in tapping_df_copy.columns if col not in metadata_columns]
        
        audio_features = audio_df_copy[audio_feature_columns].values.astype(np.float32)
        tapping_features = tapping_df_copy[tapping_feature_columns].values.astype(np.float32)
        
        # Convert to torch tensors
        audio_features_tensor = torch.FloatTensor(audio_features).to(self.device)
        tapping_features_tensor = torch.FloatTensor(tapping_features).to(self.device)
        
        # Get probabilities for each trial
        self.eval()
        with torch.no_grad():
            probs = self.predict_proba(audio_features_tensor, tapping_features_tensor).cpu().numpy()
        
        # Create a dataframe with healthCode, true_label, probabilities, and fold
        results_dict = {
            'healthCode': healthcodes,
            'label_PD': true_labels,
            'prob_class_1': probs[:, 1]
        }
        
        if folds is not None:
            results_dict['fold'] = folds
        
        results_df = pd.DataFrame(results_dict)
        
        # Group by healthCode (and fold if present) and average probabilities
        if folds is not None:
            averaged_predictions = results_df.groupby(['healthCode', 'fold']).agg({
                'prob_class_1': 'mean',
                'label_PD': 'first'
            }).reset_index()
            
            # Compute ROC AUC per fold
            fold_roc_aucs = []
            unique_folds = sorted(averaged_predictions['fold'].unique())
            
            for fold in unique_folds:
                fold_data = averaged_predictions[averaged_predictions['fold'] == fold]
                if len(fold_data) > 0 and len(fold_data['label_PD'].unique()) > 1:
                    fold_auc = roc_auc_score(fold_data['label_PD'], fold_data['prob_class_1'])
                    fold_roc_aucs.append(fold_auc)
            
            mean_roc_auc = np.mean(fold_roc_aucs)
            std_roc_auc = np.std(fold_roc_aucs)
        else:
            # No fold information, compute overall ROC AUC
            averaged_predictions = results_df.groupby('healthCode').agg({
                'prob_class_1': 'mean',
                'label_PD': 'first'
            }).reset_index()
            
            overall_auc = roc_auc_score(averaged_predictions['label_PD'], averaged_predictions['prob_class_1'])
            fold_roc_aucs = [overall_auc]
            mean_roc_auc = overall_auc
            std_roc_auc = 0.0
        
        return {
            'mean_roc_auc': mean_roc_auc,
            'std_roc_auc': std_roc_auc,
            'fold_roc_aucs': fold_roc_aucs,
            'averaged_predictions': averaged_predictions
        }

    def test(self, test_audio_features: torch.Tensor, test_tapping_features: torch.Tensor, 
             test_labels: torch.Tensor):
        """
        Evaluate the model on test data and return a dictionary of metrics.
        
        Args:
            test_audio_features: Test audio features of shape (N, audio_input_dim)
            test_tapping_features: Test tapping features of shape (N, tapping_input_dim)
            test_labels: Ground truth labels of shape (N,)
        
        Returns:
            Dictionary containing test metrics (loss, accuracy, F1, confusion matrix, ROC AUC, ROC curve)
        """
        self.eval()
        
        # Move tensors to the same device as the model
        test_audio_features = test_audio_features.to(self.device)
        test_tapping_features = test_tapping_features.to(self.device)
        test_labels = test_labels.to(self.device)
        
        with torch.no_grad():
            output = self.forward(test_audio_features, test_tapping_features)
            y_true = test_labels.cpu().numpy()
            y_pred = self.predict(test_audio_features, test_tapping_features).cpu().numpy()

        result_metrics = {
            "test_loss": nn.CrossEntropyLoss()(output, test_labels).item(),
            "test_acc": accuracy_score(y_true, y_pred),
            "test_f1": f1_score(y_true, y_pred),
            "test_conf_mat": confusion_matrix(y_true, y_pred),
            "test_roc_auc": roc_auc_score(y_true, torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()),
            "test_roc_curve": roc_curve(y_true, torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        }

        return result_metrics