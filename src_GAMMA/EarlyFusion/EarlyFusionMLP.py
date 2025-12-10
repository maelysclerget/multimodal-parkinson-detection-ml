import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


class EarlyFusionMLP(nn.Module):
    def __init__(self, 
                 input_dim: int = 1280,
                 hidden_dims: list = [512, 256],
                 dropout: float = 0.3,
                 verbose: bool = True):
        """
        Early fusion MLP model that takes a single concatenated feature vector,
        and passes it through an MLP classifier.
        
        Args:
            input_dim: Dimension of input features (default: 1280)
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
            dropout: Dropout probability (default: 0.3)
            verbose: Whether the constructor should output verbal execution tracing (default: True)
        """
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
        layers.append(nn.Linear(in_dim, 2))
        
        self.mlp = nn.Sequential(*layers)

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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        if verbose:
            print(f"Input dimension: {self.input_dim}")
            print(f"MLP architecture: {self.input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 2 (binary classification)")
            print(f"Device: {self.device}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, 2)
        """
        # Pass through MLP
        forward_output = self.mlp(features)

        return forward_output

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Get class predictions (0 or 1)."""
        self.eval()
        output = self.forward(features)
        return torch.argmax(output, dim=1)
    
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        self.eval()
        output = self.forward(features)
        probs = torch.softmax(output, dim=1)
        return probs
    
    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor,
            val_features: Optional[torch.Tensor] = None, val_labels: Optional[torch.Tensor] = None,
            epochs: int = 50, batch_size: int = 32, class_weight: float = 2.0, 
            lr: float = 1e-3, weight_decay: float = 0, verbose: bool = True):
        """
        Fit the model on training data with early stopping and learning rate reduction.
        
        Args:
            train_features: Training features (N, input_dim)
            train_labels: Training labels (N,)
            val_features: Validation features (N_val, input_dim) (optional)
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
            - Early stopping: Training stops if validation loss doesn't decrease for 20 consecutive epochs
            - LR scheduling: Learning rate is halved if validation loss doesn't decrease for 10 consecutive epochs
        """
        # Reset history
        self.train_loss_history = []
        self.train_acc_history = []
        val_loss_history = []
        
        # Create data loader
        dataset = TensorDataset(train_features, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        class_weights = torch.FloatTensor([class_weight, 1.0]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Early stopping and LR scheduling parameters
        best_val_loss = float('inf')
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
            
            for feat, lbl in loader:
                feat, lbl = feat.to(self.device), lbl.to(self.device)
                
                optimizer.zero_grad()
                logits = self.forward(feat)
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
            if val_features is not None and val_labels is not None:
                self.eval()
                with torch.no_grad():
                    val_features_device = val_features.to(self.device)
                    val_labels_device = val_labels.to(self.device)
                    
                    val_logits = self.forward(val_features_device)
                    val_loss = criterion(val_logits, val_labels_device).item()
                    val_loss_history.append(val_loss)
                    
                    # Check for improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
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
                                      f"No improvement for {early_stopping_patience} epochs.")
                            break
                    
                    if verbose and (epoch + 1) % 10 == 0:
                        val_preds = self.predict(val_features_device).cpu().numpy()
                        val_acc = accuracy_score(val_labels.cpu().numpy(), val_preds)
                        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
                              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        if verbose:
            print(f"Training complete. Final - Loss: {self.train_loss_history[-1]:.4f}, "
                  f"Acc: {self.train_acc_history[-1]:.4f}")
        
        return val_loss_history if val_features is not None else None

    def test_with_majority_counting(self, test_df):
        """
        Evaluate the model on test data with majority voting aggregation per patient.
        
        This method handles multiple trials per patient by applying majority voting to get
        patient-level predictions and metrics.
        
        Args:
            test_df: Pandas DataFrame containing feature columns plus 'healthCode' and 'label_PD'.
                    Each row represents one trial. Multiple trials may exist per patient (healthCode).
                    
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
            The 'healthCode' and 'label_PD' columns are excluded from features during forward pass.
            Majority voting is applied across all trials for each healthCode to determine the
            final patient-level prediction. Instance variables (TP, TN, FP, FN, test_acc, 
            test_f1, test_auc) are updated with patient-level metrics.
        """
        import pandas as pd
        import numpy as np
        
        # Make a copy to avoid modifying the original dataframe
        test_df_copy = test_df.copy()
        
        # Extract healthCode and label_PD before dropping them
        healthcodes = test_df_copy['healthCode'].values
        true_labels = test_df_copy['label_PD'].values
        
        # Drop healthCode, label_PD, and any other metadata columns to get only numeric features
        # Metadata columns commonly include: filename, record_id, healthCode, label_PD
        metadata_columns = ['healthCode', 'label_PD', 'filename', 'record_id', 'healthcode']
        feature_columns = [col for col in test_df_copy.columns if col not in metadata_columns]
        test_features = test_df_copy[feature_columns].values
        
        # Ensure features are float32 type
        test_features = test_features.astype(np.float32)
        
        # Convert to torch tensor
        test_features_tensor = torch.FloatTensor(test_features).to(self.device)
        
        # Get probabilities for each trial
        self.eval()
        with torch.no_grad():
            probs = self.predict_proba(test_features_tensor).cpu().numpy()
        
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
            "num_trials": len(test_df_copy)
        }
        
        # Store confusion matrix components
        self.TN, self.FP, self.FN, self.TP = conf_mat.ravel()
        self.test_acc = result_metrics["test_acc"]
        self.test_f1 = result_metrics["test_f1"]
        self.test_auc = result_metrics["test_roc_auc"]
        
        return result_metrics


    def test(self, test_features: torch.Tensor, test_labels: torch.Tensor):
        """
        Evaluate the model on test data and return a dictionary of metrics.
        
        Args:
            test_features: Test features of shape (N, input_dim)
            test_labels: Ground truth labels of shape (N,)
        
        Returns:
            Dictionary containing test metrics (loss, accuracy, F1, confusion matrix, ROC AUC, ROC curve)
        """
        self.eval()
        
        # Move tensors to the same device as the model
        test_features = test_features.to(self.device)
        test_labels = test_labels.to(self.device)
        
        with torch.no_grad():
            output = self.forward(test_features)
            y_true = test_labels.cpu().numpy()
            y_pred = self.predict(test_features).cpu().numpy()

        result_metrics = {
            "test_loss": nn.CrossEntropyLoss()(output, test_labels).item(),
            "test_acc": accuracy_score(y_true, y_pred),
            "test_f1": f1_score(y_true, y_pred),
            "test_conf_mat": confusion_matrix(y_true, y_pred),
            "test_roc_auc": roc_auc_score(y_true, torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()),
            "test_roc_curve": roc_curve(y_true, torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        }

        return result_metrics
