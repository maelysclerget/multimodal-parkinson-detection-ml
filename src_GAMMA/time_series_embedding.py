import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from typing import List, Union
import numpy as np


class TimeSeriesEmbeddingExtractor:
    def __init__(self, model_name: str = 'facebook/wav2vec2-base', 
                 pooling: str = 'mean',
                 verbose: bool = True):
        """
        Initialize the time series embedding extractor with Wav2Vec2.
        
        Args:
            model_name: HuggingFace model name (default: 'facebook/wav2vec2-base')
            pooling: How to pool temporal dimension - 'mean', 'max', or 'cls' (default: 'mean')
            verbose: Whether the constructor should output verbal execution tracing (default: True)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pooling = pooling
        
        # Load Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        
        self.hidden_size = self.model.config.hidden_size  
        
        if verbose:
            print(f"Model: {model_name}")
            print(f"Device: {self.device}")
            print(f"Model hidden size: {self.hidden_size}")
            print(f"Pooling strategy: {pooling}")
    
    def preprocess(self, time_series: Union[np.ndarray, torch.Tensor, List],
                   sampling_rate: int = 16000) -> torch.Tensor:
        """
        Preprocess a single time series.
        
        Args:
            time_series: 1D array of the time series / audio waveform
            sampling_rate: Sampling rate in Hz (default: 16000)
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(time_series, list):
            time_series = torch.tensor(time_series, dtype=torch.float32)
        
        inputs = self.processor(
            time_series, 
            sampling_rate=sampling_rate, 
            return_tensors="pt",
            padding=True
        )

        return inputs.input_values.squeeze(0)
    
    def pool_embeddings(self, hidden_states: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool the temporal dimension of hidden states.
        
        Args:
            hidden_states: Tensor of shape (batch, time, hidden_size)
            attention_mask: Tensor of shape (batch, time) with 1s for valid positions
            
        Returns:
            Pooled tensor of shape (batch, hidden_size)
        """
        if self.pooling == 'mean':
            # Mask out padded positions and compute mean only over valid positions
            mask = attention_mask.unsqueeze(-1)  # (batch, time, 1)
            masked = hidden_states * mask
            return masked.sum(dim=1) / mask.sum(dim=1)
        elif self.pooling == 'max':
            # Set padded positions to large negative value before max
            mask = attention_mask.unsqueeze(-1)  # (batch, time, 1)
            masked = hidden_states.masked_fill(mask == 0, float('-inf'))
            return masked.max(dim=1).values
        elif self.pooling == 'cls':
            return hidden_states[:, 0, :]  # First token
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def extract_embeddings(self, time_series_list: List[Union[np.ndarray, torch.Tensor]],
                          sampling_rate: int = 16000,
                          batch_size: int = 8,
                          verbose: bool = True) -> torch.Tensor:
        """
        Extract embeddings for a list of time series.
        
        Args:
            time_series_list: List of 1D arrays (waveforms / time series)
            sampling_rate: Sampling rate in Hz (default: 16000)
            batch_size: Number of samples to process at once (default: 8)
            verbose: Whether the method should output verbal execution tracing (default: True)
            
        Returns:
            PyTorch tensor of shape (num_samples, embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(time_series_list), batch_size):
                batch_data = time_series_list[i:i + batch_size]
                
                # Preprocess batch
                batch_tensors = []
                for ts in batch_data:
                    try:
                        processed = self.preprocess(ts, sampling_rate)
                        batch_tensors.append(processed)
                    except Exception as e:
                        if verbose:
                            print(f"Error processing sample {i}: {e}")
                        continue
                
                if batch_tensors == []:
                    continue
                
                # Pad sequences to same length within batch
                max_len = max(t.shape[0] for t in batch_tensors)
                padded = torch.zeros(len(batch_tensors), max_len)
                attention_mask = torch.zeros(len(batch_tensors), max_len)

                for j, t in enumerate(batch_tensors):
                    padded[j, :t.shape[0]] = t
                    attention_mask[j, :t.shape[0]] = 1
                
                padded = padded.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Extract hidden states with attention mask
                outputs = self.model(padded, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # (batch, time, hidden_size)
                
                # Compute output attention mask (accounts for conv downsampling)
                output_lengths = self.model._get_feat_extract_output_lengths(
                    attention_mask.sum(dim=1).long()
                )
                output_mask = torch.zeros(hidden_states.shape[:2], device=self.device)
                for j, length in enumerate(output_lengths):
                    output_mask[j, :length] = 1
                
                # Pool temporal dimension with mask
                pooled = self.pool_embeddings(hidden_states, output_mask)
                
                embeddings.append(pooled.cpu())
                
                if verbose:
                    print(f"Processed {min(i + batch_size, len(time_series_list))}/{len(time_series_list)} samples")
        
        if embeddings:
            return torch.cat(embeddings, dim=0)
        else:
            return torch.empty(0, self.embedding_dim)


if __name__ == "__main__":
    extractor = TimeSeriesEmbeddingExtractor(
        model_name='facebook/wav2vec2-base',
        pooling='mean'
    )
    
    sample_rate = 16000
    duration_seconds = 2
    
    time_series_list = [
        np.random.randn(sample_rate * duration_seconds) for _ in range(5)
    ]
    
    # Extract embeddings
    embeddings = extractor.extract_embeddings(
        time_series_list, 
        sampling_rate=sample_rate,
        batch_size=2
    )
    
    print(f"\nFinal embeddings shape: {embeddings.shape}")
    # Output: torch.Size([5, 512])