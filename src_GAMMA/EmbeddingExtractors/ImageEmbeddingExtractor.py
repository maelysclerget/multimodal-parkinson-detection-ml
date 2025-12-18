import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from typing import List, Union
from pathlib import Path


class ImageEmbeddingExtractor:
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True, verbose: bool = True):
        """
        Initialize the embedding extractor with a Timm model.
        
        Args:
            model_name: Name of the Timm model to use (default: 'resnet50')
            pretrained: Whether to use pretrained weights (default: True)
            verbose: Whether the constructor should output verbal execution tracing (default: True)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model with num_classes=0 to get embeddings instead of classifications
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # Remove classification as embeddings are needed
        )
        self.model.eval()
        self.model.to(self.device)
        
        # Get the embedding dimension
        self.embedding_dim = self.model.num_features
        
        # Setup image transforms
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        
        if verbose:
            print(f"Model: {model_name}")
            print(f"Device: {self.device}")
            print(f"Embedding dimension: {self.embedding_dim}")
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load a single image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object in RGB format
        """
        img = Image.open(image_path).convert('RGB')
        return img
    
    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess and normalize a PIL Image using Timm's transform pipeline.
        
        Args:
            img: PIL Image object to preprocess
            
        Returns:
            Preprocessed image as a PyTorch tensor with applied normalization
        """
        img_tensor = self.transform(img)
        return img_tensor
    
    def extract_embeddings(self, image_paths: List[Union[str, Path]], 
                          batch_size: int = 32, verbose: bool = True) -> torch.Tensor:
        """
        Extract embeddings for a list of images.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once (default: 32)
            verbose: Whether the method should output verbal execution tracing (default: True)
            
        Returns:
            PyTorch tensor of shape (num_images, embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                # Load and preprocess batch
                batch_tensors = []
                for img_path in batch_paths:
                    try:
                        img = self.load_image(img_path)
                        img_tensor = self.preprocess_image(img)
                        batch_tensors.append(img_tensor)
                    except Exception as e:
                        if verbose:
                            print(f"Error processing {img_path}: {e}")
                        continue
                
                if batch_tensors == []:
                    continue
                
                # Stack into batch and move to device
                batch = torch.stack(batch_tensors).to(self.device)
                
                # Extract embeddings
                batch_embeddings = self.model(batch)
                embeddings.append(batch_embeddings.cpu())
                
                if verbose:
                    print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
        
        # Concatenate all embeddings
        if embeddings:
            return torch.cat(embeddings, dim=0)
        else:
            return torch.empty(0, self.embedding_dim)


