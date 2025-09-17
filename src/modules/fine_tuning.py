"""
Fine-tuning Module for Sentiment Analysis
Demonstrates real fine-tuning on IMDb movie reviews using DistilBERT
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import time
import os

# Optional imports with error handling
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        DistilBertTokenizer, 
        DistilBertForSequenceClassification,
        AdamW,
        get_linear_schedule_with_warmup
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


if TORCH_AVAILABLE:
    class IMDbDataset(Dataset):
        """Custom Dataset for IMDb movie reviews"""
        
        def __init__(self, texts, labels, tokenizer, max_length=512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            # Tokenize the text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
else:
    class IMDbDataset:
        """Placeholder class when PyTorch is not available"""
        def __init__(self, *args, **kwargs):
            pass


if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
    class SentimentClassifier(nn.Module):
        """DistilBERT-based sentiment classifier"""
        
        def __init__(self, num_classes=2, freeze_bert=False):
            super(SentimentClassifier, self).__init__()
            self.bert = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=num_classes
            )
            
            if freeze_bert:
                # Freeze all parameters except the classification head
                for param in self.bert.parameters():
                    param.requires_grad = False
                # Unfreeze the classification head
                for param in self.bert.classifier.parameters():
                    param.requires_grad = True
        
        def forward(self, input_ids, attention_mask, labels=None):
            return self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
else:
    class SentimentClassifier:
        """Placeholder class when PyTorch/Transformers are not available"""
        def __init__(self, *args, **kwargs):
            pass


class FineTuningDemo:
    """Main class for fine-tuning demonstration"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def load_imdb_data(self, sample_size=1000):
        """Load and preprocess IMDb dataset"""
        if not DATASETS_AVAILABLE:
            st.error("❌ The 'datasets' library is not installed. Please install it with: pip install datasets")
            return None
        
        if not SKLEARN_AVAILABLE:
            st.error("❌ The 'scikit-learn' library is not installed. Please install it with: pip install scikit-learn")
            return None
        
        try:
            # Load the IMDb dataset from Hugging Face
            st.info("📥 Loading IMDb dataset...")
            dataset = load_dataset("imdb")
            
            # Get train and test splits
            train_data = dataset['train']
            test_data = dataset['test']
            
            # Sample data for faster training (optional)
            if sample_size < len(train_data):
                train_indices = np.random.choice(len(train_data), sample_size, replace=False)
                train_texts = [train_data[i]['text'] for i in train_indices]
                train_labels = [train_data[i]['label'] for i in train_indices]
            else:
                train_texts = train_data['text']
                train_labels = train_data['label']
            
            # Sample test data
            test_size = min(sample_size // 4, len(test_data))
            test_indices = np.random.choice(len(test_data), test_size, replace=False)
            test_texts = [test_data[i]['text'] for i in test_indices]
            test_labels = [test_data[i]['label'] for i in test_indices]
            
            # Split train into train/validation
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
            )
            
            return {
                'train_texts': train_texts,
                'train_labels': train_labels,
                'val_texts': val_texts,
                'val_labels': val_labels,
                'test_texts': test_texts,
                'test_labels': test_labels
            }
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None
    
    def prepare_data_loaders(self, data, batch_size=16, max_length=512):
        """Prepare data loaders for training"""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.error("❌ PyTorch and Transformers are required for data preparation")
            return False
        
        try:
            # Initialize tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Create datasets
            train_dataset = IMDbDataset(
                data['train_texts'], 
                data['train_labels'], 
                self.tokenizer, 
                max_length
            )
            val_dataset = IMDbDataset(
                data['val_texts'], 
                data['val_labels'], 
                self.tokenizer, 
                max_length
            )
            test_dataset = IMDbDataset(
                data['test_texts'], 
                data['test_labels'], 
                self.tokenizer, 
                max_length
            )
            
            # Create data loaders
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return True
            
        except Exception as e:
            st.error(f"Error preparing data loaders: {str(e)}")
            return False
    
    def initialize_model(self, freeze_bert=False):
        """Initialize the sentiment classification model"""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.error("❌ PyTorch and Transformers are required for model initialization")
            return False
        
        try:
            self.model = SentimentClassifier(num_classes=2, freeze_bert=freeze_bert)
            self.model.to(self.device)
            return True
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return False
    
    def train_epoch(self, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader):
        """Evaluate the model on given data loader"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1, all_predictions, all_labels
    
    def train_model(self, epochs=3, learning_rate=2e-5, freeze_bert=False):
        """Train the sentiment classification model"""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.error("❌ PyTorch and Transformers are required for training")
            return False
        
        try:
            # Initialize model
            if not self.initialize_model(freeze_bert):
                return False
            
            # Setup optimizer and scheduler
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            total_steps = len(self.train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            # Training loop
            best_val_f1 = 0
            patience = 2
            patience_counter = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(epochs):
                status_text.text(f"Epoch {epoch + 1}/{epochs}")
                
                # Train
                train_loss, train_acc = self.train_epoch(optimizer, scheduler)
                
                # Validate
                val_loss, val_acc, val_f1, _, _ = self.evaluate(self.val_loader)
                
                # Store history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_acc'].append(val_acc)
                self.training_history['val_f1'].append(val_f1)
                
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                
                # Display metrics
                st.write(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
                # Early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        st.info(f"Early stopping at epoch {epoch + 1}")
                        break
            
            progress_bar.empty()
            status_text.empty()
            
            return True
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            return False
    
    def plot_training_history(self):
        """Plot training history"""
        if not MATPLOTLIB_AVAILABLE:
            st.warning("Matplotlib not available for plotting")
            return None
        
        if not self.training_history['train_loss']:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.training_history['train_acc'], label='Train Accuracy')
        axes[1].plot(self.training_history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # F1 plot
        axes[2].plot(self.training_history['val_f1'], label='Validation F1')
        axes[2].set_title('Validation F1 Score')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        return fig
    
    def test_model(self):
        """Test the trained model on test set"""
        if not SKLEARN_AVAILABLE:
            st.error("❌ Scikit-learn is required for model evaluation")
            return None
        
        if self.model is None or self.test_loader is None:
            st.error("Model not trained or test data not available")
            return None
        
        test_loss, test_acc, test_f1, predictions, labels = self.evaluate(self.test_loader)
        
        # Classification report
        report = classification_report(labels, predictions, target_names=['Negative', 'Positive'])
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'predictions': predictions,
            'labels': labels,
            'classification_report': report
        }
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.error("❌ PyTorch and Transformers are required for predictions")
            return None
        
        if self.model is None or self.tokenizer is None:
            return None
        
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'positive': probabilities[0][1].item()
            }
        }
    
    def render_interface(self):
        """Render the fine-tuning interface"""
        st.header("🎯 Fine-tuning Demo: Sentiment Analysis")
        
        # Check dependencies first
        missing_deps = []
        if not TORCH_AVAILABLE:
            missing_deps.append("torch")
        if not TRANSFORMERS_AVAILABLE:
            missing_deps.append("transformers")
        if not SKLEARN_AVAILABLE:
            missing_deps.append("scikit-learn")
        if not MATPLOTLIB_AVAILABLE:
            missing_deps.append("matplotlib")
        if not DATASETS_AVAILABLE:
            missing_deps.append("datasets")
        
        if missing_deps:
            st.error(f"❌ Missing required dependencies: {', '.join(missing_deps)}")
            st.markdown("""
            **To use the fine-tuning demo, please install the missing dependencies:**
            
            ```bash
            pip install torch transformers scikit-learn matplotlib datasets seaborn
            ```
            
            Or install all requirements:
            ```bash
            pip install -r requirements.txt
            ```
            """)
            return
        
        st.markdown("""
        **Goal**: Train a DistilBERT model to classify movie reviews as positive or negative sentiment.
        
        **What you'll learn**:
        - Tokenization and data preprocessing
        - Train/validation split and data loaders
        - Cross-entropy loss and optimization
        - Accuracy and F1 score evaluation
        - Early stopping and model selection
        """)
        
        # Configuration section
        st.subheader("⚙️ Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample Size", 100, 2000, 1000, 100, 
                                  help="Number of training samples (smaller = faster training)")
        
        with col2:
            epochs = st.slider("Epochs", 1, 10, 3, 1, 
                             help="Number of training epochs")
        
        with col3:
            batch_size = st.slider("Batch Size", 8, 32, 16, 8, 
                                 help="Training batch size")
        
        col4, col5 = st.columns(2)
        
        with col4:
            learning_rate = st.selectbox("Learning Rate", 
                                       [1e-5, 2e-5, 5e-5], 
                                       index=1,
                                       help="Learning rate for optimizer")
        
        with col5:
            freeze_bert = st.checkbox("Freeze BERT Layers", 
                                    value=False,
                                    help="Freeze pre-trained BERT layers, only train classification head")
        
        # Action buttons
        st.subheader("🚀 Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Load Data", key="load_data"):
                with st.spinner("Loading IMDb dataset..."):
                    data = self.load_imdb_data(sample_size)
                    if data:
                        st.session_state.ft_data = data
                        st.success(f"✅ Data loaded: {len(data['train_texts'])} train, "
                                 f"{len(data['val_texts'])} val, {len(data['test_texts'])} test samples")
        
        with col2:
            if st.button("🏋️ Train Model", key="train_model"):
                if 'ft_data' not in st.session_state:
                    st.error("Please load data first!")
                else:
                    with st.spinner("Preparing data and training model..."):
                        # Prepare data loaders
                        if self.prepare_data_loaders(st.session_state.ft_data, batch_size):
                            # Train model
                            if self.train_model(epochs, learning_rate, freeze_bert):
                                st.success("✅ Model training completed!")
                                st.session_state.ft_model = self
                            else:
                                st.error("❌ Training failed!")
        
        with col3:
            if st.button("📊 Test Model", key="test_model"):
                if 'ft_model' not in st.session_state:
                    st.error("Please train a model first!")
                else:
                    with st.spinner("Testing model..."):
                        results = self.test_model()
                        if results:
                            st.session_state.ft_results = results
                            st.success("✅ Model testing completed!")
        
        # Display results
        if 'ft_data' in st.session_state:
            self.display_data_info(st.session_state.ft_data)
        
        if 'ft_model' in st.session_state and st.session_state.ft_model.training_history['train_loss']:
            self.display_training_results()
        
        if 'ft_results' in st.session_state:
            self.display_test_results(st.session_state.ft_results)
        
        # Interactive prediction
        self.render_prediction_interface()
    
    def display_data_info(self, data):
        """Display information about loaded data"""
        st.subheader("📊 Dataset Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Samples", len(data['train_texts']))
        with col2:
            st.metric("Validation Samples", len(data['val_texts']))
        with col3:
            st.metric("Test Samples", len(data['test_texts']))
        
        # Show sample reviews
        st.subheader("📝 Sample Reviews")
        
        sample_idx = st.slider("Select sample", 0, min(10, len(data['train_texts'])-1), 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Sample:**")
            st.write(f"**Label:** {'Positive' if data['train_labels'][sample_idx] == 1 else 'Negative'}")
            st.write(f"**Text:** {data['train_texts'][sample_idx][:200]}...")
        
        with col2:
            st.write("**Validation Sample:**")
            st.write(f"**Label:** {'Positive' if data['val_labels'][sample_idx] == 1 else 'Negative'}")
            st.write(f"**Text:** {data['val_texts'][sample_idx][:200]}...")
    
    def display_training_results(self):
        """Display training results and plots"""
        st.subheader("📈 Training Results")
        
        # Plot training history
        fig = self.plot_training_history()
        if fig:
            st.pyplot(fig)
        
        # Final metrics
        if self.training_history['val_f1']:
            final_val_f1 = self.training_history['val_f1'][-1]
            final_val_acc = self.training_history['val_acc'][-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Validation F1", f"{final_val_f1:.4f}")
            with col2:
                st.metric("Final Validation Accuracy", f"{final_val_acc:.4f}")
    
    def display_test_results(self, results):
        """Display test results"""
        st.subheader("🎯 Test Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
        with col2:
            st.metric("Test F1 Score", f"{results['test_f1']:.4f}")
        with col3:
            st.metric("Test Loss", f"{results['test_loss']:.4f}")
        
        # Classification report
        st.subheader("📋 Detailed Classification Report")
        st.text(results['classification_report'])
    
    def render_prediction_interface(self):
        """Render interface for interactive predictions"""
        st.subheader("🔮 Interactive Prediction")
        
        if 'ft_model' not in st.session_state:
            st.info("Please train a model first to make predictions!")
            return
        
        # Text input
        user_text = st.text_area(
            "Enter a movie review to analyze:",
            placeholder="This movie was absolutely fantastic! The acting was superb and the plot was engaging...",
            height=100
        )
        
        if st.button("🔍 Predict Sentiment", key="predict_sentiment"):
            if user_text.strip():
                with st.spinner("Analyzing sentiment..."):
                    prediction = self.predict_sentiment(user_text)
                    
                    if prediction:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Predicted Sentiment", prediction['sentiment'])
                            st.metric("Confidence", f"{prediction['confidence']:.4f}")
                        
                        with col2:
                            st.metric("Negative Probability", f"{prediction['probabilities']['negative']:.4f}")
                            st.metric("Positive Probability", f"{prediction['probabilities']['positive']:.4f}")
                        
                        # Visualize probabilities
                        if MATPLOTLIB_AVAILABLE:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sentiments = ['Negative', 'Positive']
                            probs = [prediction['probabilities']['negative'], prediction['probabilities']['positive']]
                            colors = ['red', 'green']
                            
                            bars = ax.bar(sentiments, probs, color=colors, alpha=0.7)
                            ax.set_ylabel('Probability')
                            ax.set_title('Sentiment Prediction Probabilities')
                            ax.set_ylim(0, 1)
                            
                            # Add value labels on bars
                            for bar, prob in zip(bars, probs):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{prob:.3f}', ha='center', va='bottom')
                            
                            st.pyplot(fig)
                        else:
                            st.info("Matplotlib not available for probability visualization")
            else:
                st.warning("Please enter some text to analyze!")
