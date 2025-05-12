import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from torch.optim import lr_scheduler
from PIL import Image
import timm
import string
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import pickle

NUM_CLASSES = 29
INPUT_SIZE = 224


class SLR:
    def __init__(self, num_classes=NUM_CLASSES, input_shape=(INPUT_SIZE, INPUT_SIZE)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Few-shot learning components
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine')
        self.few_shot_embeddings = {}
        self.few_shot_labels = []

    ################## build() #######################
    def build_model(self):
        """EfficientNet-B0 model for fine-tuning"""
        model = timm.create_model('efficientnet_b0', pretrained=True)

        # Freeze all
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last layers of backbone
        for name, param in model.named_parameters():
            if any(x in name for x in ['blocks.6', 'conv_head', 'bn2']):  # just finetune last layers
                param.requires_grad = True

        # Replace final classifier
        num_feas = model.classifier.in_features
        model.classifier = nn.Linear(num_feas, self.num_classes)
        model.classifier.requires_grad = True

        self.model = model.to(self.device)
        print("Model built: backbone partially unfrozen.")
        return self.model

    ################## ext_features() #######################
    def extract_features(self, img):
        """Extract features from an image using the backbone"""
        # Ensure model is in eval mode for feature extraction
        self.model.eval()

        # Preprocess the image
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8')).convert('RGB')

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Extract features - we'll use the penultimate layer
        with torch.inference_mode():
            # Register a forward hook to get features from the penultimate layer
            features = []

            def hook_fn(module, input, output):
                features.append(output)

            # Attach the hook to the global pooling layer
            hook = self.model.global_pool.register_forward_hook(hook_fn)

            # Forward pass
            _ = self.model(img_tensor)

            # Remove the hook
            hook.remove()

            # Get the features
            return features[0].squeeze().cpu().numpy()

    ################## add_few_shot() #######################
    def add_few_shot_sample(self, image, label):
        """Add a new few-shot sample for a custom gesture"""
        features = self.extract_features(image)

        if label in self.few_shot_embeddings:
            self.few_shot_embeddings[label].append(features)
        else:
            self.few_shot_embeddings[label] = [features]
            self.few_shot_labels.append(label)

        # Retrain KNN whenever new samples are added
        self._update_few_shot_model()

        return True

    ################## update() #######################
    def _update_few_shot_model(self):
        """Update the few-shot learning model with the current embeddings"""
        all_embeddings = []
        all_labels = []

        for label, embeddings in self.few_shot_embeddings.items():
            for emb in embeddings:
                all_embeddings.append(emb)
                all_labels.append(label)

        if len(all_embeddings) > 0:
            X = np.array(all_embeddings)
            y = np.array(all_labels)
            self.knn.fit(X, y)

    ################## predict() #######################
    def predict(self, img, include_few_shot=True, confidence_threshold=0.7):
        """
        Predict the class for an image.
        Returns: {'label': ..., 'confidence': ..., 'source': 'standard' | 'few-shot'}
        """
        self.model.eval()

        # Preprocess the image
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8')).convert('RGB')

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Extract features for few-shot
        features = self.extract_features(img)

        # Standard model prediction
        with torch.inference_mode():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()

        standard_class_idx = int(np.argmax(probabilities))
        standard_confidence = float(probabilities[standard_class_idx])

        # If no few-shot or not enabled, return standard prediction
        if not include_few_shot or not self.few_shot_labels:
            return {
                'label': standard_class_idx,
                'confidence': standard_confidence,
                'source': 'standard'
            }

        # Few-shot KNN prediction
        few_shot_pred = self.knn.predict([features])[0]
        
        total_samples = sum(len(emb_list) for emb_list in self.few_shot_embeddings.values())
        n_neighbors = min(5, total_samples)  
        
        if n_neighbors > 0:  
            distances, _ = self.knn.kneighbors([features], n_neighbors=n_neighbors)
            similarities = 1 - distances[0]
            avg_similarity = float(np.mean(similarities))
        else:
            avg_similarity = 0.0

        # Compare and select best source
        if avg_similarity > confidence_threshold and avg_similarity > standard_confidence:
            return {
                'label': few_shot_pred,
                'confidence': avg_similarity,
                'source': 'few-shot'
            }
        else:
            return {
                'label': standard_class_idx,
                'confidence': standard_confidence,
                'source': 'standard'
            }

    ################## save() #######################
    def save(self, model_path, few_shot_path=None):
        dir_path = os.path.dirname(model_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        torch.save(self.model.state_dict(), model_path)

        if few_shot_path:
            few_shot_dir = os.path.dirname(few_shot_path)
            if few_shot_dir and not os.path.exists(few_shot_dir):
                os.makedirs(few_shot_dir, exist_ok=True)

            few_shot_data = {
                'embeddings': self.few_shot_embeddings,
                'labels': self.few_shot_labels
            }
            with open(few_shot_path, 'wb') as f:
                pickle.dump(few_shot_data, f)

    ################## load() #######################
    def load(self, model_path, few_shot_path=None):
        """Load the model and few-shot embeddings"""
        # Build model architecture if not already built
        if self.model is None:
            self.build_model()

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load few-shot data if exists
        if few_shot_path and os.path.exists(few_shot_path):
            with open(few_shot_path, 'rb') as f:
                few_shot_data = pickle.load(f)
                self.few_shot_embeddings = few_shot_data['embeddings']
                self.few_shot_labels = few_shot_data['labels']

            # Update the KNN model with loaded embeddings
            self._update_few_shot_model()

    ################## plot() #######################
    def _plot_training_history(self, history):
        epochs = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        plt.plot(epochs, history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], label='Train Accuracy')
        plt.plot(epochs, history['val_acc'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

    ################## evaluate() #######################
    def evaluate_model(self, loader, split_name):
        """
        Evaluate model on dataset (val/test), print confusion matrix and classification report
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        if self.num_classes == 29:
            class_names = list(string.ascii_uppercase) + ['del', 'nothing', 'space']
        else:
            class_names = list(string.ascii_uppercase)[:self.num_classes]

        with torch.inference_mode():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{split_name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        # Classification report
        print(f"\n{split_name} Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    ################## train() #######################
    def train_model(self, train_loader, val_loader, epochs=10, lr=0.001, patience=3, warmup_epochs=3):
        """
        Train the model with early stopping, plotting, and warm-up
        """
        self.model.train()

        # Loss with label smoothing
        criterion = nn.CrossEntropyLoss()

        # Optimizer with weight decay (L2 regularization)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=1e-5
        )

        # Warm-up scheduler: increase LR linearly from lr/10 to lr over warmup_epochs
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs * len(train_loader)
        )

        # Main scheduler after warm-up
        main_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        best_val_acc = 0
        best_model_wts = None
        early_stop_counter = 0

        # Store loss/accuracy history for visualization
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'log': []
        }

        for epoch in range(epochs):
            running_loss = 0.0
            running_corrects = 0

            # Add tqdm progress bar for training
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
            for inputs, labels in train_pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()

                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar description with current loss
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

                # Step warm-up scheduler during warm-up phase
                if epoch < warmup_epochs:
                    warmup_scheduler.step()

            # Compute training metrics
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_corrects = 0

            # Add tqdm progress bar for validation
            with torch.inference_mode():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
                for inputs, labels in val_pbar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

                    # Update progress bar description with current loss
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc.item())

            # Store metrics for this epoch
            history['log'].append({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_acc': epoch_acc.item(),
                'val_loss': val_epoch_loss,
                'val_acc': val_epoch_acc.item()
            })

            # Adjust learning rate with main scheduler after warm-up
            if epoch >= warmup_epochs:
                main_scheduler.step(val_epoch_loss)

            # Print to console
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f} | "
                  f"Learning Rate: {current_lr:.6f}", flush=True)

            # Early stopping check
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                best_model_wts = self.model.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1} - no improvement for {patience} epochs.")
                    break

        # Restore best weights
        if best_model_wts:
            self.model.load_state_dict(best_model_wts)

        # Final report and plots
        self._plot_training_history(history)

        return history
