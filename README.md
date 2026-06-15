# Reducing Overfitting through Convolutional Neural Networks
## A Case Study using the GTZAN Dataset
 
## Overview
 
This project explores techniques to mitigate overfitting in Convolutional Neural Networks (CNNs) applied to image-based music genre classification. Using mel spectrogram images from the GTZAN dataset as input, several deep learning architectures and regularization strategies are compared, including a custom CNN with hyperparameter tuning, transfer learning with VGG16, and a Feed-forward Neural Network baseline.
 
The work was developed as an individual project for the *Statistical Learning* course at Università degli Studi di Milano-Bicocca (A.Y. 2024/25).
 
---
 
## Dataset
 
The [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data) contains 1,000 audio files evenly distributed across 10 music genres (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock). Each audio file is represented as a **mel spectrogram** in PNG format (224×224 pixels, RGB).
 
The dataset was split into:
- Train: 918 images (70%)
- Validation: 273 images (15%)
- Test: 280 images (15%)
---
 
## Methods
 
### 1. Initial CNN
A shallow CNN with 4 convolutional layers, max-pooling, and a dense classifier with dropout (rate = 0.4). Trained with the Adam optimizer (lr = 0.001), Early Stopping, and ReduceLROnPlateau callbacks.
 
**Test accuracy: 92.50%** — Test loss: 0.384
 
### 2. Hyperparameter Tuning
Grid search over dropout rate (0.3, 0.4) and learning rate (0.0001, 0.001). Best configuration: dropout = 0.4, lr = 0.0001.
 
**Test accuracy: 93.57%** — Test loss: 0.205 — reduced overfitting compared to the initial model.
 
### 3. Transfer Learning — VGG16 Feature Extraction
Features were extracted from the VGG16 convolutional base (pre-trained on ImageNet, top excluded). A custom classifier was added on top: Dense(512) → BatchNorm → Dropout(0.6) → Dense(128) → Dropout(0.5) → Softmax(10), with L2 regularization. Trained for up to 50 epochs with adaptive learning rate.
 
**Validation accuracy: ~94.87%**
 
### 4. Fine-tuning
VGG16 layers from `block3_conv1` onward were unfrozen (BatchNorm layers kept frozen) and the model was retrained with a lower learning rate (lr = 0.0001).
 
Validation accuracy remained stable (~94%), with slightly improved generalization.
 
### 5. Feed-forward Neural Network (baseline)
A fully connected network applied directly to flattened 224×224×3 images. Despite multiple dense layers and dropout, the model failed to learn meaningful patterns.
 
**Test accuracy: 10.36%** — confirms that spatial structure captured by CNNs is essential for this task.
 
---
 
## Results Summary
 
| Model | Test Accuracy | Test Loss |
|---|---|---|
| Initial CNN | 92.50% | 0.384 |
| Tuned CNN (dropout=0.4, lr=1e-4) | 93.57% | 0.205 |
| VGG16 Feature Extraction | ~94–95% (val) | — |
| FFNN | 10.36% | 2.302 |
 
Transfer learning with VGG16 outperformed the custom CNN, while the FFNN confirmed the importance of convolutional operations for image-based audio classification.
 
---
 
## Future Directions
 
- Incorporating raw audio features (MFCCs, chroma, spectral contrast) alongside or instead of spectrograms
- Applying Recurrent Neural Networks (RNNs/LSTMs) to capture temporal dependencies in audio
- Data augmentation on spectrograms to further reduce overfitting
---
 
## Tech Stack
 
- **Language:** R (RMarkdown)
- **Key packages:** `keras`, `tensorflow`
- **Architecture:** Custom CNN, VGG16 (ImageNet weights), Feed-forward NN
---
 
## Author
 
Beatrice Malvezzi
*Statistical Learning — A.Y. 2024/25, Università degli Studi di Milano-Bicocca*
