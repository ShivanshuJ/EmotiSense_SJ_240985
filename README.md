# EmotiSense_SJ_240985

Emotisense is an end-to-end Computer Vision pipeline designed to recognize human emotions in real-time. Developed as a technical project under Stamatics, IIT Kanpur, it leverages Convolutional Neural Networks (CNNs) to bridge the gap between raw video frames and human sentiment analysis.

**Key Features:**    

- **Real-Time Inference:** High-speed processing of webcam feeds using optimized OpenCV pipelines.

- **Robust Face Detection:** Implements a dual-approach using Haar Cascades and DNN-based methods for consistent tracking under varying lighting.

- **Deep Learning Core:** A custom-trained CNN architecture optimized on the FER-2013 dataset, classifying seven distinct emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

**Tech Stack:**  

- **Language:** Python

- **ML/DL:** TensorFlow, Keras, Scikit-learn

- **Computer Vision:** OpenCV

- **Data Handling:** Pandas, NumPy


# Working    
**1. Pre-processing & Face Detection:**  The system captures frames from a live webcam feed. To reduce computational load, frames are converted to grayscale. We utilize a Haar Cascade classifier (or a DNN-based ResNet-10 SSD) to localize the face. The detected region of interest (ROI) is then cropped and resized to $48 \times 48$ pixels to match the input layer of our model.  

**2. The CNN Architecture:** The heart of Emotisense is a deep Convolutional Neural Network. Unlike standard flat networks, CNNs are designed to recognize spatial hierarchies in images (like the curve of a smile or the furrow of a brow).  
- **Convolutional Layers:** These act as filters that scan the image to detect edges, textures, and eventually complex facial features.
- **Pooling Layers:** We use Max Pooling to downsample the feature maps, which reduces the number of parameters and prevents the model from overfitting to specific noise in the background.
- **Dropout:** Applied at $25\%$ and $50\%$ intervals to ensure the model generalizes well to faces it hasn't seen before.

**3. Inference & Classification:** The final "Dense" layers of the network act as a classifier. The output layer uses a Softmax activation function, which provides a probability distribution across the seven emotion categories. The emotion with the highest probability ($P_{max}$) is displayed as the real-time label on the video feed.  
<p align= center> $$P(y=j | \mathbf{x}) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$ </p>

