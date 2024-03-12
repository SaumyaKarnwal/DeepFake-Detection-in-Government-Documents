# DeepFake Detection in Government Documents

The proliferation of DeepFake technology poses a significant threat to the integrity and authenticity of government documents. DeepFake images, which are manipulated to appear real, can be used to create fraudulent documents with potentially serious consequences. There is a need for a reliable and efficient method to detect DeepFake images in government documents to ensure their trustworthiness and prevent misuse. Developing a deep learning model that can accurately detect DeepFake images in government documents based on texture analysis is crucial for maintaining the integrity of such documents.

## Problem Statement

This project focuses on detecting DeepFake images used in government documents by employing a texture-based approach. DeepFake images, which are manipulated to appear real, pose a significant threat to the authenticity and integrity of government records. To address this, we have developed a deep learning model that can identify these fake images with high accuracy.

## Methodology

### 1. Region of Interest (ROI) Extraction

We start by using YOLOv8, a state-of-the-art object detection model, to identify the region of interest in the scanned document that contains the image. This step is crucial for focusing our analysis on the relevant parts of the document.

### 2. Contrast Enhancement

Since our approach is based on texture analysis, we enhance the contrast of the extracted region using Contrast Limited Adaptive Histogram Equalization (CLAHE). This helps in highlighting the texture details, which are important for detecting DeepFake images.

### 3. Feature Detection and Extraction

We use Circularly Shifted Local Binary Patterns (CS LBP) in conjunction with a sliding window approach to detect and extract texture features from the enhanced region. CS LBP is effective in capturing the subtle texture variations that are characteristic of DeepFake images.

### 4. Convolutional Neural Network (CNN) Training

The extracted features are then used to train a CNN model consisting of 3 convolutional layers and 1 fully connected layer. Each convolutional block is followed by max pooling and batch normalization to improve the model's performance. The output of the model is binary, indicating whether the image is real (1) or fake (0).

## System Configurations

To execute this implementation, the following system configurations are needed:

- Python 3.x
- TensorFlow
- OpenCV
- YOLOv8
- scikit-image
- numpy
- matplotlib
- other required libraries (mentioned in requirements.txt)

## References

1. Le, T.-N., Nguyen, H. H., Yamagishi, J., & Echizen, I. (2021). OpenForensics: Multi-Face Forgery Detection And Segmentation In-The-Wild Dataset [V.1.0.0] (1.0.0) [Data set]. International Conference on Computer Vision (ICCV). Zenodo. https://doi.org/10.5281/zenodo.5528418
2. Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). Faceforensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1-11).
3. X. Chang, J. Wu, T. Yang and G. Feng, "DeepFake Face Image Detection based on Improved VGG Convolutional Neural Network," 2020 39th Chinese Control Conference (CCC), Shenyang, China, 2020, pp. 7252-7256, doi: 10.23919/CCC50068.2020.9189596.
4. Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). Celeb-df: A large-scale challenging dataset for deepfake forensics. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 3207-3216).
5. Jian Wu, Kai Feng, Xu Chang, and Tongfeng Yang. 2020. A Forensic Method for DeepFake Image based on Face Recognition. In Proceedings of the 2020 4th High Performance Computing and Cluster Technologies Conference & 2020 3rd International Conference on Big Data and Artificial Intelligence (HPCCT & BDAI '20). Association for Computing Machinery, New York, NY, USA, 104–108. https://doi.org/10.1145/3409501.3409544

## Methodology

### 1. Region of Interest (ROI) Extraction

We start by using YOLOv8, a state-of-the-art object detection model, to identify the region of interest in the scanned document that contains the image. This step is crucial for focusing our analysis on the relevant parts of the document.

### 2. Contrast Enhancement

Since our approach is based on texture analysis, we enhance the contrast of the extracted region using Contrast Limited Adaptive Histogram Equalization (CLAHE). This helps in highlighting the texture details, which are important for detecting deepfake images.

### 3. Feature Detection and Extraction

We use Circularly Shifted Local Binary Patterns (CS LBP) in conjunction with a sliding window approach to detect and extract texture features from the enhanced region. CS LBP is effective in capturing the subtle texture variations that are characteristic of deepfake images.

### 4. Convolutional Neural Network (CNN) Training

The extracted features are then used to train a CNN model consisting of 3 convolutional layers and 1 fully connected layer. Each convolutional block is followed by max pooling and batch normalization to improve the model's performance. The output of the model is binary, indicating whether the image is real (1) or fake (0).

## System Configurations

To execute this implementation, the following system configurations are needed:

- Python 3.x
- TensorFlow
- OpenCV
- YOLOv8
- scikit-image
- numpy
- matplotlib

