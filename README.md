# DIS25K Image Splicing Detection

A deep learning-based image forgery detection and pixel-level localization system developed as an Artificial Intelligence CEP project.

## Project Overview

This project uses a U-Net architecture with an EfficientNet-B0 encoder to detect and localize manipulated regions in digital images.

## Features

- AI-powered image splicing detection
- Pixel-level forgery localization
- Binary segmentation mask
- Probability heatmap visualization
- Forgery overlay visualization
- Image-level forgery score
- Interactive Flask web interface
- CPU and GPU support

## Technologies Used

- Python
- PyTorch
- EfficientNet-B0
- U-Net
- Flask
- NumPy
- OpenCV
- HTML, CSS and JavaScript

## Model Performance

The following results were recorded during the original project evaluation.

| Metric | Score |
|---|---:|
| Validation IoU | 83.10% |
| Validation F1 | 89.49% |
| Test Mean IoU | 59.90% |
| Test Mean F1 | 67.60% |
| Test Median IoU | 75.20% |
| Test Median F1 | 85.90% |

## Installation

Install the required dependencies:

`pip install -r requirements.txt`

Run the application:

`python app.py`

Open `http://127.0.0.1:5000` in your browser.

## Future Improvements

- Improve small-region forgery detection
- Increase input image resolution
- Explore transformer-based encoders
- Improve generalization across datasets
- Introduce test-time augmentation

## Author

Mahrukh Shahzad

Computer Engineering | COMSATS University Islamabad, Lahore Campus

Artificial Intelligence CEP Project
