# MNIST Handwritten Digit Recognizer - Streamlit Frontend
**Developer Name**: Qayoom Ali

**Roll Number**: 100072

**Subject**: Digital Image Processing

**Degree Program**: BS Computer Science

**Project Overview:**
This repository contains a semester project developed for the course Digital Image Processing (DIP).
The project focuses on building a Handwritten Digit Recognition System that is capable of identifying digits from handwritten images using the MNIST dataset. Image preprocessing and classification techniques are applied to achieve reliable recognition results.

**Objective of the Project:**
To develop an automated system for recognizing handwritten digits

To apply image preprocessing techniques for improving input quality

To utilize the MNIST dataset for training and validation

To implement a classification model for digit identification

To test the system using unseen handwritten digit images

To analyze the accuracy and effectiveness of the recognition process

**Tools and Technologies:**

Python Programming

OpenCV (Image Processing)

NumPy (Numerical Operations)

Matplotlib (Visualization)

Machine Learning Library (Scikit-learn / TensorFlow)

**Steps to Run the Code"**

Install Python and required libraries

Open the project and load the MNIST dataset

Run the main program file or notebook

View predicted results in the output folder

## Features

✨ **Interactive Interface:**
- 📤 Upload images of handwritten digits
- 🎨 Draw digits directly in the app (with canvas)
- 📊 View model architecture and training details

🔮 **Predictions:**
- Get instant digit predictions with confidence scores
- View probability distribution for all digits (0-9)
- Visual feedback with color-coded predictions

📈 **Model Information:**
- View detailed network architecture
- Training configuration details
- Tips for best results


## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install streamlit tensorflow keras numpy matplotlib Pillow opencv-python scikit-learn pandas seaborn streamlit-drawable-canvas
```

### 2. Ensure Model File Exists

Make sure `handWritten.keras` is in the same directory as `app.py`.

## Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Command Line Options

```bash
# Run with specific port
streamlit run app.py --server.port 8505

# Run in headless mode (server only)
streamlit run app.py --headless

# Set theme
streamlit run app.py --theme.base=dark
```

## Usage Guide

### 📤 Upload Image Mode
1. Select "Upload Image" from the sidebar
2. Upload an image of a handwritten digit
3. The app processes the image and displays the prediction
4. View confidence scores and probability distribution

### 🎨 Draw Digit Mode
1. Select "Draw Digit" from the sidebar
2. Draw a digit (0-9) on the canvas
3. The prediction appears automatically
4. Clear and draw again to test multiple digits

### 📊 Model Info Mode
1. Select "Model Info" from the sidebar
2. View complete network architecture
3. See training configuration
4. Get tips for best results

## Model Architecture

```
Layer 1: Conv2D(32, 3x3, ReLU)
         MaxPooling2D(2x2)
         
Layer 2: Conv2D(64, 3x3, ReLU)
         MaxPooling2D(2x2)
         
Layer 3: Flatten()
         Dense(128, ReLU)
         Dropout(0.5)
         Dense(10, Softmax)
```

**Training Details:**
- Dataset: MNIST (60,000 training, 10,000 test images)
- Input Size: 28×28 pixels (grayscale)
- Output: 10 classes (digits 0-9)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

## Tips for Best Results

✅ Keep digits centered in the image
✅ Ensure good contrast (dark digit on light background)
✅ Use images similar to MNIST dataset format
✅ Avoid heavily tilted or distorted digits
✅ The model works best with black and white images

## Project Structure

```
Dip/
├── app.py                 # Streamlit application
├── handWritten.keras      # Trained model
├── MINST_CNNs.ipynb      # Jupyter notebook with model training
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Troubleshooting

### Model Not Found
- Ensure `handWritten.keras` is in the same directory as `app.py`
- Check file name spelling (case-sensitive on some systems)

### Drawing Canvas Not Working
- Install `streamlit-drawable-canvas`: `pip install streamlit-drawable-canvas`
- Use the "Upload Image" mode as an alternative

### Slow Predictions
- First prediction loads the model (may take a few seconds)
- Subsequent predictions are faster due to caching
- Consider running on a machine with GPU for faster processing

### Memory Issues
- Close other applications
- Reduce browser extensions
- Clear browser cache

## Performance

- **Typical Accuracy:** ~99% on MNIST test set
- **Prediction Time:** <100ms per image (CPU)
- **Model Size:** ~1MB

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.28+
- See `requirements.txt` for full list

## License

This project uses the MNIST dataset and TensorFlow/Keras libraries under their respective licenses.

## Contributing

Feel free to improve the app! Possible enhancements:
- Batch processing of multiple images
- Model performance metrics comparison
- Confidence threshold adjustment
- Export predictions to CSV
- Add more visualization options

---

**Created:** 2024
**Status:** Active

**conclusion:**
In this project, a Handwritten Digit Recognition System was successfully implemented using the MNIST dataset. The system effectively processes handwritten digit images and accurately predicts the corresponding numerical value. This project provides a clear understanding of how Digital Image Processing and machine learning techniques can be applied to solve real-world recognition problems