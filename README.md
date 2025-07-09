#  Handwritten Digit Recognition

> RISE Internship Project 2 – Tamizhan Skills  
> Built with TensorFlow, Keras, and Streamlit

A deep learning-based web app that recognizes handwritten digits from images using a trained CNN (Convolutional Neural Network). This is the second project from the **Machine Learning & AI** track of the RISE Internship by Tamizhan Skills.

---

##  Project Objective

To build an accurate digit recognizer that:
  - Loads and processes image data from the MNIST dataset
  - Trains a CNN for digit classification (0–9)
  - Provides an interactive Streamlit interface for real-time prediction

---

##  Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **PIL (Pillow)**
- **CNN (Convolutional Neural Network)**
- **Streamlit** (for UI)

---

##  Project Structure

```bash
digit-recognition/
├── app.py                  # Streamlit frontend for digit prediction
├── main.py                 # CNN training script
├── requirements.txt        # All required packages          # (optional) cached dataset
├── models/
│   └── cnn_model.h5        # Trained CNN model
├── src/
│   └── preprocess.py       # Data preprocessing and reshaping
├── .gitignore
└── README.md               # You're reading it 😉
```

---

## Dataset

- Source: MNIST dataset via Keras (tensorflow.keras.datasets.mnist)
- Contains 60,000 training and 10,000 test images (28x28 pixels)
- Each image is a grayscale handwritten digit (0–9)

---

## How to Run

- Step 1: Install Dependencies
  
```bash
  pip install -r requirements.txt
```

- Step 2: Train the Model
  
```bash
  python main.py
```

- Step 3: Launch the Web App
  
```bash
  streamlit run app.py
```

  ---

## Model Performance

✅ Accuracy: ~98% on test set
✅ Fast and efficient CNN architecture
✅ Supports grayscale image uploads for real-time digit recognition

---

## Highlights

- Utilizes a 3-layer CNN with ReLU + MaxPooling
- Preprocesses and reshapes image data for model input
- Provides a minimal but interactive UI for inference
- Modular codebase for easy reusability and upgrades

---

## Acknowledgements

Thanks to Tamizhan Skills for the RISE Internship opportunity.

Inspired by real-world spam filtering problems.

Built by @ShaikJasmin11
