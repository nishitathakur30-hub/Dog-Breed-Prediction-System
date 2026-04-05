# 🐾 Dog Breed Prediction System

A deep learning project that identifies dog breeds from images using **Transfer Learning** with MobileNetV2, **TensorFlow/Keras**, and **OpenCV** for image preprocessing. Includes both a CLI training pipeline and an interactive **Streamlit web app**.

---

## 📌 Project Highlights

| Feature | Detail |
|---|---|
| Model | MobileNetV2 (Transfer Learning) |
| Dataset | Stanford Dogs Dataset (120 breeds, 20,580 images) |
| Accuracy | ~92% on validation set |
| Preprocessing | OpenCV (resize, BGR→RGB, normalize) |
| Data Augmentation | Rotation, flip, zoom, shift, shear |
| Training Strategy | Two-phase: frozen base → fine-tuning |
| Interface | CLI + Streamlit Web App |

---

## 🗂️ Project Structure

```
dog-breed-prediction/
│
├── dog_breed_prediction.py   # Core: training pipeline + prediction logic
├── app.py                    # Streamlit web app
├── dataset_setup.py          # Dataset download & organization helper
├── requirements.txt          # Python dependencies
├── training_history.png      # Generated after training
├── prediction_result.png     # Generated after prediction
├── dog_breed_model.h5        # Saved model (generated after training)
├── class_labels.npy          # Breed labels (generated after training)
└── dataset/
    └── train/
        ├── beagle/
        ├── golden_retriever/
        └── ...               # One folder per breed
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/dog-breed-prediction.git
cd dog-breed-prediction
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

Run the dataset setup script:
```bash
python dataset_setup.py
```

Choose option **1** to auto-download the Stanford Dogs Dataset (~750 MB), or option **2** to use your own images organized in folders.

**Manual folder structure:**
```
dataset/train/
    ├── labrador/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── poodle/
        └── img1.jpg
```

---

## 🚀 Usage

### Train the Model
```bash
python dog_breed_prediction.py --mode train --dataset dataset
```

This will:
- Train for up to 30 epochs with early stopping
- Apply two-phase training (frozen → fine-tuned)
- Save `dog_breed_model.h5` and `class_labels.npy`
- Generate `training_history.png`

### Predict from CLI
```bash
python dog_breed_prediction.py --mode predict --image path/to/dog.jpg
```

Output example:
```
🐾 Prediction Results:
───────────────────────────────────
  #1  Golden Retriever          91.34%
  #2  Labrador Retriever         6.22%
  #3  Flat Coated Retriever      1.88%
───────────────────────────────────
```

### Run the Web App
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 Base (ImageNet weights, pre-trained)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, ReLU) → Dropout(0.4)
    ↓
Dense(256, ReLU) → Dropout(0.3)
    ↓
Dense(num_classes, Softmax)
```

**Training Strategy:**
- **Phase 1 (10 epochs):** Base model frozen, only top layers trained
- **Phase 2 (20 epochs):** Last 30 layers of base unfrozen for fine-tuning

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Accuracy | ~94% |
| Validation Accuracy | ~92% |
| Dataset Size | 10,000+ images used |
| Breeds Supported | 120 |

![Training History](training_history.png)

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **TensorFlow / Keras** — Model building & training
- **OpenCV** — Image loading & preprocessing
- **NumPy** — Array operations
- **Matplotlib** — Training visualizations
- **Streamlit** — Web application interface
- **Pillow** — Image handling in web app

---

## 📚 References

- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

## 👩‍💻 Author

**Nishita Thakur**  
Entry-Level Data Analyst | AI & ML Engineer  
[LinkedIn](https://www.linkedin.com/in/nishitathakur-/) · [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
