🔤 Alphanumeric Handwritten Word OCR Pipeline
An end-to-end Machine Learning web application that accepts images of handwritten words or alphanumeric codes, automatically segments them into individual characters using computer vision, and decodes the entire text string using a custom Convolutional Neural Network (CNN).

The underlying deep learning engine is trained on the EMNIST (Extended MNIST) Balanced Dataset, achieving ~88% classification accuracy across 47 unique character classes (numbers, uppercase, and lowercase letters).

🚀 Features
Intelligent Word Segmentation: Uses OpenCV contour detection to isolate distinct characters dynamically from a single image line and sorts them chronologically from left to right.

Custom 3-Layer CNN Architecture: Fully trained sequence evaluator optimized to recognize handwritten variations in line weight, style, and skew.

Modern Async API Backend: Powered by FastAPI to handle super-fast asynchronous image array conversions and multi-character batch inference streams.

Clean Single-Page UI: A modern, simple, and responsive HTML5 interface that uploads and flashes prediction updates instantly without requiring a full page refresh.

🛠️ Tech Stack
Backend Engine: FastAPI, Uvicorn

Deep Learning Framework: TensorFlow (Keras API)

Computer Vision Processing: OpenCV (cv2), NumPy

Frontend Layer: HTML5, Modern CSS, Vanilla JavaScript (Fetch API)

📁 Repository Structure
Plaintext
ocr-fastapi-app/
│
├── app.py                     # FastAPI API backend & processing pipeline
├── emnist_character_model.keras # Pre-trained Keras CNN model weights
├── index.html                 # Frontend user interface webpage
└── README.md                  # Project documentation
⚙️ Setup & Installation
Follow these steps to get a local copy of this web application running on your personal machine.

Prerequisites
Ensure you have Python 3.12 or Python 3.13 installed on your system (TensorFlow does not currently support Python 3.14+).

1. Clone the Workspace
Bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
2. Configure an Isolated Environment
Set up a clean Python virtual environment to avoid package version updates clashing with your native machine setups.

Bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
3. Install Package Dependencies
Bash
pip install fastapi uvicorn tensorflow opencv-python numpy python-multipart
💻 How to Run the Project
Verify that your pre-trained model file (emnist_character_model.keras) is sitting inside the root folder of the project directory.

Fire up the local ASGI web-server engine using the following terminal execution command:

Bash
uvicorn app:app --reload
Once your console outputs confirmation that the network is initialized, open your favorite web browser (like Google Chrome) and head to the local serving address:
👉 http://127.0.0.1:8000

Drop or select an image asset containing clear, spaced handwritten characters and hit Scan Word Matrix to see your model outputs render live!

🧠 Core Processing Pipeline Overview
Whenever an image file enters the web app, it passes through the following lifecycle stages:

Binarization: Converts the incoming color matrix to Grayscale and applies an Inverse Binary Threshold to produce pure white characters sitting against a pure black canvas.

Contour Extraction: Uses OpenCV topological tracking routines to draw precise bounding boxes around isolated character frames, discarding noise artifacts.

EMNIST Orientation Layout: To match the precise configuration structures expected by the trained neural weights, each bounding-box matrix gets padded symmetrically to a perfect square, downsized cleanly to a 28x28 grid array, and transposed (.T) to handle structural axis flips.

CNN Evaluation & Join: Feeds all formatted matrices sequentially through the CNN layers, decodes class indices into actual alphanumeric characters via lookup maps, and stitches the letters back together into a final output sentence response.
