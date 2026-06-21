import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="Alphanumeric OCR API")

# Load your trained Keras model globally
try:
    model = tf.keras.models.load_model('emnist_character_model.keras')
    print("🤖 Neural Network successfully initialized.")
except Exception as e:
    print(f"❌ Error loading model file: {e}. Ensure it is named correctly in this folder.")

emnist_labels = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# Serve the frontend webpage
@app.get("/")
async def read_index():
    return FileResponse('index.html')

# The API endpoint that processes incoming images
@app.post("/predict")
async def predict_word(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    # OpenCV Segmentation Pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"prediction": "No characters detected"}
        
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0]))
    
    predicted_word = ""
    
    for box in boundingBoxes:
        x, y, w, h = box
        roi = thresh[y:y+h, x:x+w]
        
        pad = max(w, h) + 20
        letter_pad = np.zeros((pad, pad), dtype="uint8")
        letter_pad[(pad-h)//2:(pad-h)//2+h, (pad-w)//2:(pad-w)//2+w] = roi
        
        # Format for EMNIST (Resize -> Transpose -> Normalize -> Reshape)
        resized = cv2.resize(letter_pad, (28, 28), interpolation=cv2.INTER_AREA)
        transposed = resized.T
        input_tensor = (transposed / 255.0).reshape(1, 28, 28, 1)
        
        # Infer using your model weights
        pred = model.predict(input_tensor, verbose=0)
        predicted_word += emnist_labels[np.argmax(pred)]
        
    return {"prediction": predicted_word}