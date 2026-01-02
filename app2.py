import os, io, json, pickle, numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


app = Flask(__name__, static_folder="static", template_folder="templates")

MODEL_PATH = os.path.join("model", "model_fold_1.keras")
SEX_ENCODER_PATH = os.path.join("model", "sex_encoder.pkl")
LOC_ENCODER_PATH = os.path.join("model", "loc_encoder.pkl")
LABEL_ENCODER_PATH = os.path.join("model", "label_encoder.pkl")
PREPROCESS_CONFIG = os.path.join("model", "preprocess_config.json")

IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

with open(SEX_ENCODER_PATH, "rb") as f:
    sex_encoder = pickle.load(f)
with open(LOC_ENCODER_PATH, "rb") as f:
    loc_encoder = pickle.load(f)
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
with open(PREPROCESS_CONFIG, "r") as f:
    preprocess_cfg = json.load(f)

CANCER_CLASSES = {"mel", "bcc","akiec"}

def preprocess_image_file(file_storage):
    file_storage.stream.seek(0)
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    img_batch = np.expand_dims(arr, axis=0)
    return img_batch

def preprocess_metadata(age, sex, localization):
    try:
        age_val = float(age)
    except:
        age_val = 0.0

    sex_val = sex if (sex is not None and sex != "") else "unknown"
    sex_code = sex_encoder.transform([sex_val])[0]  # LabelEncoder -> integer

    loc_val = localization if (localization is not None and localization != "") else "unknown"
    loc_ohe = loc_encoder.transform([[loc_val]])[0]  # returns 1D array

    meta_vec = np.concatenate([[age_val], [sex_code], loc_ohe]).astype("float32")
    meta_batch = np.expand_dims(meta_vec, axis=0)
    return meta_batch

def predict_image_and_meta(img_batch, meta_batch):
    preds = model.predict([img_batch, meta_batch])
    probs = preds[0]        
    idx = int(np.argmax(probs))
    class_name = label_encoder.inverse_transform([idx])[0]
    confidence = float(probs[idx])
    is_cancer = bool(class_name in CANCER_CLASSES)
    return class_name, confidence, is_cancer, probs.tolist()

 
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    image_file = request.files["image"]
    age = request.form.get("age", "")
    sex = request.form.get("gender", "")
    localization = request.form.get("location", "")

    try:
        img_batch = preprocess_image_file(image_file)
        meta_batch = preprocess_metadata(age, sex, localization)
    except Exception as e:
        return jsonify({"error": f"preprocessing error: {str(e)}"}), 500

    class_name, confidence, is_cancer, probs = predict_image_and_meta(img_batch, meta_batch)

    if class_name == "mel":
        cancer_type = "Melanoma"
    elif class_name == "bcc":
        cancer_type = "Basal Cell Carcinoma"
    elif class_name == "akiec":
        cancer_type = "Actinic Keratoses and Intraepithelial Carcinoma"
    elif class_name == "bkl":
        cancer_type = "Benign Keratosis-like Lesions"
    elif class_name == "df":
        cancer_type = "Dermatofibroma"
    elif class_name == "nv":
        cancer_type = "Melanocytic Nevus"
    elif class_name == "vasc":
        cancer_type = "Vascular Lesions"
    else:
        cancer_type = "Unknown lesion type"


    prompt = f"""
    You are a medical assistant AI. Explain the following skin lesion classification result 
    in simple, patient-friendly language (not too technical).

    Prediction result:
    - Cancer / lesion type: {cancer_type}

    Include:
    - What this cancer/lesion type generally means
    - Whether it is dangerous or not
    - Symptoms and warning signs
    - What the user should do next (example: visit dermatologist)
    Keep it under 50 words and the explanation should be very easy to understand such that normal people can understand this.
    """

    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        gemini_response = gemini_model.generate_content(prompt)
        detailed_explanation = gemini_response.text
    except Exception as e:
        detailed_explanation = (
            "The model predicted your result, but the AI explanation service failed. "
            "Please try again later."
        )


    return jsonify({
        "cancer_type": class_name,
        "risk": "high" if is_cancer else "low",
        "explanation": detailed_explanation  
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)