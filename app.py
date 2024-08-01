import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import DetrImageProcessor, DetrForObjectDetection
from diffusers import StableDiffusionPipeline
from TTS.api import TTS
from deepface import DeepFace
from PIL import Image
import numpy as np
import io

@st.cache_resource
def load_tensorflow_model():
    try:
        return MobileNetV2(weights='imagenet')
    except Exception as e:
        st.error(f"Fehler beim Laden des TensorFlow-Modells: {str(e)}")
        return None

@st.cache_resource
def load_pytorch_model():
    try:
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    except Exception as e:
        st.error(f"Fehler beim Laden des PyTorch-Modells: {str(e)}")
        return None

@st.cache_resource
def load_huggingface_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Fehler beim Laden der Hugging Face Pipeline: {str(e)}")
        return None

@st.cache_resource
def load_image_generation_model():
    try:
        return StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    except Exception as e:
        st.error(f"Fehler beim Laden des Bildgenerierungsmodells: {str(e)}")
        return None

@st.cache_resource
def load_voice_cloning_model():
    try:
        return TTS("tts_models/en/ljspeech/tacotron2-DDC")
    except Exception as e:
        st.error(f"Fehler beim Laden des Voice-Cloning-Modells: {str(e)}")
        return None

@st.cache_resource
def load_text_generation_model():
    try:
        model_name = "databricks/dolly-v2-3b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Fehler beim Laden des Textgenerierungsmodells: {str(e)}")
        return None, None

@st.cache_resource
def load_object_detection_model():
    try:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        return processor, model
    except Exception as e:
        st.error(f"Fehler beim Laden des Objekterkennungsmodells: {str(e)}")
        return None, None

def generate_image(prompt, model):
    with st.spinner('Generiere Bild...'):
        image = model(prompt).images[0]
    return image

def clone_voice(text, model):
    with st.spinner('Generiere Sprachausgabe...'):
        wav = model.tts(text)
    return wav

def generate_text(prompt, model, tokenizer):
    with st.spinner('Generiere Text...'):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def detect_age(image):
    with st.spinner('Erkenne Alter...'):
        try:
            result = DeepFace.analyze(img_path=np.array(image), actions=['age'])
            return result[0]['age']
        except Exception as e:
            st.error(f"Fehler bei der Alterserkennung: {str(e)}")
            return None

def detect_objects(image, processor, model):
    with st.spinner('Erkenne Objekte...'):
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "label": model.config.id2label[label.item()],
                "score": score.item(),
                "box": box.tolist()
            })
        
        return detections

def main():
    st.set_page_config(page_title="KI-Modell-Demo", layout="wide")
    st.title("KI-Modell-Demonstrator")

    model_choice = st.sidebar.selectbox(
        "Wählen Sie ein KI-Modell:",
        (
            "TensorFlow Modell", 
            "PyTorch Modell", 
            "Hugging Face Pipeline",
            "Bildgenerierung (Stable Diffusion)",
            "Voice Cloning (TTS)",
            "Textgenerierung (Dolly)",
            "Alterserkennung",
            "Objekterkennung (DETR)"
        )
    )

    if model_choice == "TensorFlow Modell":
        model = load_tensorflow_model()
        if model:
            st.write("TensorFlow Modell geladen")
            uploaded_file = st.file_uploader("Laden Sie ein Bild hoch:")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = image.resize((224, 224))
                image = preprocess_input(np.expand_dims(np.array(image), axis=0))
                predictions = model.predict(image)
                decoded_predictions = decode_predictions(predictions, top=3)[0]
                st.write("Vorhersagen:")
                for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                    st.write(f"{i + 1}: {label} ({score:.2f})")

    elif model_choice == "PyTorch Modell":
        model = load_pytorch_model()
        if model:
            st.write("PyTorch Modell geladen")
            uploaded_file = st.file_uploader("Laden Sie ein Bild hoch:")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = image.resize((224, 224))
                image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    output = model(image)
                _, predicted = torch.max(output, 1)
                st.write(f"Vorhergesagte Klasse: {predicted.item()}")

    elif model_choice == "Hugging Face Pipeline":
        pipeline = load_huggingface_pipeline()
        if pipeline:
            st.write("Hugging Face Pipeline geladen")
            user_input = st.text_input("Geben Sie einen Text für die Sentimentanalyse ein:")
            if user_input:
                result = pipeline(user_input)[0]
                st.write(f"Sentiment: {result['label']} (Konfidenz: {result['score']:.2f})")

    elif model_choice == "Bildgenerierung (Stable Diffusion)":
        model = load_image_generation_model()
        if model:
            st.write("Stable Diffusion Modell geladen")
            prompt = st.text_input("Geben Sie eine Bildbeschreibung ein:")
            if prompt:
                image = generate_image(prompt, model)
                st.image(image)

    elif model_choice == "Voice Cloning (TTS)":
        model = load_voice_cloning_model()
        if model:
            st.write("TTS Modell geladen")
            text = st.text_input("Geben Sie den zu sprechenden Text ein:")
            if text:
                wav = clone_voice(text, model)
                st.audio(wav, format="audio/wav")

    elif model_choice == "Textgenerierung (Dolly)":
        model, tokenizer = load_text_generation_model()
        if model and tokenizer:
            st.write("Dolly Modell geladen")
            prompt = st.text_input("Geben Sie einen Prompt für die Textgenerierung ein:")
            if prompt:
                generated_text = generate_text(prompt, model, tokenizer)
                st.write(generated_text)

    elif model_choice == "Alterserkennung":
        st.write("Alterserkennung geladen")
        uploaded_file = st.file_uploader("Laden Sie ein Bild hoch:")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            age = detect_age(image)
            if age is not None:
                st.write(f"Geschätztes Alter: {age} Jahre")
            st.image(image)

    elif model_choice == "Objekterkennung (DETR)":
        processor, model = load_object_detection_model()
        if processor and model:
            st.write("DETR Modell geladen")
            uploaded_file = st.file_uploader("Laden Sie ein Bild hoch:")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                detections = detect_objects(image, processor, model)
                st.image(image)
                for detection in detections:
                    st.write(f"Objekt: {detection['label']}, Konfidenz: {detection['score']:.2f}")

if __name__ == "__main__":
    main()