import json
from typing import Dict, Tuple

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# --- Configurazione ---
MODEL_PATH = "model.pth"
CLASS_NAMES_PATH = "class_names.json"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# --- Trasformazione immagine ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --- Utils ---
def load_class_names(path: str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File {path} non trovato. Assicurati di aver completato il training.")
        st.stop()


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_s(weights=None)

    in_features = model.classifier[-1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )

    return model


@st.cache_resource
def load_model() -> Tuple[nn.Module, list]:
    class_names = load_class_names(CLASS_NAMES_PATH)

    model = build_model(len(class_names))

    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE)
        )
    except FileNotFoundError:
        st.error(f"File {MODEL_PATH} non trovato. Assicurati di aver completato il training.")
        st.stop()
    except Exception as e:
        st.error(f"Errore nel caricamento del modello: {e}")
        st.stop()

    model.to(DEVICE)
    model.eval()

    return model, class_names


def predict(
    image: Image.Image,
    model: nn.Module,
    class_names: list
) -> Tuple[str, float, Dict[str, float]]:

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    predicted_idx = int(torch.argmax(probs).item())
    predicted_class = class_names[predicted_idx]
    confidence = float(probs[predicted_idx].item())

    all_probs = {
        class_names[i]: float(probs[i].item())
        for i in range(len(class_names))
    }

    return predicted_class, confidence, all_probs


def top_k_predictions(prob_dict: Dict[str, float], k: int = 3):
    return sorted(
        prob_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]


# --- Interfaccia Utente ---
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Image Classifier")
st.markdown(
    "Carica un'immagine per ottenere una classificazione automatica basata sul modello addestrato."
)

st.divider()

model, class_names = load_model()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("Caricamento immagine")

    uploaded_file = st.file_uploader(
        "Trascina qui l'immagine o clicca per cercare",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Immagine caricata",
            use_container_width=True
        )

with col2:
    if uploaded_file is not None:
        st.subheader("Risultato")

        with st.spinner("Analisi in corso..."):
            predicted_class, confidence, probabilities = predict(
                image=image,
                model=model,
                class_names=class_names
            )

        st.markdown(f"### Classe prevista: **{predicted_class}**")
        st.progress(confidence, text=f"Confidenza: {confidence:.1%}")

        st.markdown("---")

        st.subheader("Top predizioni")

        for label, prob in top_k_predictions(probabilities, k=3):
            st.write(f"- **{label}**: {prob:.1%}")

        st.markdown("---")

        with st.expander("Mostra tutte le probabilità"):
            for label, prob in top_k_predictions(probabilities, k=len(probabilities)):
                st.write(f"- **{label}**: {prob:.1%}")

        with st.expander("Classi del modello"):
            for class_name in class_names:
                st.write(f"- {class_name}")

    else:
        st.info("👈 Carica un'immagine nella colonna di sinistra per iniziare.")