import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Cassava Leaf Disease Classifier",
    page_icon="🌿"
)

st.title("🌿 Cassava Leaf Disease Classifier")
st.write("Upload a cassava leaf image to detect disease.")

DISEASE_INFO = {
    "Cassava Bacterial Blight (CBB)": "Caused by Xanthomonas axonopodis. Symptoms include angular leaf spots and wilting.",
    "Cassava Brown Streak Disease (CBSD)": "Caused by a virus. Symptoms include brown streaks on stems and root necrosis.",
    "Cassava Green Mottle (CGM)": "Caused by a virus. Symptoms include green mottling and leaf distortion.",
    "Cassava Mosaic Disease (CMD)": "Most common cassava disease. Caused by a virus. Distinct yellow mosaic pattern on leaves.",
    "Healthy": "No disease detected. The plant appears healthy."
}

LABELS = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model():
    with st.spinner("Downloading model... (only first time)"):
        model_path = hf_hub_download(
            repo_id="aryanparija/cassava-leaf-disease-efficientnet-b0",
            filename="efficientnet_b0_85.pth"
        )

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 5)
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


model = load_model()

uploaded_file = st.file_uploader(
    "Choose a cassava leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Analyzing..."):

        image = image.convert("RGB")

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_class = probabilities.argmax().item()

        disease = LABELS[predicted_class]

        confidence = round(
            probabilities[predicted_class].item() * 100,
            2
        )

        probs = {
            LABELS[i]: round(probabilities[i].item() * 100, 2)
            for i in range(5)
        }

    if disease == "Healthy":
        st.success(f"✅ {disease} ({confidence}% confidence)")
    else:
        st.error(f"⚠️ {disease} ({confidence}% confidence)")

    st.info(DISEASE_INFO[disease])

    st.subheader("All Class Probabilities")

    for label, prob in sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        st.progress(
            int(prob),
            text=f"{label}: {prob}%"
        )