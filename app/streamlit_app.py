import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Cassava Leaf Disease Classifier", page_icon="🌿")

st.title("🌿 Cassava Leaf Disease Classifier")
st.write("Upload a cassava leaf image to detect disease.")

DISEASE_INFO = {
    "Cassava Bacterial Blight (CBB)": "Caused by Xanthomonas axonopodis. Symptoms include angular leaf spots and wilting.",
    "Cassava Brown Streak Disease (CBSD)": "Caused by a virus. Symptoms include brown streaks on stems and root necrosis.",
    "Cassava Green Mottle (CGM)": "Caused by a virus. Symptoms include green mottling and leaf distortion.",
    "Cassava Mosaic Disease (CMD)": "Most common cassava disease. Caused by a virus. Distinct yellow mosaic pattern on leaves.",
    "Healthy": "No disease detected. The plant appears healthy."
}

uploaded_file = st.file_uploader("Choose a cassava leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        uploaded_file.seek(0)
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": ("image.jpg", uploaded_file, "image/jpeg")}
        )

    if response.status_code == 200:
        result = response.json()

        disease = result["disease"]
        confidence = result["confidence"]
        probs = result["all_probabilities"]

        if disease == "Healthy":
            st.success(f"✅ {disease} ({confidence}% confidence)")
        else:
            st.error(f"⚠️ {disease} ({confidence}% confidence)")

        st.info(DISEASE_INFO[disease])

        st.subheader("All Class Probabilities")
        for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            st.progress(int(prob), text=f"{label}: {prob}%")
    else:
        st.error("API error. Make sure FastAPI is running.")