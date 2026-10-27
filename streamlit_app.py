import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="NeuroArt AI",
    page_icon="🎨",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

ENCODER_PATH = BASE_DIR / "vgg_normalised.pth"
DECODER_PATH = (
    BASE_DIR / "experiment" / "trial2" / "decoder_2.pth"
)

DEVICE = torch.device("cpu")

IMAGE_SIZE = 512

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# -------------------------
# Load models
# -------------------------
@st.cache_resource
def load_models():
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Encoder checkpoint missing: {ENCODER_PATH}"
        )

    if not DECODER_PATH.exists():
        raise FileNotFoundError(
            f"Decoder checkpoint missing: {DECODER_PATH}"
        )

    encoder = VGGEncoder(str(ENCODER_PATH)).to(DEVICE)
    decoder = Decoder().to(DEVICE)

    decoder_state = torch.load(
        str(DECODER_PATH),
        map_location="cpu",
        weights_only=True
    )

    decoder.load_state_dict(decoder_state)

    encoder.eval()
    decoder.eval()

    return encoder, decoder


# -------------------------
# Neural style transfer
# -------------------------
def transfer_style(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha
):
    content_tensor = transform(
        content_image
    ).unsqueeze(0).to(DEVICE)

    style_tensor = transform(
        style_image
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        content_features = encoder(
            content_tensor,
            is_test=True
        )

        style_features = encoder(
            style_tensor,
            is_test=True
        )

        target_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        target_features = (
            alpha * target_features
            + (1 - alpha) * content_features
        )

        output = decoder(target_features)

    output = output.squeeze(0).cpu().clamp(0, 1)

    return transforms.ToPILImage()(output)


# -------------------------
# Streamlit interface
# -------------------------
st.title(" NeuroArt AI")
st.subheader("Neural Style Transfer")

st.write(
    "Transform your photograph using the artistic style "
    "of another image."
)

st.info(
    "Upload one content image and one style image "
    "to create your artwork."
)

with st.sidebar:
    st.header(" Settings")

    alpha = st.slider(
        "Style strength",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05
    )

    st.caption(
        "0 = original content features; "
        "1 = full style-transfer features."
    )


col1, col2 = st.columns(2)

with col1:
    content_file = st.file_uploader(
        " Upload Content Image",
        type=["jpg", "jpeg", "png"],
        key="content_image"
    )

with col2:
    style_file = st.file_uploader(
        "Upload Style Image",
        type=["jpg", "jpeg", "png"],
        key="style_image"
    )


content_image = None
style_image = None

if content_file is not None:
    content_image = Image.open(
        content_file
    ).convert("RGB")

    with col1:
        st.image(
            content_image,
            caption="Content Image",
            use_container_width=True
        )

if style_file is not None:
    style_image = Image.open(
        style_file
    ).convert("RGB")

    with col2:
        st.image(
            style_image,
            caption="Style Reference",
            use_container_width=True
        )


if st.button(" Generate Artwork", type="primary"):

    if content_image is None or style_image is None:
        st.warning(
            "Please upload both content and style images."
        )

    else:
        try:
            with st.spinner(
                "Loading models and generating artwork..."
            ):
                encoder, decoder = load_models()

                result = transfer_style(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha
                )

            st.success("Artwork generated!")

            st.subheader("Your NeuroArt Result")

            st.image(
                result,
                caption="Generated Artwork",
                use_container_width=True
            )

            image_buffer = io.BytesIO()
            result.save(image_buffer, format="PNG")

            st.download_button(
                label=" Download Artwork",
                data=image_buffer.getvalue(),
                file_name="neuroart_result.png",
                mime="image/png"
            )

        except Exception as error:
            st.error(
                "Could not generate artwork. "
                "Please check the application logs."
            )

            st.code(str(error))
