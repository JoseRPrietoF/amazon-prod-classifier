import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# API Configuration
API_URL = "http://api:8000"  # Replace with your FastAPI URL

# App Title and Branding
st.set_page_config(
    page_title="Product Category Predictor",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar Branding
with st.sidebar:
    st.image("src/streamlit_app/logo.webp", use_container_width=True)
    st.markdown(
        """
        **Product Category Predictor**
        """
    )
    st.markdown(
        "This tool predicts the main category of a product based on its description."
    )
    st.markdown("---")

# Main Header
st.title("🛒 Product Category Predictor")
st.markdown(
    "Choose a product from the dropdown or manually enter details to predict its category."
)


# Fetch items from the API
@st.cache_data
def fetch_sample_data(num_items: int = 1000):
    """
    Fetch a list of product samples from the API.

    Args:
        num_items (int): Number of items to fetch.

    Returns:
        List[dict]: List of product data dictionaries.
    """
    try:
        response = requests.post(f"{API_URL}/get_data", json={"num_items": num_items})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch sample data: {e}")
        return []


# Helper to fetch and display images
def display_images(image_links):
    """
    Fetch and display images from a list of image URLs in a row with small, consistent sizes.

    Args:
        image_links (list): List of image URLs.
    """
    if image_links:
        st.markdown("### Product Images:")
        # Create a row of images using columns
        for idx in range(0, len(image_links), 3):  # Group images in rows of 3
            cols = st.columns(3)  # 3 images per row
            for col, link in zip(cols, image_links[idx : idx + 3]):
                try:
                    # Fetch the image from the URL
                    response = requests.get(link)
                    response.raise_for_status()

                    # Open the image
                    image = Image.open(BytesIO(response.content))

                    # Display the resized image in the column
                    with col:
                        st.image(image, use_container_width=False)
                except Exception as e:
                    with col:
                        st.error(f"Failed to load image: {e}")


# Load sample data
sample_data = fetch_sample_data()

# Create dropdown for selecting a product
if sample_data:
    sample_options = {
        f"{item['title']} {'📷' if item.get('image') else ''}": item
        for item in sample_data
    }
    selected_title = st.selectbox(
        "Select a sample product:",
        ["Select a product..."] + list(sample_options.keys()),
    )

    if selected_title != "Select a product...":
        selected_product = sample_options[selected_title]

    else:
        selected_product = None
else:
    st.error("No sample data available. Please try again later.")
    selected_product = None

# Input Form
with st.form("product_form"):
    # Render dynamic input fields based on selected_product
    if selected_product:
        st.markdown("### Edit Product Details")
        inputs = {}
        avoid = ["image", "Image", "images", "Images"]
        for key, value in selected_product.items():
            if (
                isinstance(value, list) and key not in avoid
            ):  # Handle lists as text areas
                inputs[key] = st.text_area(
                    key.replace("_", " ").capitalize(), value="\n".join(value)
                )
            elif key not in avoid:  # Skip the "image" field
                inputs[key] = st.text_input(
                    key.replace("_", " ").capitalize(), value=value
                )
    else:
        st.info("No product selected. Please choose a product.")

    submitted = st.form_submit_button("Predict Category")
    # Display images if available
    if selected_product:
        if "image" in selected_product and selected_product["image"]:
            print(selected_product["image"])
            display_images(selected_product["image"])

# On Form Submission
if submitted:
    with st.spinner("Predicting the product category..."):
        # Prepare the payload
        payload = inputs

        try:
            # Send request to the FastAPI prediction endpoint
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()

            # Parse response
            result = response.json()
            predicted_category = result.get("main_cat", "Unknown Category")

            # Display the result
            st.success(f"Predicted Category: **{predicted_category}**")
            # st.balloons()

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the API. Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred. Error: {e}")

# Footer
st.markdown("---")
