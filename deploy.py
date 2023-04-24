import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import os
from src import BaselineModel

# Define the class labels
class_labels = os.listdir("/home/sohan/scratch/deepherb/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images")

# Define the Streamlit app
def app():
    st.title("A Robust Class Incremental Computer Vision System for Medicinal Plants Classification Using Knowledge Distillation")

    # Add file uploader for the leaf image
    uploaded_file = st.file_uploader('Upload an image of a leaf', type=['jpg', 'png'])

    # Add a drop down to select the model
    model_choice = st.selectbox('Select the model to use', ['Baseline', 'Incremental model without teacher', 'Incremental model with teacher'])
    encoder_choice = st.selectbox('Select the encoder type', ['resnet50', 'inceptionv3', 'vgg19', 'densenet'])

    if uploaded_file is not None:
        # Read the image file and convert it to a PyTorch tensor
        image = Image.open(uploaded_file)
        image_tensor = F.to_tensor(image).unsqueeze(0)

        model = BaselineModel(encoder_choice, pretrained = False, num_classes = 30)

        # Select the appropriate model
        if model_choice == 'Baseline':
            model_path = f"./checkpoints/Baseline_{encoder_choice}"
        elif model_choice == 'Incremental model without teacher':
            model_path = f"./checkpoints/No_teacher_{encoder_choice}"
        else:
            model_path = f"./checkpoints/With_teacher_{encoder_choice}"

        model.load_state_dict(torch.load(model_path))

        # Make a prediction using the selected model
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1).item()

        # Display the leaf image and its predicted class
        st.image(image, caption=class_labels[prediction], use_column_width = True)

if __name__ == "__main__":
    app()
