import streamlit as st
import torch
import torchvision.transforms.functional as F
from PIL import Image
import os
from src import BaselineModel
from torchvision import transforms
import numpy as np

# Define the class labels
class_labels = sorted(os.listdir("/home/sohan/scratch/deepherb/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images"))

model1 = BaselineModel("resnet50", pretrained = False, num_classes = 30)
print(model1)
model2 = BaselineModel("resnet50", pretraiend = False, num_classes = 30)
model2.load_state_dict(torch.load("/home/sohan/projects/def-mpederso/sohan/leaf-classification/checkpoints/Baseline_resnet50.pth"))
print(model2)


for name, p in model1.named_parameters():
    print(name, p)

for name, p in model2.named_parameters():
    print(name, p)

exit(0)


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
        image = np.array(Image.open(uploaded_file), dtype = np.float32)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        transforms.CenterCrop(224)])
        # image_tensor = F.to_tensor(image).unsqueeze(0)
        # image = np.array(Image.open(self.df["image_path"][idx], mode = "r"), dtype = np.float32)
        image = transform(image)

        model = BaselineModel(encoder_choice, pretrained = False, num_classes = 30)
        print(model)
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
