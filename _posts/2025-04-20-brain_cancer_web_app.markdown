---
layout: post
title:  "Simple FastAPI App for MRI Brain Cancer Classification with Torch Models"
date:   2025-04-20 15:33:15 +0200
categories: jekyll update
---

# Building an MRI Brain Cancer Classification Web App with FastAPI and PyTorch

In this blog post, we’ll walk through the creation of a web application that classifies MRI brain scans into three categories: **Glioma**, **Meningioma**, and **Pituitary**.   
This project uses **FastAPI** for the web interface and **PyTorch** for the deep learning model.  

This project is ideal for demonstrating how to integrate machine learning models into a simple web application.

## Key Features

### 1. Upload and Display Images
The app allows users to upload an MRI image, which is displayed on the same page along with the prediction result.

### 2. PyTorch Model Integration
The app uses a custom PyTorch model (`BCC_Model`) to classify the uploaded images. The model is loaded and evaluated in the backend. It should be trivial to exchange the model with every other deep-learning and especially PyTorch model.

### 3. Dynamic Frontend
The frontend dynamically updates the page to show the uploaded image and prediction result without reloading the page.

### 4. Case view
Display an overview over all uploaded cases and prediction.

## Project Structure

```
dl_web_app/
├── main.py                     # FastAPI application
├── utils/
│   ├── utils.py            # Helper functions (e.g., image transforms, predictions)
│   ├── model.py            # Custom PyTorch model definition
├── model/
│   └── bcc_model.pth       # Pre-trained PyTorch model
├── templates/
│   └── index.html          # HTML template for the index and upload interface
│   └── cases.html          # HTML template for the case interface
├── static/
│   └── style.css           # CSS for styling the web interface
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata and dependencies
├── Dockerfile              # Docker configuration
├── README.md               # Project documentation
```

## Setting Up the Project
I would recommend to just use Docker to run the app:

1. Build the Docker image:
   ```bash
   docker build -t mri-classification-app .
   ```

2. Run the Docker container:
   ```bash
   docker run -d -p 8000:8000 --name mri-classification-container mri-classification-app
   ```

3. Access the app at:
   ```
   http://127.0.0.1:8000
   ```

## Code Highlights

### Handling File Uploads
The `/upload/` endpoint processes the uploaded image:
```python
@app.post("/upload/")
async def upload_image(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    transformed_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(transformed_image)
        predicted_class_idx = torch.argmax(output, dim=1).item()

    predicted_class_name = CLASS_MAP.get(predicted_class_idx, "Unknown")
    return {"predicted_class_name": predicted_class_name}
```
