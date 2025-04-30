---
layout: post
title:  "Part 2: Simple FastAPI App for MRI Brain Cancer Classification with Torch Models"
date:   2025-04-20 15:33:15 +0200
categories: jekyll update
---

# Building an MRI Brain Cancer Classification Web App with FastAPI and PyTorch

In this blog post, I describe mainly the FastAPI part of the web application that classifies MRI brain scans.   
I am using **FastAPI** for the web interface and **PyTorch** for the deep learning model.  

## Key Features

- The app allows users to upload an MRI image, which is displayed on the same page along with the prediction result.
- The app uses a custom PyTorch model (`BCC_Model`) to classify the uploaded images. The model is loaded and evaluated in the backend. It should be trivial to exchange the model with every other deep-learning and especially PyTorch model.
- The frontend dynamically updates the page to show the uploaded image and prediction result without reloading the page.
- Some metadata and the prediction are automaticallz ingested into PostgreSQL DB.
- Display an overview over all uploaded cases and prediction.

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

### Handling File Uploads and predictions
The `/upload/` endpoint processes the uploaded image. We read the uploaded file and convert it to an RGB image.
We also need to use the same transforms as in the model training in the previous [post](). 
```python
@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"File uploaded: {file.filename}")
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        return {"error": "Invalid image file"}

    with torch.no_grad():
        output = model(transformed_image)
        predicted_class_idx = torch.argmax(output, dim=1).item()

    predicted_class_name = CLASS_MAP.get(predicted_class_idx, "Unknown")
```

### Add Prediction to DB
We also have PostgreSQL DB in this docker environment to save the predictions for each upload. In the DB I created a SQLAlchemy CaseModel:
```python
class CaseModel(Base):
    __tablename__ = "case_model"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now())
    prediction = Column(String, nullable=False)
```

After making a prediction we ingest the result into our table:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

new_case = CaseModel(prediction=predicted_class_name)
db.add(new_case)
db.commit()
db.refresh(new_case)
```

The cases can be viewed in the cases view, which just queries all cases from the DB and shows them in a HTML table:

FastAPI:
```python
@app.get("/cases/", response_class=HTMLResponse)
def get_cases(request: Request, db: Session = Depends(get_db)):
    cases = db.query(CaseModel).all()
    return templates.TemplateResponse("cases.html", {"request": request, "cases": cases})
```

HTML Template:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Uploaded Cases</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>Uploaded Cases</h1>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Created At</th>
                <th>Prediction</th>
            </tr>
        </thead>
        <tbody>
            {% for case in cases %}
            <tr>
                <td>{{ case.id }}</td>
                <td>{{ case.created_at }}</td>
                <td>{{ case.prediction }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <a href="/">Back to Home</a>
</body>
</html>
```

### Further Improvements
- A view for a specific case: click on a case in the cases view to display the uploaded image with the prediction
- Diagnostician's opinion: Add the option that the user can confirm or alter the prediction of the model
- SECURITY!!!! Obviously we don't have any login or user accounts here. Everyone can access all the cases and make Uploads