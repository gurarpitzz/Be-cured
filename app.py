import ast

import openai
from flask import Flask, render_template, request, request, redirect, url_for, jsonify, send_file
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import hj
app = Flask(__name__)
import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import perf_counter
from pathlib import Path
from IPython.display import Image, display, Markdown

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import google.generativeai as genai

import subprocess
def printmd(string):
    display(Markdown(string))

imageDir = Path('gaussian_filtered_images/gaussian_filtered_images')
image_path = ""
prediction = ""
import tensorflow as tf

import seaborn as sns

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
filepaths = list(imageDir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

image_df = pd.concat([filepaths, labels], axis=1)
image_df = image_df.sample(frac=1).reset_index(drop=True)

vc = image_df['Label'].value_counts()

sns.barplot(x=vc.index, y=vc, palette="rocket")

trainImages = None
valImages = None
testImages = None
your_test_images = None
your_image_df = pd.DataFrame({"Filepath": [image_path], "Label": ["unclassified"]})
train_df, test_df = train_test_split(image_df, train_size=0.9, shuffle=True, random_state=1)
import subprocess
import json
import google.generativeai as genai

genai.configure(api_key="AIzaSyBQsJsWkqn6l7SljvPmcAhEqr3rccoL2XQ")  # Replace with your Gemini API key

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def create_health_report_pdf(data, output_filename):
    """
    Creates a well-formatted PDF health report from the given data.

    Args:
        data (dict): The health report data.
        output_filename (str): The output filename for the PDF.
    """

    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=4))
    styles['Heading1'].spaceAfter = 12  # Modify existing Heading1
    styles['Heading2'].spaceAfter = 6  # Modify existing Heading2
    styles['Normal'].spaceAfter = 12  # Modify existing Normal

    # Register fonts if needed (e.g., for different languages or styles)
    try:
        pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))  # Replace 'arial.ttf' with your font file
        styles.add(ParagraphStyle(name='NormalArial', parent=styles['Normal'], fontName='Arial'))
    except FileNotFoundError:
        print("Font file not found. Using default font.")

    story = []

    # Title Page
    title_page = data.get("title_page", {}) #Handle missing title_page
    if title_page:
        story.append(Paragraph(f"<b>Patient Name:</b> {title_page.get('patient_name', 'N/A')}", styles['Heading1']))
        story.append(Paragraph(f"<b>Date:</b> {title_page.get('date', 'N/A')}", styles['Heading2']))
        story.append(Spacer(1, 0.5 * inch))

    # Summary
    summary = data.get("summary", "No summary provided.") #handle missing summary
    story.append(Paragraph("<b>Summary</b>", styles['Heading2']))
    story.append(Paragraph(summary, styles['Justify']))
    story.append(Spacer(1, 0.25 * inch))

    # Findings
    findings = data.get("findings", "").split("\n\n") #handle missing findings
    story.append(Paragraph("<b>Findings</b>", styles['Heading2']))
    for finding in findings:
        story.append(Paragraph(finding.replace("*", "•"), styles['Justify']))
    story.append(Spacer(1, 0.25 * inch))

    # Recommendations
    recommendations = data.get("recommendations", "").split("\n\n") #handle missing recommendations
    story.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
    for recommendation in recommendations:
        story.append(Paragraph(recommendation.replace("*", "•"), styles['Justify']))
    story.append(Spacer(1, 0.25 * inch))

    # Results Table
    results_table = data.get("results_table", []) #Handle missing results_table
    if results_table:
        story.append(Paragraph("<b>Results</b>", styles['Heading2']))
        table_data = [["Metric", "Value"]]
        for row in results_table:
            table_data.append([row.get("metric", "N/A"), row.get("value", "N/A")])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)

    doc.build(story)


def generate_latex_report(patient_data, model_outputs):
    # Define the prompt
    prompt = f"""
    You are a medical expert. Given the following patient data and AI model outputs, generate a structured response with the following fields:

    - title_page (object): contains patient_name (string) and date (string in YYYY-MM-DD format)
    - summary (string): Summary of the findings
    - findings (string): Detailed findings based on patient data
    - recommendations (string): Suggested medical recommendations
    - results_table (array): Each item contains 'metric' and 'value' fields for key health indicators

    **Patient Data:** {patient_data}

    **Model Outputs (Diagnosis Predictions & Key Metrics):** {model_outputs} make it extensive and detailed 
    """

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    x = response.text
    print(x)
    data = x.replace("```json", "").replace("```", "").strip()
    print(data)
    data = ast.literal_eval(data)
    print(data)
    # Example usage:
    create_health_report_pdf(data, "health_report.pdf")
    # Convert structured data to LaTeX


@app.route('/')
def index():
    return render_template('index.html')


import joblib
import os
import numpy as np
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load('kidney_disease_model.pkl')


selected_columns = ['sg', 'al', 'sc', 'hemo', 'pcv', 'rc', 'htn', 'dm', 'appet', 'pe']

model_2 = tf.keras.models.load_model("model.h5")

trainGen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )

testGen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

trainImages = trainGen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    subset='training',
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
# Reverse the class indices dictionary (ensure trainImages is loaded in your training script)
labels = {v: k for k, v in trainImages.class_indices.items()}  # Ensure this dictionary is available

import numpy as np
import tensorflow as tf
from PIL import Image
import io


def preprocess_user_image(image_file, target_size=(224, 224)):
    """
    Preprocesses a user-uploaded image for prediction.
    - Converts image to RGB
    - Resizes to MobileNetV2's expected input shape
    - Applies MobileNetV2 preprocessing
    - Ensures the shape is correct before returning
    """
    try:
        image_file.seek(0)  # Reset file pointer to the start
        img = Image.open(image_file)

        # Check if it's a valid image
        if img.format not in ["JPEG", "JPG", "PNG"]:
            raise ValueError(f"Unsupported image format: {img.format}")

        img = img.convert("RGB")  # Ensure RGB format
        img = img.resize(target_size)  # Resize to match model input
        img = np.array(img)  # Convert to NumPy array

        if img.shape != (224, 224, 3):  # Ensure correct shape
            raise ValueError(f"Unexpected image shape: {img.shape}")

        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Normalize input
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        return img

    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}"}  # Return error


def predict_user_image(image_file, model_2, labels):
    """
    Predicts the class of a user-uploaded image using a trained MobileNetV2 model.
    - Calls preprocess_user_image()
    - Ensures successful preprocessing before prediction
    - Returns predicted label and confidence score
    """
    processed_img = preprocess_user_image(image_file)

    # If preprocessing fails, return the error dictionary
    if isinstance(processed_img, dict):
        return processed_img

    try:
        prediction = model_2.predict(processed_img)  # Get probability scores
        predicted_class_idx = np.argmax(prediction, axis=1)[0]  # Get highest probability index
        predicted_label = labels.get(predicted_class_idx, "Unknown")  # Map index to label
        confidence = float(prediction[0][predicted_class_idx])  # Confidence score

        return {
            "label": predicted_label,
            "confidence": f"{confidence * 100:.2f}%"
        }

    except Exception as e:
        return {"error": f"Model prediction failed: {str(e)}"}  # Handle any prediction errors


@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Retrieve form data
        all_features = {
            "age": request.form['age'],
            "bp": request.form['bp'],
            "sg": request.form['sg'],
            "al": request.form['al'],
            "su": request.form['su'],
            "rbc": request.form['rbc'],
            "pc": request.form['pc'],
            "pcc": request.form['pcc'],
            "ba": request.form['ba'],
            "bgr": request.form['bgr'],
            "bu": request.form['bu'],
            "sc": request.form['sc'],
            "sod": request.form['sod'],
            "pot": request.form['pot'],
            "hemo": request.form['hemo'],
            "pcv": request.form['pcv'],
            "wc": request.form['wc'],
            "rc": request.form['rc'],
            "htn": request.form['htn'],
            "dm": request.form['dm'],
            "cad": request.form['cad'],
            "appet": request.form['appet'],
            "pe": request.form['pe'],
            "ane": request.form['ane']
        }

        # Filter only the selected features
        filtered_features = [float(all_features[feature]) for feature in selected_columns]

        # Convert inputs to the correct format
        processed_features = np.array(filtered_features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(processed_features)[0]
        prediction_label = "Kidney Disease Detected" if prediction == 1 else "No Kidney Disease"

        # Save uploaded image
        image_file = request.files.get('image')
        img_path = None
        if image_file and image_file.filename:
            filename = image_file.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(img_path)

        # Return prediction result
        print(prediction_label)
        result = predict_user_image(image_file, model_2, labels)
        patient_data = all_features
        model_outputs = str(result) + str(prediction_label)
        generate_latex_report(patient_data, model_outputs)

        return send_file("health_report.pdf", as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# pretrained_model = tf.keras.applications.MobileNetV2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet',
#     pooling='avg'
# )
# pretrained_model.trainable = False
#
# inputs = pretrained_model.input
#
# x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
#
# outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
#
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# history = model.fit(
#     trainImages,
#     validation_data=valImages,
#     batch_size = 32,
#     epochs=10,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=2,
#             restore_best_weights=True
#         )
#     ]
# )

#

if __name__ == '__main__':
    app.run(port=5000)

