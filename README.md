# Be Cured - AI for Diabetic Retinopathy & Kidney Disease

## Introduction
**Be Cured** is an AI-powered healthcare solution for the early diagnosis of **diabetic retinopathy** and **kidney disease**. The system leverages **deep learning models** for medical imaging analysis and **clinical data processing**, enabling early detection and risk assessment.

---

## Installation & Setup

### **Prerequisites**
Ensure you have the following installed:
- **Python 3.8+**
- **Flask**
- **TensorFlow**
- **Scikit-Learn**
- **NumPy, Pandas, Matplotlib, Seaborn**

### **Installation Steps**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/gurarpitzz/BeCured.git
   cd BeCured
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application**:
   ```bash
   python app.py
   ```

4. **Access the web app**:  
   Open your browser and go to `http://127.0.0.1:5000/`

---

## Usage Guide

### **1. Upload Medical Data**
- For **Diabetic Retinopathy**: Upload a **retinal scan image**.
- For **Kidney Disease**: Upload **clinical data (CSV/Excel)**.

### **2. AI Model Analysis**
- The system uses **CNNs** for image-based disease detection.
- Clinical data is processed for **risk assessment**.

### **3. View Diagnostic Results**
- The web interface displays **disease presence, severity levels, and recommendations**.
- Users can **download a health report** in PDF format.

---

## Project Structure

```
📂 BeCured
│-- 📂 static                 # Static files (CSS, JS, Images)
│   │-- style.css             # Styling for the UI
│   │-- script.js             # (Optional) JavaScript for interactivity
│   │-- images/               # (Optional) Image assets
│
│-- 📂 templates              # HTML templates for Flask
│   │-- index.html            # Main UI page (File upload & results display)
│   │-- result.html           # Displays diagnosis results
│
│-- app.py                    # Main Flask application, handles requests and AI processing
│-- Dibetic_Retinopology.ipynb # Jupyter Notebook for Retinopathy analysis
│-- Kidney_Disease_Analysis.ipynb # Jupyter Notebook for Kidney Disease
│-- model.h5                  # Trained model for Retinopathy detection
│-- kidney_disease_model.pkl   # Trained model for Kidney Disease prediction
│-- health_report.pdf         # Sample generated health report
│-- README.md                 # Documentation
│-- requirements.txt          # Required Python packages
```

### **File Descriptions**

| **File/Folder**                  | **Description** |
|-----------------------------------|---------------|
| `app.py`                          | Main Flask application. Handles data input, model processing, and result display. |
| `templates/index.html`            | Main frontend page for uploading files and viewing results. |
| `templates/result.html`           | Displays the AI-generated diagnosis and recommendations. |
| `static/style.css`                | CSS styles for the frontend. |
| `static/script.js`                | (Optional) JavaScript for frontend interactivity. |
| `Dibetic_Retinopology.ipynb`      | Jupyter Notebook for analyzing diabetic retinopathy data. |
| `Kidney_Disease_Analysis.ipynb`   | Jupyter Notebook for kidney disease analysis. |
| `model.h5`                        | Pretrained deep learning model for retinal disease detection. |
| `kidney_disease_model.pkl`        | Machine learning model for kidney disease prediction. |
| `health_report.pdf`               | Sample output report for AI-generated results. |
| `requirements.txt`                | Contains all Python dependencies required for the project. |
| `README.md`                       | This documentation file. |

---

## Tech Stack
- **Machine Learning & AI**: TensorFlow, Scikit-Learn
- **Deep Learning**: CNNs for image classification
- **Web Framework**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Data Visualization**: Matplotlib, Seaborn

---



## License
This project is **open-source** under the **MIT License**.

---
