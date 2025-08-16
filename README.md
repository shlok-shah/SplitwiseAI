
# Splitwise AI

## Introduction 📖

Splitwise-AI is a smart web app that automates bill splitting using image processing, OCR, and AI.
Built with Streamlit, it processes bill photos, extracts key details, and integrates directly with the Splitwise API to record and split expenses among group members, uses Google Gemini for interpreting bill data, offering faster and more accurate extractions.

## Installation 🛠️

1. Clone the repository:
   ```
   git clone https://github.com/your-username/splitwisegpt-vision.git
   ```
2. Navigate to the project directory:
   ```
   cd splitwisegpt-vision
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```


## Configuration 🔧

Before running the application, you need to set up your environment variables. Use the `example.env` file as a template:

1. Rename `example.env` to `.env`.
2. Add your specific keys for the Splitwise API and other necessary configurations as shown in the `.env` file.

## Usage 🖥️

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Upload a bill image in the supported formats (PNG, JPG, JPEG).
3. Select the person who paid the bill.
4. View the extracted bill details and splits.

## Technologies Used 🧰

- **Streamlit**: For creating the web application interface.
- **Pandas & NumPy**: For data manipulation and numerical computations.
- **OpenCV**: For image processing tasks.
- **Pytesseract**: For optical character recognition (OCR).
- **Google Gemini** – AI-powered bill interpretation
