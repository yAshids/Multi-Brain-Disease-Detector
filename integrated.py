from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
import sys
from skimage.transform import resize
from scipy.ndimage import gaussian_filter1d
import plot as p
import smtplib
import requests

from secret import GEMINI_API_KEY
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)


import streamlit as st

st.title("Multi-Disease Brain Detector")
# Your app code here, e.g. image upload, model prediction etc.


app = Flask(__name__)
app.secret_key = 'supersecretkey'

def init_db():
    conn = sqlite3.connect('hospital.db', timeout=10.0)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS admins (
                        admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY, 
                        username TEXT UNIQUE, 
                        password TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        age INTEGER CHECK(age > 0),
                        dob TEXT NOT NULL,
                        phone TEXT UNIQUE NOT NULL,
                        gender TEXT CHECK(gender IN ('Male', 'Female', 'Other')),
                        email TEXT UNIQUE NOT NULL,
                        medical_issues TEXT,
                        added_by INTEGER,
                        FOREIGN KEY (added_by) REFERENCES users(id))''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS appointments (
                        appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        appointment_date TEXT NOT NULL,
                        appointment_time TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        status TEXT DEFAULT 'Pending')''')


    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", tables)

    conn.commit()
    conn.close()
    print("db initialized successfully")

init_db()


ALZ_MODEL_1_PATH = r"D:\DO NOT DELETE FOLDER\DESKTOP\Project\final_model_alzeimer_MobileNetV2.h5"
ALZ_MODEL_2_PATH = r"D:\DO NOT DELETE FOLDER\DESKTOP\Project\final_model_alzeimer_InceptionV3.h5"
STROKE_MODEL_1_PATH = r"D:\DO NOT DELETE FOLDER\DESKTOP\Project\final_model.h5"
STROKE_MODEL_2_PATH = r"D:\DO NOT DELETE FOLDER\DESKTOP\Project\final_inception_model.h5"
tumor_model_1 = r"D:\DO NOT DELETE FOLDER\DESKTOP\Project\brain_tumor_mobilenet.h5"
tumor_model_2 = r"D:\DO NOT DELETE FOLDER\DESKTOP\Project\brain_tumor_inception.h5"
HEALTHY_BRAIN_PATH = r"D:\DO NOT DELETE FOLDER\DESKTOP\Project\normal_brain.jpeg"

alz_model_1 = tf.keras.models.load_model(ALZ_MODEL_1_PATH)
alz_model_2 = tf.keras.models.load_model(ALZ_MODEL_2_PATH)
tumor_model_1 = tf.keras.models.load_model(tumor_model_1)
tumor_model_2 = tf.keras.models.load_model(tumor_model_2)
stroke_model_1 = tf.keras.models.load_model(STROKE_MODEL_1_PATH)
stroke_model_2 = tf.keras.models.load_model(STROKE_MODEL_2_PATH)


# Load Stroke Model 1
try:
    stroke_model_1 = tf.keras.models.load_model(STROKE_MODEL_1_PATH, compile=False)
    stroke_model_1.save("models/brain_stroke_model_1_fixed.h5")
    stroke_model_1 = tf.keras.models.load_model("models/brain_stroke_model_1_fixed.h5")
except Exception as e:
    print(f"Error loading Stroke Model 1: {e}")
    stroke_model_1 = None

# Load Stroke Model 2
try:
    stroke_model_2 = tf.keras.models.load_model(STROKE_MODEL_2_PATH, compile=False)
    stroke_model_2.save("models/brain_stroke_model_2_fixed.h5")
    stroke_model_2 = tf.keras.models.load_model("models/brain_stroke_model_2_fixed.h5")
except Exception as e:
    print(f"Error loading Stroke Model 2: {e}")
    stroke_model_2 = None


alz_classes = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
stroke_classes = ["Stroke", "Non-Stroke"]
tumor_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def preprocess_image(image_path, target_size):

    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('index2'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['username'] = username
            session['user_id'] = user[0]
            return redirect(url_for('index2'))
        else:
            flash("Invalid username or password", "danger")
    return render_template('login.html')



# Admin registration route
@app.route('/admin_register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match. Please try again.", "danger")
            return redirect(url_for('admin_register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO admins (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            flash("Admin registration successful! Please log in.", "success")
            return redirect(url_for('admin_login'))
        except sqlite3.IntegrityError:
            flash("Username already exists. Please choose a different one.", "danger")

        conn.close()

    return render_template('admin_register.html')


@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admins WHERE username=?", (username,))
        admin = cursor.fetchone()
        conn.close()
        
        if admin and check_password_hash(admin[2], password):
            session['admin_username'] = username
            print("Logged in, session:", dict(session))
            return redirect(url_for('admin_dashboard'))  
        else:
            flash("Invalid admin credentials", "danger")
    
    return render_template('admin_login.html')

@app.route('/admin_dashboard', methods=['GET'])
def admin_dashboard():
    if 'admin_username' not in session:
        print("Session missing, redirecting to login")
        return redirect(url_for('admin_login'))

    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients")
    patients = cursor.fetchall()
    conn.close()

    return render_template('admin_dashboard.html', patients=patients)



# Add patient route
@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        dob = request.form['dob']
        phone = request.form['phone']
        gender = request.form['gender']
        email = request.form['email']
        medical_issues = request.form['medical_issues']

        # Ensure user is logged in
        if 'user_id' not in session:
            flash("Please log in to add a patient.", "danger")
            return redirect(url_for('login'))

        added_by = session['user_id']  # Get logged-in user's ID

        # Insert patient into database
        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO patients (name, age, dob, phone, gender, email, medical_issues, added_by)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                       (name, age, dob, phone, gender, email, medical_issues, added_by))
        conn.commit()
        conn.close()

        flash("Patient details added successfully.", "success")
        return redirect(url_for('login'))

    return render_template('add_patient.html')



@app.route('/index2')
def index2():
    return render_template('index2.html')



# User registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('add_patient'))  # Redirect to add_patient after registration
        except sqlite3.IntegrityError:
            flash("Username already exists. Please choose a different one.", "danger")
        conn.close()

    return render_template('register.html')




@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files or 'disease' not in request.form:
        return jsonify({"error": "Invalid request. Please provide an image and select a disease."}), 400

    file = request.files['file']
    disease = request.form['disease']

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Store the disease type in session
    session['latest_disease_detected'] = disease

    if disease == "alzheimers":
        img_array = preprocess_image(file_path, (256, 256))

        pred_1 = alz_model_1.predict(img_array)
        pred_2 = alz_model_2.predict(img_array)

        print(pred_1,pred_2)
        #p.plot(pred_1,pred_2)
    
        avg_pred = (pred_1 + pred_2) / 2
        predicted_class_1 = np.argmax(pred_1, axis=1)[0]
        predicted_class_2 = np.argmax(pred_2, axis=1)[0]
        predicted_class_avg = np.argmax(avg_pred, axis=1)[0]
        disease_detected = alz_classes[predicted_class_avg] != "Non Demented"

        result = {
            "Model 1 Prediction": alz_classes[predicted_class_1],
            "Model 2 Prediction": alz_classes[predicted_class_2],
            "Final Prediction (Averaged)": alz_classes[predicted_class_avg]
        }

    elif disease == "stroke":
        img_array = preprocess_image(file_path, (256, 256))
    
        if stroke_model_1 is None or stroke_model_2 is None:
            return jsonify({"error": "One or both Stroke models could not be loaded. Please check the model files."}), 500

        pred_1 = stroke_model_1.predict(img_array)
        pred_2 = stroke_model_2.predict(img_array)

        print(pred_1,pred_2)
        #p.plot(pred_1,pred_2)
        
        avg_pred = (pred_1 + pred_2) / 2
        predicted_class_1 = int(pred_1[0][0] > 0.5)
        predicted_class_2 = int(pred_2[0][0] > 0.5)
        predicted_class_avg = int(avg_pred[0][0] > 0.5)

        disease_detected = stroke_classes[predicted_class_avg] == "Stroke"

        result = {
            "Model 1 Prediction": stroke_classes[predicted_class_1],
            "Model 2 Prediction": stroke_classes[predicted_class_2],
            "Final Prediction (Averaged)": stroke_classes[predicted_class_avg]
        }

    elif disease == "tumor":
        img_array = preprocess_image(file_path, (256, 256))

        if tumor_model_1 is None or tumor_model_2 is None:
            return jsonify({"error": "One or both tumor models could not be loaded. Please check the model files."}), 500

        pred_1 = tumor_model_1.predict(img_array)
        pred_2 = tumor_model_2.predict(img_array)
        print(pred_1,pred_2)
        p.plot(pred_1,pred_2)
        avg_pred = (pred_1 + pred_2) / 2
        predicted_class_1 = np.argmax(pred_1, axis=1)[0]
        predicted_class_2 = np.argmax(pred_2, axis=1)[0]
        predicted_class_avg = np.argmax(avg_pred, axis=1)[0]
        disease_detected = tumor_classes[predicted_class_avg] != "No Tumor"

        result = {
            "Model 1 Prediction": tumor_classes[predicted_class_1],
            "Model 2 Prediction": tumor_classes[predicted_class_2],
            "Final Prediction (Averaged)": tumor_classes[predicted_class_avg]
        }

    else:
        return jsonify({"error": "Invalid disease type"}), 400

    # Store the prediction result in the session
    session['latest_result'] = result

    # After saving the uploaded image and before rendering result.html
    plot_path = os.path.join('static', 'comparison_plot.png')
    compare_brain_images_diagonal(plot_path, disease_detected)

    return render_template("result.html", result=result, image=file_path, disease=disease, plot_url=url_for('static', filename='comparison_plot.png'))


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import qrcode
import io
from reportlab.lib.utils import ImageReader

# Helper to generate QR code image in memory
def generate_qr_code(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    buf = io.BytesIO()
    img.save(buf)
    buf.seek(0)
    return buf

def draw_header(c, width, height, HEADER_COLOR):
    # Header background
    c.setFillColor(HEADER_COLOR)
    c.rect(0, height - 80, width, 80, fill=1, stroke=0)

    # Logo (left corner)
    logo_x = 50
    logo_y = height - 75
    logo_width = 80
    logo_height = 50
    try:
        c.drawImage("logo.jpeg", logo_x, logo_y, width=logo_width, height=logo_height, mask='auto')
    except Exception:
        pass

    # Hospital info (to the right of logo)
    text_x = logo_x + logo_width + 20
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(text_x, height - 40, "Jyothy Hospital")
    c.setFont("Helvetica", 12)
    c.drawString(text_x, height - 60, "Tataguni Main Road, Bengaluru, Karnataka")
    c.drawString(text_x, height - 75, "Phone: 123-456-7890 | Email: info@jyothyhospital.com")
    c.setFont("Helvetica", 10)
    c.drawRightString(width - 40, height - 40, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}")

def draw_footer(c, width, LINE_COLOR):
    c.setStrokeColor(LINE_COLOR)
    c.line(40, 60, width - 40, 60)
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(50, 50, "Confidential - For authorized use only")
    c.drawRightString(width - 40, 50, f"Page {c.getPageNumber()}")
    c.setFillColor(colors.black)

def draw_watermark(c):
    c.saveState()
    c.setFont("Helvetica-Bold", 40)
    c.setFillColorRGB(0.9, 0.9, 0.9, alpha=0.2)
    c.rotate(30)
    c.drawString(150, 200, "CONFIDENTIAL")
    c.restoreState()

def generate_pdf_report(patient_data_list, file_path):
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    # Colors and styles
    HEADER_COLOR = colors.HexColor("#0b6767")
    SUBHEADER_COLOR = colors.HexColor("#0b6767")
    LINE_COLOR = colors.HexColor("#CCCCCC")

    # Draw header
    draw_header(c, width, height, HEADER_COLOR)

    # Report Title
    c.setFillColor(SUBHEADER_COLOR)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 110, "PATIENT DETECTION REPORT")
    c.setFillColor(colors.black)

    y_position = height - 140

    for idx, patient_data in enumerate(patient_data_list):
        # Section divider
        c.setStrokeColor(LINE_COLOR)
        c.setLineWidth(1)
        c.line(40, y_position + 10, width - 40, y_position + 10)

        # Patient heading
        c.setFont("Helvetica-Bold", 13)
        c.setFillColor(SUBHEADER_COLOR)
        c.drawString(50, y_position, f"Patient {idx + 1}: {patient_data['Name']}")
        y_position -= 18
        c.setFillColor(colors.black)

        c.setFont("Helvetica", 11)
        for key, value in patient_data.items():
            if key != 'Name':
                # Key in bold, value normal
                c.setFont("Helvetica-Bold", 11)
                c.drawString(70, y_position, f"{key}:")
                c.setFont("Helvetica", 11)
                # Handle multi-line values
                lines = str(value).split('\n')
                for i, line in enumerate(lines):
                    c.drawString(160, y_position, line)
                    if i < len(lines) - 1:
                        y_position -= 15
                y_position -= 18

        # Add QR code for patient link or info
        qr_data = f"https://www.sakraworldhospital.com/book-health-check-up/patient?id={idx + 1}"
        qr_img_buf = generate_qr_code(qr_data)
        c.drawImage(ImageReader(qr_img_buf), width - 120, y_position + 10, width=60, height=60)

        y_position -= 80  # space for QR code

        y_position -= 10  # extra space between patients

        # Page break if needed
        if y_position < 140:
            draw_footer(c, width, LINE_COLOR)
            draw_watermark(c)
            c.showPage()
            draw_header(c, width, height, HEADER_COLOR)
            c.setFillColor(SUBHEADER_COLOR)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 110, "PATIENT DETECTION REPORT")
            c.setFillColor(colors.black)
            y_position = height - 140

    # --- Report Summary ---
    y_position -= 40
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(SUBHEADER_COLOR)
    c.drawString(50, y_position, "Report Summary")
    c.setFillColor(colors.black)
    y_position -= 18
    c.setFont("Helvetica", 11)
    summary_text = "Patient record have been reviewed. Please contact the hospital for further details or clarifications."
    c.drawString(70, y_position, summary_text)
    y_position -= 30

    # --- Signature/Stamp ---
    try:
        c.drawImage("signature.png", 70, y_position - 50, width=120, height=40, mask='auto')
        c.setFont("Helvetica", 10)
        c.drawString(70, y_position - 60, "Dr. John Doe")
        c.drawString(70, y_position - 72, "Chief Medical Officer")
    except Exception:
        pass  # If image not found, skip

    # Footer and watermark on last page
    draw_footer(c, width, LINE_COLOR)
    draw_watermark(c)

    c.save()





@app.route('/download_report')
def download_report():
    # Ensure user is logged in
    if 'user_id' not in session:
        flash("You must be logged in to download your report.", "danger")
        return redirect(url_for('login'))

    user_id = session['user_id']

    # Fetch only this user's patients
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE added_by=?", (user_id,))
    patients = cursor.fetchall()
    conn.close()

    if not patients:
        flash("No patients found for this user.", "warning")
        return redirect(url_for('add_patient'))

    patient_data_list = []
    for patient in patients:
        data = {
            "Name": patient[1],
            "Age": patient[2],
            "Date of Birth": patient[3],
            "Phone": patient[4],
            "Gender": patient[5],
            "Email": patient[6],
            "Medical Issues": patient[7],
        }

        # Optionally add detection result if available in session
        if 'latest_result' in session and 'latest_disease_detected' in session:
            result_text = f"Disease: {session['latest_disease_detected'].capitalize()}\n"
            if isinstance(session['latest_result'], dict):
                for key, val in session['latest_result'].items():
                    result_text += f"{key}: {val}\n"
            else:
                result_text += str(session['latest_result'])
            data["Detection Result"] = result_text.strip()

        patient_data_list.append(data)

    file_path = "patient_report.pdf"
    generate_pdf_report(patient_data_list, file_path)
    return send_file(file_path, as_attachment=True)



    
def get_bot_response(user_message):
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(history=[])
    # Add medical context to the prompt
    medical_prompt = (
        "You are a medical assistant. "
        "Answer the following question with medically accurate, concise, and specific information. "
        "If unsure, advise consulting a healthcare professional.\n\n"
        f"User: {user_message}"
    )
    response = chat.send_message(medical_prompt)
    return response.text



@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_message = data.get('message', '')
    bot_response = get_bot_response(user_message)
    return jsonify({"reply": bot_response})




def compare_brain_images_diagonal(output_path, disease_detected):
    N = 256
    x = np.linspace(0, 1, N)
    healthy_line = x  # y = x (diagonal)
    user_line = x if not disease_detected else x**2  # Curve if disease detected

    plt.figure(figsize=(8, 6))
    plt.plot(x, healthy_line, label='Healthy Reference (Diagonal)', color='green', linewidth=2)
    plt.plot(x, user_line, label='User Image', color='orange', linewidth=2)
    plt.legend()
    plt.title('Model-Based Brain Image Comparison')
    plt.xlabel('Normalized Position')
    plt.ylabel('Normalized Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@app.route('/result')
def result():
    # ... your code to get detection results, etc.
    plot_url = url_for('static', filename='comparison_plot.png')
    return render_template('result.html', result=result, plot_url=plot_url)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            user_img_path = os.path.join('static', 'user_upload.jpg')
            file.save(user_img_path)
            # Generate comparison plot
            plot_path = os.path.join('static', 'comparison_plot.png')
            compare_brain_images_diagonal(HEALTHY_BRAIN_PATH, user_img_path, plot_path)
            # Pass plot_url to result.html
            return render_template('result.html', plot_url=url_for('static', filename='comparison_plot.png'))
    return render_template('index2.html')

@app.route('/show_comparison', methods=['POST'])
def show_comparison():
    disease_detected = session.get('latest_disease_detected', False)
    plot_path = os.path.join('static', 'comparison_plot.png')
    compare_brain_images_diagonal(plot_path, disease_detected)
    return render_template('comparison.html', plot_url=url_for('static', filename='comparison_plot.png'))


@app.route('/admin_logout')
def admin_logout():
    session.clear() 
    return redirect(url_for('admin_login'))


@app.route('/logout')
def logout():
    session.clear()  # or session.pop('user_id', None)
    return redirect(url_for('login'))  # Redirect to your login page


@app.route('/about')
def about():
    return render_template('about.html')



TELEGRAM_BOT_TOKEN = '7681869034:AAH8JzBZqNzrtnegDERJfUoWJU4dshbVmP8'
TELEGRAM_CHAT_ID = '921645787'

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        # Format the message
        telegram_message = (
            f"📩 *New Contact Message*\n"
            f"*Name:* {name}\n"
            f"*Email:* {email}\n"
            f"*Subject:* {subject}\n"
            f"*Message:* {message}"
        )

        # Send to Telegram
        send_message_to_telegram(telegram_message)

        flash("Message sent successfully!", "success")
        return redirect(url_for('contact'))

    return render_template('contact.html')

def send_message_to_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    requests.post(url, data=payload)


@app.route('/appointments')
def appointments():
    # Render the appointment form page
    return render_template('appointment.html')


@app.route('/schedule_appointment', methods=['POST'])
def schedule_appointment():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        appointment_date = request.form['appointment_date']
        appointment_time = request.form['appointment_time']
        reason = request.form['reason']
        
        # Insert the appointment data into the database
        conn = sqlite3.connect('hospital.db')  # Change to your actual database
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO appointments (name, appointment_date, appointment_time, reason)
                          VALUES (?, ?, ?, ?)''', 
                       (name, appointment_date, appointment_time, reason))
        conn.commit()
        conn.close()
        
        # Redirect to the admin dashboard after scheduling the appointment
        return redirect(url_for('appointments'))
    

@app.route('/doctor/appointments')
def doctor_appointments():
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM appointments")
    appointments = cursor.fetchall()
    conn.close()
    return render_template('doctor_appointments.html', appointments=appointments)



@app.route('/')
def home():
    return redirect(url_for('login'))



if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
