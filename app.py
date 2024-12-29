import streamlit as st
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import os
from typing import Optional, Tuple, List
import time

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Set admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Create database and tables
def init_db():
    """Initialize database with users and attendance tables"""
    conn = sqlite3.connect(r"C:\Face_reco_attendance_management-main\database.db")  # Specify the database file
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  face_image BLOB)''')
    
    # Create attendance table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  name TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()


def verify_login(username: str, password: str) -> bool:
    """Verify admin credentials"""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

# Improved face matching function using LBPH recognizer
def train_face_recognizer() -> Tuple[np.ndarray, np.ndarray]:
    """Train face recognizer with all stored users"""
    conn = sqlite3.connect(r"C:\Face_reco_attendance_management-main\database.db")
    c = conn.cursor()
    c.execute("SELECT id, face_image FROM users")
    
    faces = []
    labels = []
    for user_id, face_image in c.fetchall():
        # Convert bytes back to image
        nparr = np.frombuffer(face_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(user_id)
    
    conn.close()
    
    # Train the face recognizer on all user data
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    
    return recognizer, labels

# Detect face and return the detected image
def detect_face(frame: np.ndarray) -> Optional[np.ndarray]:
    """Detect and return face from frame"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    elif len(faces) > 1:
        return None
        
    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]
    return cv2.resize(face, (200, 200))  # Resize for consistency

def capture_face() -> Optional[np.ndarray]:
    """Capture and return face if detected"""
    cap = cv2.VideoCapture(0)
    st.info("Camera starting... Please wait.")
    time.sleep(2)  # Give camera time to warm up
    
    if not cap.isOpened():
        st.error("Error: Could not open camera")
        return None
    
    preview = st.empty()
    
    for _ in range(30):  # Show preview for 3 seconds
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror effect
            face = detect_face(frame)
            if face is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview.image(frame_rgb)
            else:
                preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.1)
    
    ret, frame = cap.read()
    cap.release()
    preview.empty()
    
    if not ret:
        st.error("Error: Could not capture frame")
        return None
    
    frame = cv2.flip(frame, 1)
    face = detect_face(frame)
    
    if face is None:
        st.error("No face detected or multiple faces detected. Please try again.")
        return None
        
    return face

def save_user(name: str, face_image: np.ndarray) -> bool:
    """Save user to database"""
    try:
        conn = sqlite3.connect(r"C:\Face_reco_attendance_management-main\database.db")
        c = conn.cursor()
        _, img_encoded = cv2.imencode('.jpg', face_image)
        c.execute("INSERT INTO users (name, face_image) VALUES (?, ?)",
                 (name, img_encoded.tobytes()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving user: {str(e)}")
        return False

def get_all_users() -> List[Tuple[int, str, np.ndarray]]:
    """Get list of all registered users"""
    conn = sqlite3.connect(r"C:\Face_reco_attendance_management-main\database.db")
    c = conn.cursor()
    c.execute("SELECT id, name, face_image FROM users")
    users = []
    for user_id, name, face_image in c.fetchall():
        nparr = np.frombuffer(face_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        users.append((user_id, name, img))
    conn.close()
    return users

def mark_attendance(face_image: np.ndarray) -> Optional[str]:
    """Mark attendance for recognized face"""
    recognizer, _ = train_face_recognizer()
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    label, confidence = recognizer.predict(gray_face)
    
    if confidence < 100:  # A confidence threshold for recognition
        conn = sqlite3.connect(r"C:\Face_reco_attendance_management-main\database.db")
        c = conn.cursor()
        c.execute("SELECT name FROM users WHERE id=?", (label,))
        name = c.fetchone()[0]
        c.execute("INSERT INTO attendance (user_id, name) VALUES (?, ?)", (label, name))
        conn.commit()
        conn.close()
        return name
    return None

def delete_user(user_id: int) -> bool:
    """Delete a user from the database."""
    try:
        conn = sqlite3.connect(r"C:\Face_reco_attendance_management-main\database.db")
        c = conn.cursor()
        
        # Delete the user from the users table
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        
        # Also delete the user's attendance records to maintain data consistency
        c.execute("DELETE FROM attendance WHERE user_id=?", (user_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting user: {str(e)}")
        return False

def attendance_page():
    """Page for marking attendance and viewing records"""
    st.header("Mark Attendance")
    
    if st.button("Mark Attendance"):
        face_image = capture_face()
        if face_image is not None:
            name = mark_attendance(face_image)
            if name:
                st.success(f"Attendance marked for {name}")
                st.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.error("No matching face found in database")

    # Display attendance records
    st.subheader("Attendance Records")
    conn = sqlite3.connect(r"C:\Face_reco_attendance_management-main\database.db")
    c = conn.cursor()
    c.execute("SELECT id, name, timestamp FROM attendance ORDER BY timestamp DESC")
    records = c.fetchall()
    conn.close()

    if records:
        for record in records:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"{record[1]} - {record[2]}")
            with col2:
                st.write(f"Time: {record[2]}")
            with col3:
                # Add delete button for each record
                if st.button(f"Delete Record {record[0]}", key=f"delete_{record[0]}"):
                    if delete_attendance(record[0]):
                        st.success(f"Attendance record for {record[1]} deleted successfully.")
                        st.experimental_rerun()
    else:
        st.info("No attendance records found.")

def delete_attendance(record_id: int) -> bool:
    """Delete an attendance record from the database"""
    try:
        conn = sqlite3.connect(r"C:\Face_reco_attendance_management-main\database.db")
        c = conn.cursor()
        c.execute("DELETE FROM attendance WHERE id=?", (record_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting attendance record: {str(e)}")
        return False

def admin_panel():
    """Admin panel for managing users"""
    st.header("Admin Panel")
    
    # Add new user
    st.subheader("Add New User")
    with st.form("registration_form"):
        name = st.text_input("Name")
        if st.form_submit_button("Capture Face & Register"):
            if name:
                face_image = capture_face()
                if face_image is not None:
                    if save_user(name, face_image):
                        st.success(f"Successfully registered {name}")
            else:
                st.error("Please enter a name")
    
    # Delete users
    st.subheader("Manage Users")
    users = get_all_users()
    if users:
        for user_id, name, face_image in users:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"Name: {name}")
            with col2:
                st.image(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), width=100)
            with col3:
                if st.button("Delete", key=f"delete_{user_id}"):
                    if delete_user(user_id):
                        st.success(f"Deleted {name}")
                        st.rerun()
    else:
        st.info("No users registered")

def login_page():
    """Admin login page"""
    st.title("Admin Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if verify_login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

def main():
    st.set_page_config(
        page_title="Face Attendance System",
        page_icon="📸",
        layout="wide"
    )
    
    # Initialize database
    init_db()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Mark Attendance", "Admin Panel", "Attendance Records"])
    
    if st.session_state.logged_in:
        st.sidebar.success("Logged in as Admin")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    if page == "Admin Panel":
        if not st.session_state.logged_in:
            login_page()
        else:
            admin_panel()
    elif page == "Attendance Records":
        attendance_page()
    else:
        attendance_page()

if __name__ == "__main__":
    main()
