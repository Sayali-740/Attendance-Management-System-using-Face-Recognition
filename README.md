Face Attendance Management System

Overview
This project is a **Face Attendance Management System** built with **Streamlit**, **OpenCV**, and **SQLite**. It leverages face recognition technology for attendance marking and provides an admin panel for user management. The system includes features for capturing faces, registering users, marking attendance, and viewing attendance records.

Features

1. Admin Panel:
   - Add new users by capturing their faces.
   - Manage registered users (view and delete).
   - Secure login with admin credentials.

2. Face Detection & Recognition:
   - Detects and recognizes faces using OpenCV's Haar Cascade and LBPH face recognizer.

3. Attendance Management:
   - Mark attendance automatically upon successful face recognition.
   - Display attendance records with timestamps.
   - Delete attendance records as needed.

4. Database Integration:
   - User data and attendance records are stored in an SQLite database.

Prerequisites

1. Python (>= 3.7)
2. Libraries:
   - `streamlit`
   - `opencv-python`
   - `numpy`
   - `sqlite3`
3. Hardware:
   - Webcam (for face detection and recognition)

Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Face_Attendance_System.git
  
2. Navigate to the project directory:
   ```bash
   cd Face_Attendance_System
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

Project Structure

```
Face_Attendance_System/
├── app.py                 # Main application script
├── requirements.txt       # List of dependencies
├── database.db            # SQLite database file
├── Readme.md              # Documentation
└── assets/                # Assets (if any, e.g., images, icons)
```

How to Use

1. Admin Login
- Navigate to the **Admin Panel**.
- Login using admin credentials:
  - Username: `admin`
  - Password: `admin123`

2. Add New User
- Enter the user's name.
- Capture the user's face via webcam.
- The system will save the face and associate it with the user's name in the database.

3. Mark Attendance
- Navigate to **Mark Attendance**.
- Stand in front of the webcam.
- The system will recognize the face and mark attendance automatically.

4. View Attendance Records
- View attendance logs with timestamps in the **Attendance Records** section.
- Delete specific records if needed.

Security Features
- Admin-only access to user management and attendance records.
- Secure SQLite database to store user data and attendance logs.

Future Enhancements
- Add email notifications for attendance marking.
- Enhance face recognition accuracy with deep learning models.
- Implement role-based access control (RBAC).
