# Chronoface
## AI-Powered facial recognition logbook

# Overview
Chronoface is an advanced AI-powered facial recognition logbook designed to detect and track the entries of employees and visitors in a given space. Using a combination of Haar Cascade for face detection and OpenFace for facial recognition, Chronoface automatically captures the faces of people entering a monitored area, identifies whether they are an employee or visitor, and logs this data into a visual logbook. This system supports real-time facial recognition and provides a detailed log of entries along with the associated time and identity.

The application features visualization of detected faces, logs each entry, and supports various forms of analysis on the captured data.

# Features
### Haar Cascade + OpenFace Implementation:

Haar Cascade is used for efficient real-time face detection.

OpenFace is used for facial feature extraction and recognition, enabling precise identification.

### Employee and Visitor Detection:

The system is capable of differentiating between employees and visitors based on facial recognition and pre-trained datasets.

### Automatic Logbook Creation:

Chronoface maintains a log of all detected individuals, including timestamps, and stores their data for later reference.

### Visualization:

The application provides visual feedback of detected faces, displaying them on the screen as individuals enter the monitored space.

### Log File Generation:

All entries are stored in a log file containing:

Name/ID (Employee or Visitor)

Timestamp of entry

Identification status (matched to an existing person or a new entry)

### Real-Time Processing: 
The system captures facial images in real-time as individuals pass by, ensuring the logbook is updated dynamically.

# Installation
Prerequisites
Python 3.x

OpenCV for face detection
OpenFace for face recognition 

NumPy for matrix operations

Matplotlib for visualization
-------------------------------------------------------
# License
This project is licensed under the MIT License. See the LICENSE file for more details.
