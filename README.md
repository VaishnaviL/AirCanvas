# 🎨 Virtual Paint Application using OpenCV

This Python application enables you to draw on a virtual canvas using a **colored object** as a marker. The object is tracked using its **HSV (Hue, Saturation, Value)** color range through your webcam feed, and allows dynamic color selection and drawing in real-time.

---

## 📌 Features

- 🖍️ Real-time **color tracking** using OpenCV.
- 🎨 Draw with multiple colors: **Blue**, **Green**, **Red**, and **Yellow**.
- 🧽 **Clear canvas** functionality.
- 🔧 Adjustable HSV range with **trackbars** for color calibration.
- 👆 Simple gesture interface — move the colored object to the top buttons to switch colors or clear the canvas.

---

## 🛠️ Requirements

Install the following Python packages:

```bash
pip install numpy opencv-python
```

## Run the script using python 
```bash
python Air-canvas.py
```
A webcam window will open. You’ll see:

Your camera feed with control buttons on top.

A blank canvas window where your drawing appears.

A third window showing the HSV mask used to track your colored object.

--- 

## 🖌️ Color Detection Calibration
- Use the Color detectors window to fine-tune the HSV values for your marker color:

    Upper Hue, Upper Saturation, Upper Value
    
    Lower Hue, Lower Saturation, Lower Value

  This is important if lighting conditions or object color changes.

## Methodology

1. Capture Video: The application uses your webcam (cv2.VideoCapture(0)).

2. HSV Conversion: Each frame is converted from BGR to HSV.

3. Trackbar HSV Range: You define the HSV bounds for detecting the colored object.

4. Masking and Contours:

    - A binary mask is created for the object.
    
    - Morphological operations clean up the noise.
    
    - The largest contour is assumed to be the marker.

5. Interaction:

    - If the center of the detected object is within the button area, it changes color or clears canvas.
    
    - Otherwise, it adds points to the selected color path and draws lines between them on both the frame and canvas.

6. Drawing:

    - Paths are stored as lists of deques.
    
    - Lines are continuously drawn between consecutive points in the deque.
