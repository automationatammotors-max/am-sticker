# Setup Guide

## 1. Install Tesseract-OCR (Windows)
Download and install from:
https://github.com/UB-Mannheim/tesseract/wiki

Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
This path is already configured in detector.py. If you install elsewhere, update line 10.

## 2. Install Python packages
```
pip install -r requirements.txt
```

## 3. Run the app
```
streamlit run app.py
```

## 4. Test with video
- In the UI, select **Testing Video** as source
- Click **▶ Start**
- Watch detections appear on the feed

## 5. Switch to webcam / live camera
- Select **Webcam 0** for the built-in laptop camera
- For an external USB camera, try **Webcam 1** or **Webcam 2**
- The camera index depends on connection order

## File outputs
- Logs saved to: `logs/detections.xlsx`
- Colour coding: Green = all 3 stickers, Yellow/Orange = partial, Red = none

## How logging works
- System watches the frame for AM Motors stickers + vehicle plate
- A vehicle is "confirmed" after 2 consecutive detections (adjustable)
- When the car leaves the frame (no detections for 4 sec), ONE row is logged
- The row captures the best result seen while the car was present
- Prevents duplicate rows for the same vehicle
