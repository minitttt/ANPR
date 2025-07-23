import sys
import cv2
import pytesseract
import numpy as np
from collections import deque, Counter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                           QLabel, QPushButton, QWidget, QListWidget, QGroupBox,
                           QScrollArea, QFrame, QMessageBox, QFileDialog)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from datetime import datetime
import csv
import os

class LicensePlateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.log_file = "plate_log.csv"
        self.detected_plates_dir = "detected_plates"
        self.text_history = deque(maxlen=10)
        self.last_logged = None
        self.detection_enabled = True
        self.detection_count = 0
        self.video_processing = False

        self.setup_ui()
        self.setup_logging()
        
        # Initialize camera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def setup_ui(self):
        """Initialize all UI components"""
        self.setWindowTitle("License Plate Recognition System")
        self.setGeometry(100, 100, 1200, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("""
            background-color: black;
            border-radius: 6px;
            qproperty-alignment: 'AlignCenter';
        """)
        left_layout.addWidget(self.video_label)
        
        info_group = QGroupBox("Detection Info")
        info_group.setStyleSheet("""
            QGroupBox {
                font: bold 12px;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        info_layout = QVBoxLayout(info_group)
        
        self.result_label = QLabel("Ready to start detection")
        self.result_label.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            color: #4fc3f7;
            padding: 8px;
            background-color: #1e1e1e;
            border-radius: 4px;
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setStyleSheet("font-size: 14px; color: #a5d6a7;")
        
        info_layout.addWidget(self.result_label)
        info_layout.addWidget(self.confidence_label)
        left_layout.addWidget(info_group)
        
        
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Camera")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                font-size: 13px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_button.clicked.connect(self.start_camera)
        
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px;
                font-size: 13px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #ef9a9a;
            }
        """)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_camera)
        
        self.toggle_button = QPushButton("Pause Detection")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px;
                font-size: 13px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:checked {
                background-color: #FF9800;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_detection)
        
        self.upload_button = QPushButton("Upload Video")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 8px;
                font-size: 13px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self.upload_button.clicked.connect(self.upload_video)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.toggle_button)
        control_layout.addWidget(self.upload_button)
        left_layout.addLayout(control_layout)
        

        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 8px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        
        stats_group = QGroupBox("Statistics")
        stats_group.setStyleSheet(info_group.styleSheet())
        stats_layout = QVBoxLayout(stats_group)
        
        self.total_label = QLabel("Total detections: 0")
        self.today_label = QLabel("Today's detections: 0")
        self.accuracy_label = QLabel("Recent accuracy: -")
        
        for label in [self.total_label, self.today_label, self.accuracy_label]:
            label.setStyleSheet("font-size: 14px; color: white; padding: 4px;")
        
        stats_layout.addWidget(self.total_label)
        stats_layout.addWidget(self.today_label)
        stats_layout.addWidget(self.accuracy_label)
        right_layout.addWidget(stats_group)
        

        log_group = QGroupBox("Detection Log")
        log_group.setStyleSheet(info_group.styleSheet())
        log_layout = QVBoxLayout(log_group)
        
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                font-family: 'Courier New';
                font-size: 12px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #3a3a3a;
            }
        """)
        self.log_list.setAlternatingRowColors(True)
        
        log_scroll = QScrollArea()
        log_scroll.setWidgetResizable(True)
        log_scroll.setWidget(self.log_list)
        log_layout.addWidget(log_scroll)
        
        log_buttons = QHBoxLayout()
        self.clear_button = QPushButton("Clear Log")
        self.clear_button.setStyleSheet(self.start_button.styleSheet().replace("4CAF50", "607d8b"))
        self.clear_button.clicked.connect(self.clear_logs)
        
        self.export_button = QPushButton("Export Log")
        self.export_button.setStyleSheet(self.start_button.styleSheet().replace("4CAF50", "009688"))
        self.export_button.clicked.connect(self.export_logs)
        
        log_buttons.addWidget(self.clear_button)
        log_buttons.addWidget(self.export_button)
        log_layout.addLayout(log_buttons)
        
        right_layout.addWidget(log_group)

        main_layout.addWidget(left_panel, 70)
        main_layout.addWidget(right_panel, 30)

    def setup_logging(self):
        """Initialize logging system"""
        os.makedirs(self.detected_plates_dir, exist_ok=True)
        
        if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "License Plate", "Confidence", "Image Path"])
        
        self.load_existing_logs()

    def load_existing_logs(self):
        """Load previous detections from log file"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                reader = csv.reader(f)
                try:
                    next(reader)  
                    for row in reader:
                        if row and len(row) >= 3:
                            self.add_log_entry(row[0], row[1], row[2])
                            self.detection_count += 1
                except StopIteration:
                    pass
        self.update_stats()

    def add_log_entry(self, timestamp, plate, confidence):
        """Add entry to log display"""
        entry = f"{timestamp} | {plate.ljust(12)} | {confidence}%"
        self.log_list.insertItem(0, entry)

    def update_stats(self):
        """Update statistics display"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = 0
        
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row and row[0].startswith(today):
                        today_count += 1
        
        self.total_label.setText(f"Total detections: {self.detection_count}")
        self.today_label.setText(f"Today's detections: {today_count}")
        
        if self.text_history:
            counts = Counter(self.text_history)
            accuracy = int(100 * counts.most_common(1)[0][1] / len(self.text_history))
            self.accuracy_label.setText(f"Recent accuracy: {accuracy}%")

    def start_camera(self):
        """Initialize camera feed"""
        self.video_processing = False
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Camera Error", "Could not access camera.")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.result_label.setText("Live camera feed - Ready for detection")
        self.detection_enabled = True
        self.toggle_button.setChecked(False)
        self.toggle_button.setText("Pause Detection")

    def upload_video(self):
        """Open file dialog to select and process a video file"""
        options = QFileDialog.Options()
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
            options=options
        )
        
        if video_path:
            self.stop_camera()  # Stop live camera if running
            self.process_video_file(video_path)

    def process_video_file(self, video_path):
        """Process frames from uploaded video file"""
        self.video_processing = True
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.show_error("Video Error", "Could not open video file")
            return
        
       
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps if fps > 0 else 0
        
        self.result_label.setText(
            f"Processing: {os.path.basename(video_path)}\n"
            f"{fps:.1f} FPS, {frame_count} frames, {duration:.1f} sec"
        )
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.timer.start(30)

    def stop_camera(self):
        """Stop both camera and video processing"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        status = "Video processing completed" if self.video_processing else "Camera stopped"
        self.result_label.setText(status)
        self.video_processing = False

    def toggle_detection(self):
        """Toggle detection on/off"""
        self.detection_enabled = not self.detection_enabled
        if self.detection_enabled:
            self.toggle_button.setText("Pause Detection")
            self.result_label.setText("Detection resumed")
        else:
            self.toggle_button.setText("Resume Detection")
            self.result_label.setText("Detection paused")

    def update_frame(self):
        """Process each video frame with proper resizing"""
        ret, frame = self.cap.read()
        if not ret:
            if self.video_processing:
                self.stop_camera()
                self.result_label.setText("Video processing completed")
            return
        

        resized_frame, (new_width, new_height) = self.resize_to_fit(frame)
        

        rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
 
        bytes_per_line = 3 * new_width
        qt_image = QImage(rgb_image.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
        

        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        

        if not self.detection_enabled:
            return
            
     
        plate_img, bbox = self.detect_plate(frame)
        
        if plate_img is not None:

            text = self.read_plate(plate_img)
            
            if text:
                self.text_history.append(text)
                confidence = self.get_confidence(text)
                
                # Only log if confidence > 50% and new detection
                if confidence > 50 and text != self.last_logged:
                    self.log_detection(text, plate_img, confidence)
                    self.last_logged = text
                    self.detection_count += 1
                    

                    self.result_label.setText(f"Detected: {text}")
                    self.confidence_label.setText(f"Confidence: {confidence}%")
                    self.update_stats()
                    

                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                   
                    annotated_resized, _ = self.resize_to_fit(frame)
                    rgb_annotated = cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB)
                    qt_annotated = QImage(rgb_annotated.data, new_width, new_height, 
                                         bytes_per_line, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(qt_annotated))

    def resize_to_fit(self, frame):
        """Resize frame to fit display while maintaining aspect ratio"""
        label_width = self.video_label.width() - 20  # 20px padding
        label_height = self.video_label.height() - 20
        
        height, width = frame.shape[:2]
        frame_ratio = width / height
        label_ratio = label_width / label_height
        
        if frame_ratio > label_ratio:
            new_width = label_width
            new_height = int(new_width / frame_ratio)
        else:
            new_height = label_height
            new_width = int(new_height * frame_ratio)
        
        return cv2.resize(frame, (new_width, new_height)), (new_width, new_height)

    def detect_plate(self, frame):
        """Detect license plate in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)
        
        cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w > 100 and h > 30 and 2 < w/h < 6:
                    return frame[y:y+h, x:x+w], (x, y, w, h)
        return None, None

    def read_plate(self, plate_img):
        """Perform OCR on license plate"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.medianBlur(thresh, 3)
        
        config = '--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip()

    def get_confidence(self, text):
        """Calculate detection confidence"""
        counts = Counter(self.text_history)
        return int(100 * counts[text] / len(self.text_history)) if self.text_history else 0

    def log_detection(self, plate_text, plate_img, confidence):
        """Log detection to file and UI"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_filename = f"{self.detected_plates_dir}/{timestamp.replace(':', '')}_{plate_text}.jpg"
        
  
        cv2.imwrite(img_filename, plate_img)
        
        # Log to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, plate_text, confidence, img_filename])
        
        # Add to UI log
        self.add_log_entry(timestamp, plate_text, confidence)

    def clear_logs(self):
        """Clear the detection log"""
        reply = QMessageBox.question(self, 'Clear Log', 
                                   "Are you sure you want to clear all logs?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.log_list.clear()
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "License Plate", "Confidence", "Image Path"])
            self.detection_count = 0
            self.update_stats()

    def export_logs(self):
        """Export logs to file"""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self,
                                                "Export Logs",
                                                "",
                                                "CSV Files (*.csv);;All Files (*)",
                                                options=options)
        if filename:
            try:
                import shutil
                shutil.copy2(self.log_file, filename)
                QMessageBox.information(self, "Success", f"Logs exported to {filename}")
            except Exception as e:
                self.show_error("Export Failed", f"Error: {str(e)}")

    def show_error(self, title, message):
        """Display error message"""
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.exec_()

    def closeEvent(self, event):
        """Clean up on window close"""
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
   
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    window = LicensePlateApp()
    window.show()
    sys.exit(app.exec_())