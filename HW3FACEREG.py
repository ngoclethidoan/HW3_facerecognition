import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("800x600")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.video_capture = None
        self.is_running = False
        self.is_webcam = False
        self.video_fps = 30

        self.image_label = tk.Label(root)
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Menu
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Video", command=self.load_video)
        filemenu.add_command(label="Webcam", command=self.start_webcam)
        filemenu.add_command(label="Stop", command=self.stop_video)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit_app)
        menubar.add_cascade(label="Menu", menu=filemenu)
        root.config(menu=menubar)

    def process_frames(self):
        frame_delay = 1.0 / self.video_fps
        while self.is_running and self.video_capture and self.video_capture.isOpened():
            start_time = time.time()

            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                print("⚠️ Warning: Frame not read properly.")
                time.sleep(0.05)
                continue

            if self.is_webcam:
                frame = cv2.flip(frame, 1)

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_faces = self.detect_faces(frame_rgb)
                self.show_image(frame_faces)
            except Exception as e:
                print(f"⚠️ Error processing frame: {e}")
                continue

            elapsed = time.time() - start_time
            time_to_sleep = max(0, frame_delay - elapsed)
            time.sleep(time_to_sleep)

    def start_webcam(self):
        self.stop_video()
        self.video_capture = cv2.VideoCapture(0)
        if self.video_capture.isOpened():
            self.video_fps = 720  # Webcam thường khoảng 30 FPS
            self.is_running = True
            self.is_webcam = True
            threading.Thread(target=self.process_frames, daemon=True).start()
        else:
            messagebox.showerror("Error", "Could not open webcam.")

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.wmv")])
        if file_path:
            self.stop_video()
            self.video_capture = cv2.VideoCapture(file_path)
            if self.video_capture.isOpened():
                fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                self.video_fps = fps if fps > 0 else 30
                print(f"Detected video FPS: {self.video_fps:.2f}")
                self.is_running = True
                self.is_webcam = False
                threading.Thread(target=self.process_frames, daemon=True).start()
            else:
                messagebox.showerror("Error", "Could not open video file.")
    def stop_video(self):
        self.is_running = False
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
        self.video_capture = None
        self.image_label.config(image="")
        self.image_label.imgtk = None



    def exit_app(self):
        self.root.quit()


    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
        

if __name__ == '__main__':
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
