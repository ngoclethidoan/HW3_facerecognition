import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection")
        self.root.geometry("800x600")


        # Menu
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Video", command=None)
        filemenu.add_command(label="Webcam", command=None)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=None)
        menubar.add_cascade(label="Menu", menu=filemenu)
        root.config(menu=menubar)

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
