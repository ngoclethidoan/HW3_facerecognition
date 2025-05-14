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


if __name__ == '__main__':
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
