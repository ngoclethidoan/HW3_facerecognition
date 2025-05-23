import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np # Import numpy for creating blank images

class FaceDetectionApp:
    """
    A Tkinter application for real-time face detection from video files or webcam.
    Optimized to prevent white frames and ensure smooth display.
    """
    def __init__(self, root):
        """
        Initializes the FaceDetectionApp.

        Args:
            root: The main Tkinter window.
        """
        self.root = root
        self.root.title("Face Detection (Optimized)")
        # Set initial geometry, will be adjusted by image_label.pack(expand=True, fill=tk.BOTH)
        self.root.geometry("640x480") 

        # Load the pre-trained Haar Cascade classifier for face detection
        # This path points to the default Haar Cascade XML file provided by OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.video_capture = None  # Stores the OpenCV VideoCapture object
        self.is_running = False    # Flag to control the video processing loop
        self.is_webcam = False     # Flag to differentiate between webcam and video file input
        self.last_good_frame_rgb = None # Stores the last successfully processed RGB frame for display

        # Create a Tkinter Label to display video frames
        # Set its background to black to avoid white flashes when no frame is displayed
        self.image_label = tk.Label(root, bg="black")
        # Pack the label to fill the entire window, expanding with the window size
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Create a menubar for file operations
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0) # tearoff=0 prevents a dashed line at the top of the menu

        # Add commands to the File menu
        filemenu.add_command(label="Open Video", command=self.load_video)
        filemenu.add_command(label="Webcam", command=self.start_webcam)
        filemenu.add_command(label="Stop", command=self.stop_video)
        filemenu.add_separator() # Adds a visual separator
        filemenu.add_command(label="Exit", command=self.exit_app)
        
        # Add the File menu cascade to the menubar
        menubar.add_cascade(label="Menu", menu=filemenu)
        # Configure the root window to use this menubar
        root.config(menu=menubar)

        # Bind the <Configure> event to update the blank image size when the window is resized
        self.root.bind("<Configure>", self.on_resize)
        
        # Display an initial blank black image when the app starts
        self.show_blank_image()



if __name__ == '__main__':
    root = tk.Tk() # Create the main Tkinter window
    app = FaceDetectionApp(root) # Create an instance of the FaceDetectionApp
    root.mainloop() # Start the Tkinter event loop
