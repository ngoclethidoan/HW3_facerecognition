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

    def on_resize(self, event=None):
        # Only update the blank image if no video is currently running
        if not self.is_running:
            self.show_blank_image()

    def process_frames(self):
        frame_skip = 2  # Process face detection every 'frame_skip' frames for performance
        count = 0  # Frame counter
        faces = []  # Stores detected face bounding boxes

        # Loop as long as the app is running and video capture is active
        while self.is_running and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()  # Read a frame from the video source

            # Check if the frame was read successfully
            if not ret or frame is None:
                print("⚠️ Frame not read or empty. Displaying last valid frame.")
                # If a frame is not read, continue to the next iteration.
                # The self.last_good_frame_rgb will remain on display because show_frame is not called again.
                continue  # Skip the rest of the processing for this bad frame

            # If using webcam, flip the frame horizontally for a mirror effect
            if self.is_webcam:
                frame = cv2.flip(frame, 1)

            # Resize the frame for faster face detection (smaller image)
            small_frame = cv2.resize(frame, (320, 240))
            # Convert the small frame to grayscale for face detection
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Perform face detection only on skipped frames to improve performance
            if count % frame_skip == 0:
                # detectMultiScale detects objects of different sizes in the input image.
                # 1.1 is the scale factor, 5 is the minimum number of neighbors
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

            # Calculate scaling factors to draw bounding boxes back on the original frame size
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240

            # Draw rectangles around detected faces on the original frame
            for (x, y, w, h) in faces:
                cv2.rectangle(frame,
                              (int(x * scale_x), int(y * scale_y)),  # Top-left corner
                              (int((x + w) * scale_x), int((y + h) * scale_y)),  # Bottom-right corner
                              (0, 255, 0), 2)  # Green color, 2px thickness

            # Get the current dimensions of the image_label for proper resizing
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()

            # Fallback for when label dimensions are not yet available (e.g., during startup)
            if label_width <= 1 or label_height <= 1:
                # Use the root window's dimensions as a fallback, or a sensible default
                root_width = self.root.winfo_width()
                root_height = self.root.winfo_height()
                label_width = root_width if root_width > 1 else 640
                label_height = root_height if root_height > 1 else 480

            # Resize the processed frame to fit the image_label's dimensions
            display_frame = cv2.resize(frame, (label_width, label_height))
            # Convert the frame from BGR (OpenCV default) to RGB (Pillow/Tkinter compatible)
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Store the current good frame
            self.last_good_frame_rgb = rgb_frame

            # Schedule the show_frame function to run on the main Tkinter thread
            # This is crucial for thread safety when updating GUI elements
            self.root.after(0, self.show_frame, rgb_frame)
            count += 1

        print("🔚 Video loop ended.")
        # When the video loop ends (e.g., video finished, stop button pressed),
        # ensure the display is cleared and resources are released on the main thread.
        self.root.after(0, self.stop_video)


if __name__ == '__main__':
    root = tk.Tk() # Create the main Tkinter window
    app = FaceDetectionApp(root) # Create an instance of the FaceDetectionApp
    root.mainloop() # Start the Tkinter event loop
