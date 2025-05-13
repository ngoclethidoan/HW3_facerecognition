import tkinter as tk
from tkinter import filedialog, mainloop
from tkinter import messagebox
from tkinter import font as tkfont
import cv2
from PIL import Image, ImageTk
root = tk.Tk()
root.title('Menu Demonstration')
root.geometry("800x500")

# Creating Menubar
menubar = tk.Menu(root)

def video():
    label = tk.Label(root)
    label.pack()
    # Open file dialog to select video file
    file_path = filedialog.askopenfilename(title="Select a File")

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(file_path)

    # Get the width and height of the video frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the current window size
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    # Resize the frame to fit the window
    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Resize the frame to fit the window size
            frame_resized = cv2.resize(frame, (window_width, window_height))

            # Convert the frame from BGR to RGB (for display)
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Convert to ImageTk format for Tkinter
            img = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(img)

            # Update the Label widget with the new frame
            label.config(image=img_tk)
            label.image = img_tk  # Keep a reference to avoid garbage collection

            # Call the function again after 20ms (approx 50 frames per second)
            root.after(20, update_frame)
        else:
            cap.release()

    # Start updating frames
    root.after(20, update_frame)

    # Start Tkinter event loop
    root.mainloop()

# Adding File Menu and commands
menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Menu', menu=menu)
menu.add_command(label='Open File', command=video)
menu.add_command(label='WebCam', command=None)
menu.add_separator()
menu.add_command(label='Exit', command=root.destroy)

# display Menu
root.config(menu=menubar)
mainloop()
