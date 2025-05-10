import tkinter as tk
from tkinter import filedialog
# Create the main window
root = tk.Tk()
root.title("Left and Right Frames")
root.geometry("800x500")

# Maximize the window
root.state('zoomed')

# Create a frame inside the window
frame1 = tk.Frame(root, bg="white", padx=10, pady=10)
frame1.pack(fill='both', expand=True)

def upload_file():
    file_path = filedialog.askopenfilename(title="Select a File")
    print("Selected file:", file_path)
    if file_path:  # If a file is selected
        # Change the color of the button after the file is uploaded
        button2.config(bg="lightblue")#, text="Uploaded!")


# Create buttons and add them to the frame
button1 = tk.Button(frame1, text="WEBCAM", width=10, height=1, bd =5, font=('Arial', 20), command= lambda: print("Button 1 clicked"))
button1.pack(side='top', anchor='w', padx=10, pady=10)

button2 = tk.Button(frame1, text="VIDEO", width=10, height=1, bd =5, font=('Arial', 20), command= upload_file)#lambda: print("Button 2 clicked"))
button2.pack(side='top', anchor='w', padx=10, pady=10)


frame2 = tk.Frame(root, bg="lightblue", width=1070, height=640)
frame2.place(x=200, y=5)


# Run the application
root.mainloop()

