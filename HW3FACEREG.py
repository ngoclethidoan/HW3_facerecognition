import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont
# Create the main window
root = tk.Tk()
root.title("Left and Right Frames")
root.geometry("800x500")

# Maximize the window
#root.state('zoomed')

#create a dynamic font
dynamic_font = tkfont.Font(family='Arial', size=20)

# Create a frame inside the window
frame1 = tk.Frame(root, bg="white")
frame1.place(relx=0, rely=0, relwidth=0.25, relheight=1)

def upload_file():
    file_path = filedialog.askopenfilename(title="Select a File")
    print("Selected file:", file_path)
    if file_path:  # If a file is selected
        # Change the color of the button after the file is uploaded
        button2.config(bg="lightblue")#, text="Uploaded!")


# Create buttons and add them to the frame
button1 = tk.Button(frame1, text="WEBCAM", font=dynamic_font, command=lambda: print("Button 1 clicked"))
button1.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.1)

button2 = tk.Button(frame1, text="VIDEO", font=dynamic_font, command=upload_file)
button2.place(relx=0.1, rely=0.25, relwidth=0.8, relheight=0.1)

frame2 = tk.Frame(root, bg="lightblue")
frame2.place(relx=0.25, rely=0, relwidth=0.75, relheight=1)

# rezize font of buttons
def resize_font(event):
    button_height = button1.winfo_height()
    new_size = max(7, int(button_height * 0.4))
    dynamic_font.configure(size=new_size)

# call resize_font when changing window
root.bind("<Configure>", resize_font)

# Run the application
root.mainloop()
