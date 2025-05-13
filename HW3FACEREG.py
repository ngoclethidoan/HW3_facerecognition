import tkinter as tk
from tkinter import filedialog, mainloop
from tkinter import messagebox
from tkinter import font as tkfont

root = tk.Tk()
root.title('Menu Demonstration')
root.geometry("800x500")

# Creating Menubar
menubar = tk.Menu(root)

def video():
    file_path = filedialog.askopenfilename(title="Select a File")
    print(file_path)

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
