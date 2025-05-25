import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageFont, ImageDraw
import threading
import numpy as np
import os
import face_recognition

def put_text_unicode(cv2_img, text, position, font_path="arial.ttf", font_size=20, color=(0,255,0)):
    """
    Vẽ chữ Unicode (có dấu) lên ảnh OpenCV bằng Pillow.
    cv2_img: ảnh OpenCV (BGR)
    text: chuỗi cần vẽ
    position: (x, y) vị trí bắt đầu vẽ chữ
    font_path: đường dẫn tới file font .ttf (phải hỗ trợ tiếng Việt)
    font_size: kích cỡ chữ
    color: màu chữ RGB (mặc định xanh lá)
    """
    img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color[::-1])
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv2


class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition (Optimized)")
        self.root.geometry("800x600")

        self.video_capture = None
        self.is_running = False
        self.is_webcam = False
        self.last_good_frame_rgb = None

        self.known_face_encodings = []
        self.known_face_names = []

        # Các biến lưu trữ kết quả nhận diện để làm chậm khung
        self.last_face_locations = []
        self.last_face_names = []
        self.frames_since_update = 0
        self.update_interval = -10 # Cập nhật kết quả mỗi -10 frame

        # Tkinter UI
        self.image_label = tk.Label(root, bg="black")
        self.image_label.pack(expand=True, fill=tk.BOTH)

        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Known Faces", command=self.load_known_faces)
        filemenu.add_command(label="Open Video", command=self.load_video)
        filemenu.add_command(label="Webcam", command=self.start_webcam)
        filemenu.add_command(label="Stop", command=self.stop_video)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit_app)
        menubar.add_cascade(label="Menu", menu=filemenu)
        root.config(menu=menubar)

        self.root.bind("<Configure>", self.on_resize)
        self.show_blank_image()

    def load_known_faces(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Known Faces")
        if not folder_path:
            return
        self.known_face_encodings = []
        self.known_face_names = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                image = face_recognition.load_image_file(file_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)
                else:
                    print(f"⚠️ No faces found in {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        messagebox.showinfo("Info", f"Loaded {len(self.known_face_encodings)} known faces.")

    def on_resize(self, event=None):
        if not self.is_running:
            self.show_blank_image()

    def process_frames(self):
        frame_skip = 2
        count = 0

        while self.is_running and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                continue

            if self.is_webcam:
                frame = cv2.flip(frame, 1)

            if count % frame_skip == 0:
                if self.frames_since_update >= self.update_interval:
                    # Cập nhật nhận diện mới
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        name = "Unknown"

                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]

                        face_names.append(name)

                    # Lưu lại kết quả nhận diện
                    self.last_face_locations = face_locations
                    self.last_face_names = face_names
                    self.frames_since_update = 0
                else:
                    self.frames_since_update += 1

            # Vẽ khung dựa trên kết quả lưu trữ
            for (top, right, bottom, left), name in zip(self.last_face_locations, self.last_face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
                frame = put_text_unicode(frame, name, (left + 6, bottom - 27), font_path="arial.ttf", font_size=20, color=(0, 0, 0))

            label_width = self.image_label.winfo_width() or 640
            label_height = self.image_label.winfo_height() or 480
            display_frame = cv2.resize(frame, (label_width, label_height))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.last_good_frame_rgb = rgb_frame
            self.root.after(0, self.show_frame, rgb_frame)
            count += 1

        self.root.after(0, self.stop_video)

    def start_webcam(self):
        self.stop_video()
        self.video_capture = cv2.VideoCapture(0)
        if self.video_capture.isOpened():
            self.is_running = True
            self.is_webcam = True
            threading.Thread(target=self.process_frames, daemon=True).start()
        else:
            messagebox.showerror("Error", "Cannot open webcam.")
            self.show_blank_image()

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.wmv")])
        if file_path:
            self.stop_video()
            self.video_capture = cv2.VideoCapture(file_path)
            if self.video_capture.isOpened():
                self.is_running = True
                self.is_webcam = False
                threading.Thread(target=self.process_frames, daemon=True).start()
            else:
                messagebox.showerror("Error", "Cannot open video file.")
                self.show_blank_image()
        else:
            self.show_blank_image()

    def show_frame(self, frame_rgb):
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.imgtk = imgtk
        self.image_label.config(image=imgtk)

    def show_blank_image(self):
        label_width = self.image_label.winfo_width() or 640
        label_height = self.image_label.winfo_height() or 480
        blank_img = np.zeros((label_height, label_width, 3), dtype=np.uint8)
        self.root.after(0, self.show_frame, blank_img)

    def stop_video(self):
        self.is_running = False
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
        self.video_capture = None
        self.last_good_frame_rgb = None
        self.last_face_locations = []
        self.last_face_names = []
        self.frames_since_update = 0
        self.root.after(0, self.show_blank_image)

    def exit_app(self):
        self.stop_video()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()