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
    """
    Ứng dụng Tkinter để phát hiện và nhận dạng khuôn mặt theo thời gian thực từ tệp video hoặc webcam.
    Được tối ưu hóa để ngăn chặn khung hình trắng và đảm bảo hiển thị mượt mà.
    """
    def __init__(self, root):
        """
        Khởi tạo FaceDetectionApp.

        Args:
            root: Cửa sổ Tkinter chính.
        """
        self.root = root
        self.root.title("Face Recognition (Optimized)")

        # Đặt kích thước cửa sổ theo kích thước màn hình
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        # Load bộ phân loại Haar Cascade cho phát hiện khuôn mặt (chưa dùng trong code này)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.video_capture = None  # Lưu trữ đối tượng OpenCV VideoCapture
        self.is_running = False    # Cờ điều khiển vòng lặp xử lý video
        self.is_webcam = False     # Cờ phân biệt giữa đầu vào webcam và tệp video
        self.last_good_frame_rgb = None # Lưu trữ khung hình RGB được xử lý thành công cuối cùng để hiển thị

        # Biến để lưu trữ kết quả nhận dạng từ lần chạy gần nhất
        self.current_face_locations = []
        self.current_face_names = []

        # Tạo một Tkinter Label để hiển thị khung hình video
        # Đặt nền đen để tránh nháy trắng khi không có khung hình nào được hiển thị
        self.image_label = tk.Label(root, bg="black")
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Tạo thanh menu
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)

        # Thêm lệnh vào menu File
        filemenu.add_command(label="Load Known Faces", command=self.load_known_faces)
        filemenu.add_command(label="Mở Video", command=self.load_video)
        filemenu.add_command(label="Webcam", command=self.start_webcam)
        filemenu.add_command(label="Dừng", command=self.stop_video)
        filemenu.add_separator()
        filemenu.add_command(label="Thoát", command=self.exit_app)
        
        menubar.add_cascade(label="Menu", menu=filemenu)
        root.config(menu=menubar)

        # Ràng buộc sự kiện <Configure> để cập nhật kích thước ảnh trống khi cửa sổ được thay đổi kích thước
        self.root.bind("<Configure>", self.on_resize)
        
        # Hiển thị ảnh đen trống ban đầu khi ứng dụng khởi động
        self.show_blank_image()
    def load_known_faces(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Known Faces")
        if not folder_path:
            return
        self.known_encodings = []
        self.known_names = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                image = face_recognition.load_image_file(file_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    self.known_names.append(name)
                else:
                    print(f"⚠️ No faces found in {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        messagebox.showinfo("Info", f"Loaded {len(self.known_encodings)} known faces.")

    def on_resize(self, event=None):
        """
        Xử lý sự kiện thay đổi kích thước cửa sổ để cập nhật kích thước ảnh trống
        và đảm bảo hiển thị thích ứng chính xác.
        """
        # Chỉ cập nhật ảnh trống nếu không có video nào đang chạy
        if not self.is_running:
            self.show_blank_image()

    def process_frames(self):
        """
        Đọc các khung hình từ video, thực hiện phát hiện và nhận dạng khuôn mặt,
        và cập nhật hiển thị Tkinter. Chạy trong một luồng riêng biệt.
        """
        # Thực hiện nhận dạng khuôn mặt mỗi 'frame_skip' khung hình để cải thiện hiệu suất
        frame_skip = 5  
        count = 0       # Bộ đếm khung hình

        while self.is_running and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()

            if not ret or frame is None:
                # Nếu không đọc được khung hình hoặc khung hình trống, tiếp tục vòng lặp.
                # Khung hình hợp lệ cuối cùng sẽ tiếp tục được hiển thị.
                print("⚠️ Không đọc được khung hình hoặc khung hình trống. Hiển thị khung hình hợp lệ cuối cùng.")
                if self.last_good_frame_rgb is not None:
                     self.root.after(0, self.show_frame, self.last_good_frame_rgb)
                continue 

            # Nếu sử dụng webcam, lật khung hình theo chiều ngang để có hiệu ứng gương
            if self.is_webcam:
                frame = cv2.flip(frame, 1)
                
            # Lấy kích thước hiện tại của image_label để thay đổi kích thước phù hợp
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()

            # Dự phòng khi kích thước label chưa có sẵn (ví dụ: trong quá trình khởi động)
            if label_width <= 1 or label_height <= 1:
                root_width = self.root.winfo_width()
                root_height = self.root.winfo_height()
                label_width = root_width if root_width > 1 else 640
                label_height = root_height if root_height > 1 else 480

            # Thay đổi kích thước khung hình để phù hợp với kích thước của image_label
            display_frame = cv2.resize(frame, (label_width, label_height))
            # Chuyển đổi khung hình từ BGR (mặc định của OpenCV) sang RGB (tương thích với Pillow/Tkinter)
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # --- Thực hiện nhận dạng khuôn mặt chỉ trên các khung hình đã bỏ qua ---
            if count % frame_skip == 0:
                # Thay đổi kích thước khung hình xuống 1/4 (hoặc tỷ lệ khác) để tăng tốc độ phát hiện và mã hóa
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
                
                # Tìm vị trí khuôn mặt trong khung hình nhỏ
                self.current_face_locations = face_recognition.face_locations(small_frame)
                # Mã hóa khuôn mặt được tìm thấy
                current_face_encodings = face_recognition.face_encodings(small_frame, self.current_face_locations)
                
                self.current_face_names = []
                for face_encoding in current_face_encodings:
                    # Tính khoảng cách đến tất cả khuôn mặt đã biết
                    distances = face_recognition.face_distance(self.known_encodings, face_encoding)

                    # Tìm chỉ số khớp nhất
                    best_match_index = np.argmin(distances)
                    name = "Unknown"

                    # Ngưỡng để quyết định nhận dạng (thông thường < 0.6 là tốt, < 0.5 là rất chắc chắn)
                    if distances[best_match_index] < 0.45: 
                        name = self.known_names[best_match_index]
                    self.current_face_names.append(name)
                    # print(f"[INFO] Khoảng cách tới {name}: {distances[best_match_index]:.4f}")
            # --- Kết thúc khối nhận dạng khuôn mặt ---

            # Vẽ hình chữ nhật và tên bằng cách sử dụng kết quả từ lần nhận dạng gần nhất
            # Phóng to lại vị trí khuôn mặt vì việc phát hiện được thực hiện trên khung hình nhỏ hơn
            for (top, right, bottom, left), name in zip(self.current_face_locations, self.current_face_names):
                # Phóng to tọa độ khuôn mặt lên lại (ví dụ: x4 nếu fx/fy = 0.25)
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Đặt chữ ngay dưới hộp khuôn mặt hoặc bên trong nếu hộp nhỏ
                text_y = bottom + 20 if bottom + 20 < label_height else bottom - 10
                rgb_frame = put_text_unicode(rgb_frame, name, (left +4, bottom - 27), font_path="arial.ttf", font_size=20, color=(0, 255, 0))

            self.last_good_frame_rgb = rgb_frame
            
            # Lên lịch hàm show_frame chạy trên luồng chính của Tkinter
            # Điều này rất quan trọng để an toàn luồng khi cập nhật các phần tử GUI
            self.root.after(0, self.show_frame, rgb_frame)
            count += 1

        print("🔚 Vòng lặp video đã kết thúc.")
        # Khi vòng lặp video kết thúc (ví dụ: video hết, nhấn nút dừng),
        # đảm bảo hiển thị được xóa và tài nguyên được giải phóng trên luồng chính.
        self.root.after(0, self.stop_video)

    def start_webcam(self):
        """
        Bắt đầu luồng webcam để phát hiện khuôn mặt.
        """
        self.stop_video() # Dừng mọi luồng video hiện có
        self.video_capture = cv2.VideoCapture(0) # Mở webcam mặc định (chỉ số 0)
        if self.video_capture.isOpened():
            self.is_running = True
            self.is_webcam = True
            # Bắt đầu xử lý khung hình trong một luồng daemon mới
            # Một luồng daemon sẽ tự động chấm dứt khi chương trình chính thoát
            threading.Thread(target=self.process_frames, daemon=True).start()
        else:
            messagebox.showerror("Lỗi", "Không thể mở webcam. Vui lòng kiểm tra xem nó có được kết nối và không bị sử dụng không.")
            self.show_blank_image() # Hiển thị trống nếu webcam không mở được

    def load_video(self):
        """
        Mở hộp thoại chọn tệp video và bắt đầu xử lý.
        """
        # Mở hộp thoại tệp để chọn tệp video
        file_path = filedialog.askopenfilename(filetypes=[("Tệp video", "*.mp4;*.avi;*.mov;*.wmv;*.mkv")])
        if file_path: # Nếu một tệp đã được chọn
            self.stop_video() # Dừng mọi luồng video hiện có
            self.video_capture = cv2.VideoCapture(file_path) # Mở tệp video đã chọn
            if self.video_capture.isOpened():
                self.is_running = True
                self.is_webcam = False
                # Bắt đầu xử lý khung hình trong một luồng daemon mới
                threading.Thread(target=self.process_frames, daemon=True).start()
            else:
                messagebox.showerror("Lỗi", "Không thể mở tệp video. Tệp có thể bị hỏng hoặc không phải định dạng được hỗ trợ.")
                self.show_blank_image() # Hiển thị trống nếu tệp video không mở được
        else:
            self.show_blank_image() # Nếu người dùng hủy hộp thoại tệp, hiển thị trống

    def show_frame(self, frame_rgb):
        """
        Hiển thị một khung hình RGB đã cho trong image_label của Tkinter.
        Hàm này được thiết kế để được gọi trên luồng chính của Tkinter.

        Args:
            frame_rgb: Một mảng NumPy biểu diễn khung hình ảnh RGB.
        """
        # Chuyển đổi mảng NumPy (RGB) thành đối tượng Pillow Image
        img = Image.fromarray(frame_rgb)
        # Chuyển đổi Pillow Image thành đối tượng Tkinter PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        # Lưu trữ tham chiếu đến đối tượng PhotoImage để ngăn nó bị thu gom rác
        self.image_label.imgtk = imgtk
        # Cập nhật hình ảnh được hiển thị bởi Tkinter Label
        self.image_label.config(image=imgtk)

    def show_blank_image(self):
        """
        Tạo và hiển thị một ảnh đen trống trên image_label.
        Hữu ích cho trạng thái ban đầu hoặc khi không có video nào đang phát.
        """
        # Lấy kích thước label hiện tại cho ảnh trống
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()

        # Dự phòng khi kích thước label chưa biết (ví dụ: trong quá trình __init__)
        if label_width <= 1 or label_height <= 1:
            root_width = self.root.winfo_width()
            root_height = self.root.winfo_height()
            label_width = root_width if root_width > 1 else 640
            label_height = root_height if root_height > 1 else 480
            
        # Tạo một mảng NumPy đen trống (chiều cao, chiều rộng, 3 kênh cho RGB)
        blank_img = np.zeros((label_height, label_width, 3), dtype=np.uint8)
        # Lên lịch ảnh trống được hiển thị trên luồng chính của Tkinter
        self.root.after(0, self.show_frame, blank_img)


    def stop_video(self):
        """
        Dừng luồng video hiện tại, giải phóng tài nguyên và xóa hiển thị.
        """
        self.is_running = False # Đặt cờ để dừng vòng lặp xử lý
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release() # Giải phóng đối tượng video capture
        self.video_capture = None # Xóa tham chiếu
        self.last_good_frame_rgb = None # Xóa khung hình tốt cuối cùng
        self.current_face_locations = [] # Xóa kết quả nhận dạng
        self.current_face_names = []
        # Lên lịch xóa hiển thị thành ảnh trống trên luồng chính của Tkinter
        self.root.after(0, self.show_blank_image)

    def exit_app(self):
        """
        Thoát ứng dụng, đảm bảo tất cả các tài nguyên được giải phóng đúng cách.
        """
        self.stop_video() # Dừng mọi video đang hoạt động và giải phóng tài nguyên
        self.root.destroy() # Thoát cửa sổ Tkinter


if __name__ == '__main__':
    root = tk.Tk() # Tạo cửa sổ Tkinter chính
    app = FaceDetectionApp(root) # Tạo một thể hiện của FaceDetectionApp
    root.mainloop() # Bắt đầu vòng lặp sự kiện Tkinter
