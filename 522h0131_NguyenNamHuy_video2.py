import cv2
import numpy as np

def process_frame(frame):
    """
    Xử lý khung hình đầu vào và nhận diện biển báo giao thông.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa các dải màu cho các loại biển báo
    color_ranges = {
        "red": ([159, 50, 70], [180, 255, 255]),
        "red2": ([0,50,70] ,[9,255,255]),
        "red3": ([13,70,100] ,[17,255,255]),
        "blue": ([100, 150, 50], [140, 255, 255]),
        "blue2":([103,100,120], [128, 255, 255]),
        "black": ([0,0,0], [180,255,30]),
        "yellow": ([25,50,70],[35,255,255])
    }
    
    # Tạo các mặt nạ cho màu đỏ và xanh dương
    masks = {color: cv2.inRange(hsv, np.array(lower), np.array(upper)) 
             for color, (lower, upper) in color_ranges.items()}
    
    # Phát hiện và phân loại các biển báo trong khung hình
    detect_shapes(frame, masks['red'], masks['blue'],masks['blue2'], 
                  masks['red2'],masks['black'],masks['red3'], masks['yellow'])
    
    return frame

def detect_shapes(frame, red_mask, blue_mask,blue2_mask,red2_mask,black_mask,red3_mask,yellow_mask):
    """
    Phát hiện hình dạng và phân loại các biển báo dựa trên các mặt nạ màu.
    """
    masks = [red_mask, blue_mask,blue2_mask,red2_mask,black_mask,red3_mask,yellow_mask]
 
    for  mask in masks:
        # Làm sạch nhiễu của mặt nạ
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # Tìm các đường viền của các đối tượng được phát hiện
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 440:  # Bỏ qua các đối tượng nhỏ
                x, y, w, h = cv2.boundingRect(cnt)
                circularity = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2)
                classify_sign(frame, x, y, w, h, cnt, red_mask, blue_mask,blue2_mask,red2_mask,
                              black_mask,red3_mask,yellow_mask,circularity,area)
                
def classify_sign(frame, x, y, w, h, cnt, red_mask, blue_mask,blue2_mask,red2_mask,
                  black_mask,red3_mask,yellow_mask,circularity,area):
    """
    Phân loại biển báo dựa trên tỷ lệ màu và độ tròn.
    """
    roi_red = red_mask[y:y + h, x:x + w]
    roi_blue = blue_mask[y:y + h, x:x + w]
    roi_red2 = red2_mask[y:y + h, x:x + w]
    roi_blue2 = blue2_mask[y:y + h, x:x + w]
    roi_black = black_mask[y:y + h, x:x + w]
    roi_red3 = red3_mask[y:y + h, x:x]
    roi_yellow = yellow_mask[y:y + h, x:x]
    
    red_pixels = cv2.countNonZero(roi_red)
    blue_pixels = cv2.countNonZero(roi_blue)
    red2_pixels = cv2.countNonZero(roi_red2)
    blue2_pixels = cv2.countNonZero(roi_blue2)
    black_pixels = cv2.countNonZero(roi_black)
    red3_pixels = cv2.countNonZero(roi_red3)
    yellow_pixels = cv2.countNonZero(roi_yellow)
    total_pixels = w * h if w * h > 0 else 1  # Tránh chia cho 0
    
    red_ratio = red_pixels / total_pixels
    blue_ratio = blue_pixels / total_pixels
    red2_ratio = red2_pixels / total_pixels
    blue2_ratio = blue2_pixels / total_pixels
    black_ratio = black_pixels / total_pixels
    red3_ratio = red3_pixels / total_pixels
    yellow_ratio = yellow_pixels / total_pixels
    
    if circularity > 0.45 and (blue_ratio > 0.01 or blue2_ratio > 0.01) and (red_ratio > 0.01 or red2_ratio > 0.01 or red3_ratio > 0.01):
        label_object(frame, "CAM DO XE", x, y, x + w, y + h)
        
def label_object(frame, label, x1, y1, x2, y2):
    """
    Dán nhãn cho đối tượng đã nhận diện và vẽ hình chữ nhật xung quanh đối tượng.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
def video2():
    video_path = 'video2.mp4'
    output_video_path = '522h0131_NguyenNamHuy_video2.avi'
    cap = cv2.VideoCapture(video_path)

    # Kiểm tra xem video có mở thành công không
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
        
    # Lấy thông tin về kích thước khung hình và FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Định nghĩa codec và tạo đối tượng VideoWriter để lưu video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Danh sách mã sinh viên cần hiển thị
    student_ids = ["Student ID: 522h0131_NguyenNamHuy"]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Xử lý từng khung hình để phát hiện biển báo giao thông
        processed_frame = process_frame(frame)
        
        # Chèn từng mã sinh viên vào khung hình với khoảng cách dòng
        for i, student_id in enumerate(student_ids):
            y_position = frame_height - 10 - (i * 30)  # Đặt vị trí theo chiều cao khung hình và khoảng cách giữa các dòng
            cv2.putText(processed_frame, student_id, (10, y_position), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Video 2 (PRESS 'q' TO QUIT)",processed_frame)
        
        # Ghi khung hình vào video đầu ra
        out.write(processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
video2()