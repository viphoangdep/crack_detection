from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from segment_with_color import segment_with_color_ranges
# Load YOLOv8 model (replace 'model.pt' with your model path)
model = YOLO(r"C:\Users\Admin\Desktop\model\utils\predict\model\model.pt")
# Load mô hình YOLOv8 (thay 'model3.pt' bằng mô hình của bạn)


def predict_image(image_path, output_path=None, conf_threshold=0.2):
    """
    Hàm dự đoán trên ảnh bằng mô hình YOLOv8, chỉ hiển thị vùng segment với độ trong suốt.

    Parameters:
    - image_path: Đường dẫn tới ảnh đầu vào.
    - output_path: Đường dẫn lưu ảnh kết quả (nếu None thì không lưu).
    - conf_threshold: Ngưỡng confidence (mặc định là 0.2).

    Returns:
    - result_image: Ảnh kết quả với vùng segment có độ trong suốt.
    """
    # Đọc ảnh đầu vào
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}.")
        return None

    # Thực hiện dự đoán trên ảnh
    results = model.predict(image, conf=conf_threshold, verbose=False)

    # Tạo một lớp phủ (overlay) trùng kích thước với ảnh gốc
    overlay = image.copy()

    # Lấy các vùng segment từ kết quả
    for result in results[0].masks.xy:
        # `result` là một list các điểm (x, y) của polygon
        polygon_points = np.array(result, dtype=np.int32)
        # Vẽ vùng segment lên overlay
        cv2.fillPoly(overlay, [polygon_points], color=(0, 255, 0))  # Màu xanh lá

    # Pha trộn ảnh gốc với overlay để tạo độ trong suốt
    alpha = 0.3  # Độ trong suốt (0.0 - 1.0)
    transparent_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Lưu ảnh kết quả nếu output_path được cung cấp
    if output_path is not None:
        cv2.imwrite(output_path, transparent_image)
        print(f"Ảnh kết quả đã được lưu tại {output_path}")

    return transparent_image




def predict_video(video_path, output_path=None, segment=True, color_ranges=None, conf_threshold=0.2):
    """
    Hàm dự đoán trên video bằng mô hình YOLOv8 với phân đoạn trước khi phát hiện (nếu segment = True).
    """
    # Đọc video đầu vào
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video từ {video_path}.")
        return

    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Tạo VideoWriter để lưu video kết quả nếu output_path được cung cấp
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho video .mp4
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        # Đọc từng frame từ video
        ret, frame = cap.read()
        if not ret:
            break

        # Nếu segment = True, thực hiện phân đoạn trước khi phát hiện
        if segment and color_ranges:
            frame = segment_with_color_ranges(frame, color_ranges)

        # Thực hiện dự đoán trên frame đã phân đoạn (hoặc frame gốc nếu không phân đoạn)
        results = model.predict(frame, conf=conf_threshold, verbose=True)

        # Vẽ kết quả lên frame (bao gồm các bounding box của YOLO)
        annotated_frame = results[0].plot()

        # Hiển thị frame (nếu cần)
        # cv2.imshow("Predicted Video", annotated_frame)

        # Lưu frame vào video kết quả nếu output_path được cung cấp
        if output_path is not None:
            out.write(annotated_frame)

        # Dừng video nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    if output_path is not None:
        out.release()
    cv2.destroyAllWindows()

#testtest
predict_video("crack_detection\image\y2mate.com - How to Fix a Crack in Concrete A DIY Guide_720pFH (online-video-cutter.com).mp4", "crack_detection\result_output", 0.2)
