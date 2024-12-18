import cv2
import numpy as np
import os

def segment_with_color_ranges(image_path, output_path, color_ranges):
    """
    Hàm phân vùng các đối tượng dựa trên mảng các khoảng màu HSV, và cắt vùng theo hình đa giác lồi.

    Parameters:
    - image_path: Đường dẫn tới ảnh đầu vào
    - output_path: Đường dẫn lưu ảnh kết quả
    - color_ranges: Mảng các khoảng màu HSV, mỗi phần tử là một tuple (lower, upper)
                    Ví dụ: [(lower_gray, upper_gray), (lower_brown, upper_brown)]

    Returns:
    - None: Lưu ảnh kết quả vào output_path
    """
    # Đọc ảnh gốc
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}.")
        return

    # Chuyển sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tạo mặt nạ kết hợp cho tất cả các khoảng màu
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    # Duyệt qua các khoảng màu HSV trong color_ranges
    for (lower, upper) in color_ranges:
        mask = cv2.inRange(hsv_image, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Tìm các đường viền của các cụm màu
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Tìm đường viền có diện tích lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)

        # Tạo đa giác lồi từ đường viền lớn nhất
        convex_hull = cv2.convexHull(largest_contour)

        # Tạo một mặt nạ trống để chứa đa giác lồi
        mask = np.zeros_like(image, dtype=np.uint8)

        # Vẽ đa giác lồi lên mặt nạ (màu trắng)
        cv2.fillPoly(mask, [convex_hull], (255, 255, 255))

        # Cắt vùng trong ảnh bằng mặt nạ
        result_image = cv2.bitwise_and(image, mask)

        # Thay thế các vùng màu đen (0, 0, 0) bằng màu trắng (255, 255, 255)
        result_image[result_image == 0] = 255

        # Cắt hình chữ nhật bao quanh vùng đa giác lồi để crop ảnh
        x, y, w, h = cv2.boundingRect(convex_hull)
        cropped_image = result_image[y:y+h, x:x+w]

        # Lưu ảnh kết quả
        cv2.imwrite(output_path, cropped_image)

