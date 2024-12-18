import configparser
import numpy as np
import os
from predict.segment_with_color import segment_with_color_ranges
from predict.predict import predict_image

# Hàm đọc cấu hình từ file config.ini
def read_config(config_path=r'crack_detection\utils\config.ini', color_option=1):
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    # Chuyển đổi số thành tên section tương ứng
    color_section = f"COLOR_RANGES_{color_option}"

    # Kiểm tra sự tồn tại của section
    if color_section not in config.sections():
        print(f"Tùy chọn {color_option} không hợp lệ hoặc không tìm thấy. Sử dụng mặc định COLOR_RANGES_1.")
        color_section = 'COLOR_RANGES_1'

    # Đọc các thông số chung từ DEFAULT
    input_folder = config['DEFAULT'].get('input_folder', r'crack_detection\image')
    output_segment_folder = config['DEFAULT'].get('output_segment_folder', r'crack_detection\result_segment')
    output_predict_folder = config['DEFAULT'].get('output_predict_folder', r'crack_detection\result_output')
    verbose = config['DEFAULT'].getboolean('verbose', True)
    save_segmented_images = config['DEFAULT'].getboolean('save_segmented_images', True)
    save_predicted_images = config['DEFAULT'].getboolean('save_predicted_images', True)

    # Đọc các khoảng màu HSV và threshold từ section tương ứng
    threshold = float(config[color_section].get('threshold', 0.3))  # Mặc định là 0.3
    color_ranges = []

    # Đọc số lượng khoảng màu từ cấu hình
    i = 1
    while True:
        min_key = f'color_range_{i}_min'
        max_key = f'color_range_{i}_max'

        if min_key in config[color_section] and max_key in config[color_section]:
            try:
                min_range = list(map(int, config[color_section][min_key].split(',')))
                max_range = list(map(int, config[color_section][max_key].split(',')))
                color_ranges.append((np.array(min_range), np.array(max_range)))
            except ValueError as e:
                print(f"Lỗi khi chuyển đổi khoảng màu {i}: {e}. Dừng lại ở khoảng màu {i}.")
                break
        else:
            break
        i += 1

    return {
        'color_ranges': color_ranges,
        'threshold': threshold,
        'input_folder': input_folder,
        'output_segment_folder': output_segment_folder,
        'output_predict_folder': output_predict_folder,
        'verbose': verbose,
        'save_segmented_images': save_segmented_images,
        'save_predicted_images': save_predicted_images
    }

# Hàm main để xử lý ảnh
def main(color_option=1):
    # Đọc cấu hình từ file config với option chỉ định
    config = read_config(color_option=color_option)
    color_ranges = config['color_ranges']
    threshold = config['threshold']
    input_folder = config['input_folder']
    output_segment_folder = config['output_segment_folder']
    output_predict_folder = config['output_predict_folder']
    verbose = config['verbose']
    # save_segmented_images = config['save_segmented_images']
    # save_predicted_images = config['save_predicted_images']

    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs(output_segment_folder, exist_ok=True)
    os.makedirs(output_predict_folder, exist_ok=True)

    # Phân đoạn ảnh
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            output_path_segment = os.path.join(output_segment_folder, f"processed_{filename}")
            if verbose:
                print(f"Đang xử lý phân đoạn ảnh: {filename}")
            try:
                segment_with_color_ranges(input_path, output_path_segment, color_ranges)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {filename}: {e}")

    # Thực hiện dự đoán
    for filename in os.listdir(output_segment_folder):
        input_path_segment = os.path.join(output_segment_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            output_path_predict = os.path.join(output_predict_folder, f"predicted_{filename}")
            if verbose:
                print(f"Dự đoán ảnh: {filename} với threshold = {threshold}")
            try:
                predict_image(input_path_segment, output_path_predict, conf_threshold=threshold)
            except Exception as e:
                print(f"Lỗi khi dự đoán ảnh {filename}: {e}")
             
if __name__ == '__main__':   
                
    print("Chọn tùy chọn khoảng màu HSV:")
    print("1 - COLOR_RANGES_1")
    print("2 - COLOR_RANGES_2")
    print("3 - COLOR_RANGES_3")
   
    try:
        color_option = int(input("Nhập số (1, 2, 3): ").strip())
        if color_option not in [1, 2, 3, 4, 5, 6]:
            raise ValueError
    except ValueError:
        print("Nhập không hợp lệ. Sử dụng tùy chọn mặc định 1.")
        color_option = 1

    main(color_option=color_option)

