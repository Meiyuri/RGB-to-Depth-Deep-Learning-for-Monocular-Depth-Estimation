import cv2
import numpy as np
import onnxruntime as ort

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN 3 MÔ HÌNH CỦA BẠN
# ==========================================
MODELS = {
    ord('1'): '1_base_model_2output.onnx', # Thay tên file thực tế của bạn
    ord('2'): '2_combine_model_2ouput.onnx',
    ord('3'): '3_sota_model_2output.onnx',
    ord('4'): '4_1_light_eff_2output.onnx'
}

def load_model(model_path):
    print(f"Đang tải mô hình: {model_path}...")
    try:
        # Sử dụng CPU mặc định. Đổi thành 'CUDAExecutionProvider' nếu có GPU
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Xử lý kích thước đầu vào (trường hợp dynamic shape thường là chữ hoặc None)
        h = input_shape[2] if isinstance(input_shape[2], int) else 256
        w = input_shape[3] if isinstance(input_shape[3], int) else 256
        
        print("Tải thành công!")
        return session, input_name, (w, h)
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None, None, None

def process_frame(frame, session, input_name, target_size):
    # 1. Tiền xử lý (Pre-process)
    h_orig, w_orig = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    # Đưa về dạng [1, 3, H, W] và chuẩn hóa [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # (Tùy chọn) Chuẩn hóa ImageNet nếu mô hình của bạn yêu cầu:
    # mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    # std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    # img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1)) # HWC sang CHW
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension

    # 2. Suy luận (Inference)
    outputs = session.run(None, {input_name: img})
    depth_map = outputs[0]

    # 3. Hậu xử lý (Post-process)
    depth_map = np.squeeze(depth_map) # Xóa các chiều dư thừa
    
    # Chuẩn hóa giá trị độ sâu về 0-255 để hiển thị
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 0:
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    
    depth_map = (depth_map * 255).astype(np.uint8)
    
    # Phóng to lại bằng khung hình gốc
    depth_map = cv2.resize(depth_map, (w_orig, h_orig))
    
    # Áp dụng màu (ColorMap) để dễ nhìn hơn
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
    
    return depth_colormap

def main():
    # Tải mô hình mặc định (phím 1)
    current_key = ord('1')
    session, input_name, target_size = load_model(MODELS[current_key])

    # Mở camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    print("\n--- HƯỚNG DẪN ---")
    print("Nhấn '1', '2', '3' để đổi mô hình.")
    print("Nhấn 'q' hoặc 'ESC' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Nếu mô hình được tải thành công, tiến hành xử lý
        if session is not None:
            depth_img = process_frame(frame, session, input_name, target_size)
            
            # Ghép ảnh gốc và ảnh độ sâu cạnh nhau
            combined = np.hstack((frame, depth_img))
            
            # Hiển thị tên mô hình đang chạy
            cv2.putText(combined, f"Model: {MODELS[current_key]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Real-time Depth Estimation", combined)
        else:
            cv2.imshow("Real-time Depth Estimation", frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]: # Bấm Q hoặc ESC để thoát
            break
        elif key in MODELS.keys() and key != current_key:
            current_key = key
            session, input_name, target_size = load_model(MODELS[current_key])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()