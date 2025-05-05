# API Phân Loại Món Ăn Việt Nam với YOLO

Một microservice FastAPI cho việc phân loại hình ảnh món ăn Việt Nam sử dụng YOLOv8.

## Tổng Quan

Dịch vụ này cung cấp REST API để phân loại món ăn Việt Nam sử dụng mô hình YOLOv8 đã được huấn luyện trước. Nó chấp nhận tải lên hình ảnh và trả về kết quả phân loại cùng với điểm tin cậy.

## Yêu Cầu Hệ Thống

- Python 3.8+
- fastapi
- uvicorn
- pillow
- numpy
- ultralytics (YOLOv8)

## Cài Đặt

1. Clone repository:
   ```
   git clone <repository-url>
   cd yolo-microservice
   ```

2. Tạo và kích hoạt môi trường ảo:
   ```
   python -m venv venv
   source venv/bin/activate  # Trên Windows: venv\Scripts\activate
   ```

3. Cài đặt các thư viện phụ thuộc:
   ```
   pip install -r requirements.txt
   ```

4. Đặt file mô hình YOLOv8 vào thư mục `data`:
   ```
   mkdir -p data
   # Đặt file mô hình "yolov8-vn-food-classification.pt" của bạn vào thư mục data
   ```

## Chạy Dịch Vụ

Khởi động dịch vụ với lệnh:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

API sẽ có thể truy cập tại:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Các Endpoint API

### GET /

Kiểm tra xem API có đang chạy không.

**Phản hồi**:
```json
{
  "message": "YOLO Vietnamese Food Classification API",
  "status": "running"
}
```

### GET /status

Lấy thông tin về trạng thái API và mô hình.

**Phản hồi**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "data/yolov8-vn-food-classification.pt",
  "model_exists": true
}
```

### POST /predict

Tải lên file hình ảnh để phân loại món ăn Việt Nam.

**Yêu cầu**:
- Form data với một file hình ảnh

**Phản hồi**:
```json
{
  "predictions": [
    {
      "class": 9,
      "score": 0.95
    }
  ],
  "processing_time": 0.542
}
```

## Hướng Dẫn Sử Dụng Chi Tiết

### 1. Chuẩn Bị Hình Ảnh

Để sử dụng API, bạn cần chuẩn bị hình ảnh món ăn Việt Nam cần phân loại. Hình ảnh nên rõ ràng, có độ phân giải tốt và chứa món ăn cần phân loại ở vị trí trung tâm.

### 2. Gửi Yêu Cầu Phân Loại

#### Sử dụng Swagger UI

1. Mở trình duyệt web và truy cập: http://localhost:8000/docs
2. Tìm và mở rộng endpoint POST /predict
3. Click vào nút "Try it out"
4. Click vào nút "Choose File" để chọn hình ảnh từ máy tính của bạn
5. Click vào nút "Execute" để gửi yêu cầu
6. Kết quả sẽ hiển thị ở phần Response body

#### Sử dụng curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@duong_dan_den_hinh_anh.jpg"
```

#### Sử dụng Python với thư viện requests

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("duong_dan_den_hinh_anh.jpg", "rb")}
response = requests.post(url, files=files)
predictions = response.json()
print(predictions)
```

### 3. Hiểu Kết Quả

Kết quả trả về sẽ có cấu trúc như sau:

```json
{
  "predictions": [
    {
      "class": 9,
      "score": 0.95
    }
  ],
  "processing_time": 0.542
}
```

Trong đó:
- `class`: ID của lớp món ăn được dự đoán
- `score`: Điểm tin cậy của dự đoán (từ 0 đến 1, càng gần 1 càng tin cậy)
- `processing_time`: Thời gian xử lý (tính bằng giây)

### 4. Xử Lý Lỗi

Nếu gặp lỗi, API sẽ trả về thông báo lỗi với mã HTTP thích hợp:
- 400: Lỗi yêu cầu (không có file, file không hợp lệ, v.v.)
- 500: Lỗi máy chủ (mô hình không được tải, lỗi xử lý, v.v.)

## Hỗ Trợ Docker

Sắp có Dockerfile để đóng gói ứng dụng này thành container. 