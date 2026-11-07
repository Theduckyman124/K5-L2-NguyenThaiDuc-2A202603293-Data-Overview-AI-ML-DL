# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. 
{
  "sample_id": "traffic",
  "coco_image_id": 210273,
  "image_width": 640,
  "image_height": 428,
  "task": "image_classification",
  "taxonomy_name": "ImageNet-1K",
  "model_file": "yolo11n-cls.pt",
  "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
  "ultralytics_version": "8.4.145",
  "rank": 1,
  "class_id": 468,
  "class_name": "cab",
  "score": 0.510915
},

2. Record này nói về một loại vật thể trong ảnh là "cab". Vì nó chỉ rõ class name là "cab"

3. Danh sách tên lớp (class list) được xác định là do đi kèm taxonomy

4. Class_id: mã định danh của lớp, dùng để xử lý bằng chương trình và liên kết dữ liệu
class_name : tên gọi để thân thiện với con người
taxonomy_name: tên hệ thống phân loại má class này thuộc về 

5. Guideline cần quy định rõ tiêu chí chọn một lớp duy nhất, chẳng hạn ưu tiên chủ thể chính, vật chiếm diện tích lớn nhất hoặc nổi bật nhất trong ảnh. Đồng thời, cần nêu thứ tự ưu tiên khi các tiêu chí xung đột để đảm bảo gán nhãn nhất quán.

6. Vì score chỉ phản ánh đánh giá của model, không phải sự thật. 
Nếu lấy score làm sự thật có thể dẫn đến lan truyền lỗi, đánh giá sai lệch khi sử dụng dự đoán của model để dnahs giá model. Trong AI không có model nào là chính xác 100% . Score dùng để hỗ trợ dự đoán hoặc ưu tiên kiểm tra nhãn; ground truth dùng làm chuẩn để huấn luyện và đánh giá model. Ground truth phải được xác minh độc lập theo annotation guideline, không được quyết định chỉ dựa vào score của model.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1.   
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 5,
    "class_name": "bus",
    "score": 0.912558,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96
  },

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Qua thông tin ở trên, góc trên-trái là (93.17, 187.95), góc dưới phải là (223.01, 320.91). Cùng với đó chiều rộng của box là 129.84 pixel và chiều cao là 132.96 pixel 

3. Ngưỡng yolo11n.pt: 0.2; số vật thể phát hiện: 80, ngưỡng yolo11n.pt: 0.6; số vật thể phát hiện: 25

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải thực hiện việc kiểm tra xem mô hình có trả về đúng kết quả mong muốn hay không. Từ đó chọn ngưỡng phù hợp nhất cho mô hình 

5. Quy tắc vẽ bounding box: Vẽ một hình chữ nhật bao sát toàn bộ đối tượng cần gán nhãn, sao cho góc trên-trái (x_min, y_min) và góc dưới-phải (x_max, y_max) xác định vùng chứa đối tượng. Box phải bao phủ đầy đủ đối tượng, hạn chế tối đa phần nền thừa và áp dụng nhất quán cho mọi ảnh.

6. Cách xử lý: Đưa vào guideline quy tắc: Chỉ vẽ bounding box bao sát phần vật thể nhìn thấy, không suy đoán phần bị che khuất hoặc nằm ngoài ảnh.Đây là tình huống phổ biến, có thể áp dụng thống nhất cho nhiều ảnh. Chỉ cần escalation lên Lab Coach nếu không xác định được vật thể hoặc guideline chưa rõ cách xử lý.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1.   {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ]
    ]
  }

2. So với boxpolygon cho biết thông tin chỉ tiết hơn, bó sát vật thể  so với bbox chỉ chỉ ra ô vuông bao trọn vật thể bên trong

3. instance_id dùng để định danh và phân biệt từng cá thể vật thể riêng biệt trong một ảnh, kể cả khi chúng thuộc cùng một lớp (class).

4. Vẽ mask bám sát đường viền thực tế của phần vật thể nhìn thấy trong ảnh, không lấn sang nền hoặc vật thể khác, không suy đoán phần bị che khuất hay nằm ngoài ảnh, đảm bảo chính xác và nhất quán giữa các ảnh.

5. Trên ảnh `kitchen`, 'dinning table' bị che khuất bỏi những chiếc 'bowl'. trong trường hợp naỳ ưu tiên dùng guideline để vẽ mask theo đường biên thực tế quan sát được, không để mask lấn sang vật thể khác.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn class cho toàn bộ ảnh | Ảnh có nhiều vật thể, khó chọn một lớp chính. Score của model không đảm bảo nhãn đúng | Chọn một class theo guideline, không dựa hoàn toàn vào score. | Kiểm tra class có đúng nội dung ảnh và guideline không. |
| Phát hiện vật thể | Mỗi vật thể có một class và một bounding box | cần kiểm tra box có bao sát vật thể không. | Vẽ box bao sát phần vật thể nhìn thấy và gán class. | Kiểm tra vị trí, kích thước box, class và vật thể bị bỏ sót |
| Instance segmentation | Mỗi cá thể có một mask, class và instance_id riêng | Khó xác định biên khi hai vật chạm nhau hoặc một vật bị che khuất | Vẽ mask sát biên phần nhìn thấy, tách từng instance. | Kiểm tra mask có lấn nền, chồng sai, thiếu vùng hoặc nhầm instance không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu: Không tự ý sao chép, chia sẻ, tải lên nền tảng công khai hoặc sử dụng dữ liệu, hình ảnh, JSON và tài liệu bài tập về nhà ngoài phạm vi chương trình đào tạo khi chưa được cho phép, nhằm đảm bảo tính bảo mật và tuân thủ hợp đồng đào tạo.
2. Tôi sẽ dừng xử lý dữ liệu ngay lập tức và báo cho Lab Coach hoặc người phụ trách chương trình đào tạo để được hướng dẫn, không tự ý sử dụng, chỉnh sửa hay chia sẻ dữ liệu ngoài phạm vi cho phép.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
