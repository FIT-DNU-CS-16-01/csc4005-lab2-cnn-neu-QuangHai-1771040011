# CSC4005 – Lab 2 Report: CNN Image Classification (From Scratch vs Transfer)

## 1. Thông tin chung
- Họ và tên: Bế Quang Hải
- Lớp: KHMT 1701
- Repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab1-neu-mlp-QuangHai-1771040011.git
- W&B project: `csc4005-lab2-neu-cnn`
- Link W&B dashboard: https://wandb.ai/quanghaia2005-samsung-electronics/csc4005-lab2-neu-cnn?nw=nwuserquanghaia2005

---

## 2. Bài toán
**Phân loại 6 loại lỗi bề mặt thép trên bộ dữ liệu NEU-CLS:**
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

Mục tiêu: So sánh hiệu quả của **MLP baseline (Lab 1)**, **CNN from scratch**, và **Transfer learning** trên cùng một tập dữ liệu.

---

## 3. Mô hình và cấu hình

### 3.1. MLP baseline từ Lab 1
- **Model**: MLP (Multilayer Perceptron)
- **Input**: Ảnh flatten thành vector
- **Optimizer**: adamw
- **Learning rate**: 0.001
- **Epochs**: 20
- **Best validation accuracy**: 41.85%

### 3.2. CNN from scratch
- **Model**: CNN-small
- **Architecture**: 2 lớp Conv + pooling + Flatten + 2 lớp Dense
- **Train mode**: scratch
- **Optimizer**: adamw
- **Learning rate**: 0.001
- **Weight decay**: 0.0001
- **Dropout**: 0.3
- **Batch size**: 32
- **Image size**: 64×64
- **Epochs**: 20
- **Patience**: 5
- **Augmentation**: Yes
- **Best validation accuracy**: 97.41% (debug_run)

### 3.3. Transfer learning
- **Model**: ResNet18
- **Train mode**: finetune (unfrozen layers)
- **Optimizer**: adamw
- **Learning rate**: 0.0001
- **Weight decay**: 0.0001
- **Dropout**: 0.3
- **Batch size**: 16
- **Image size**: 128×128
- **Epochs**: 10
- **Patience**: 3
- **Augmentation**: Yes
- **Best validation accuracy**: 100%

---

## 4. Bảng kết quả tổng hợp

| Model | Train mode | Best Val Acc | Test Acc | Avg Epoch time (s) | Trainable Params | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MLP | scratch | 41.85% | 38.15% | N/A | N/A | Baseline Lab 1 - Kém nhất |
| CNN-small | scratch | 97.41% | 97.41% | 4.63 | 32,614 | debug_run |
| CNN-small | scratch | 94.44% | 94.81% | 4.54 | 32,614 | cnn_small_baseline |
| ResNet18 | finetune | 100% | 100% | 49.33 | 11,179,590 | resnet18_finetune ✓ Best |

---

## 5. So sánh trên W&B Dashboard

### 5.1. Các chỉ số chính để so sánh
Sau khi chạy ít nhất 2-3 hướng tiếp cận, so sánh các run trên W&B project dựa trên:

- **Best validation accuracy cao nhất**: Run nào đạt `best_val_acc` cao nhất?
- **Validation loss thấp nhất**: Run nào có `val_loss` thấp nhất?
- **Learning curves ổn định hơn**: Run nào không overfitting, val_loss ổn định?
- **Tốc độ huấn luyện**: Run nào train nhanh hơn (theo epoch)?
- **Mức độ overfitting**: So sánh `train_loss` vs `val_loss` – run nào khoảng cách nhỏ hơn?

### 5.2. Kết quả so sánh
```
- Best validation accuracy: Run 'resnet18_finetune' với 100%
- Validation loss thấp nhất: Run 'resnet18_finetune' với loss = 0.00182
- Learning curves ổn định nhất: Run 'debug_run' vì val_loss ổn định từ epoch 12 trở đi
- Train nhanh nhất: Run 'cnn_small_baseline' (~4.54s/epoch)
- Overfitting ít nhất: Run 'debug_run' (khoảng cách train_loss và val_loss nhỏ từ epoch 10+)
```

**Phân tích chi tiết:**
- ResNet18 finetune thắng lớn: 100% accuracy trên cả validation và test set
- CNN scratch vẫn rất tốt: 97.41% accuracy, nhưng chậm hơn nhiều (49.33s/epoch vs 4.63s/epoch)
- Dù ResNet18 có 11.2M parameters, nó không overfitting nhờ transfer learning tốt
- CNN scratch nhanh hơn 10.6 lần/epoch, nhưng kém 2.59% accuracy so với ResNet18

**Lưu ý:** Kết luận phải dựa trên số liệu từ W&B dashboard, **không nên viết cảm tính** hoặc đoán mò.

---

## 6. Phân tích learning curves

### 6.1. Đường cong huấn luyện
(Chèn hoặc mô tả ảnh curves từ `outputs/*/curves.png`)

**Dấu hiệu tốt:**
- train_loss giảm đều
- val_loss giảm đều hoặc ổn định
- train_acc tăng
- val_acc tăng theo

**Dấu hiệu overfitting:**
- train_acc tăng cao
- train_loss tiếp tục giảm
- nhưng val_loss bắt đầu tăng
- val_acc không tăng hoặc giảm

**Dấu hiệu underfitting:**
- cả train_acc và val_acc đều thấp
- train_loss và val_loss đều còn cao

### 6.2. Nhận xét từng run
- **CNN-small (cnn_small_baseline)**: Hội tụ từ từ, đạt best_val_acc ở epoch 16 (94.44%). Có hiện tượng overfitting nhẹ từ epoch 13 (val_loss tăng lên 1.145)
- **CNN-small (debug_run)**: Hội tụ nhanh hơn, đạt best_val_acc ở epoch 10-11 (97.41%). Val_loss ổn định từ epoch 12. Là run tốt nhất trong CNN scratch
- **ResNet18 (finetune)**: Hội tụ rất nhanh, đạt 100% val_acc từ sớm (pretrained backbone mạnh). Learning curve rất mượt, không có dấu hiệu overfitting

---

## 7. Confusion matrix và lỗi dự đoán sai

### 7.1. Confusion matrix của best model
(Chèn ảnh từ `outputs/*/confusion_matrix.png`)

### 7.2. Phân tích chi tiết
- Lớp nào được dự đoán đúng nhất?
- Lớp nào hay bị nhầm lẫn?
- Lớp nào khó nhất (F1-score thấp nhất)?

### 7.3. Ví dụ dự đoán
**Một vài ví dụ dự đoán đúng:**
- (Chèn ảnh + nhãn thực + nhãn dự đoán)

**Một vài ví dụ dự đoán sai:**
- (Chèn ảnh + nhãn thực + nhãn dự đoán + lý do có thể)

---

## 8. Lựa chọn best model

### 8.1. Nguyên tắc lựa chọn
1. **So sánh trên validation set** (không phải test set)
2. **Chọn một cấu hình tốt nhất** dựa trên các tiêu chí:
   - Val accuracy cao nhất
   - Val loss thấp và ổn định
   - Learning curves đẹp
   - Ít overfitting hơn
   - Thời gian train/epoch hợp lý
3. **Ghi rõ lý do chọn**
4. **Chỉ sau đó** mới dùng test set đánh giá cuối cùng

**⚠️ Hết sức lưu ý:** Không được chọn mô hình chỉ vì `train_accuracy` cao.

### 8.2. Best model được chọn
- **Model**: ResNet18 finetune (run: resnet18_finetune)
- **Lý do**: 
  1. Best validation accuracy cao nhất: 100% (so với 97.41% CNN scratch)
  2. Test accuracy: 100% (perfect classification)
  3. Val loss cực kỳ thấp: 0.00182
  4. Learning curve mượt và ổn định, không overfitting
  5. Transfer learning từ ImageNet giúp học đặc trưng tốt hơn
- **Best validation accuracy**: 100%
- **Test accuracy**: 100%

### 8.3. Kết quả đánh giá trên test set

**Classification Report (ResNet18 finetune):**
```
              precision    recall  f1-score   support
    Crazing       1.00      1.00      1.00        45
   Inclusion      1.00      1.00      1.00        45
     Patches      1.00      1.00      1.00        45
Pitted Surface   1.00      1.00      1.00        45
Rolled-in Scale  1.00      1.00      1.00        45
    Scratches     1.00      1.00      1.00        45
    
       accuracy                        1.00       270
      macro avg    1.00      1.00      1.00       270
   weighted avg    1.00      1.00      1.00       270
```

**Nhận xét:** Mô hình hoàn toàn chính xác trên tất cả 6 lớp. Không có lỗi dự đoán sai nào.

---

## 9. Khi nào transfer learning tốt hơn?

### 9.1. Khi transfer learning thường tốt hơn scratch
- **Dữ liệu không quá lớn**: Bộ NEU-CLS chỉ có ~300 ảnh/lớp, khá nhỏ
- **Muốn hội tụ nhanh**: Pretrained backbone đã học tính năng ảnh chung
- **Tận dụng đặc trưng từ ImageNet**: ResNet18 được train trên ImageNet rất đa dạng
- **Cần baseline mạnh trong thời gian ngắn**: Transfer learning thường nhanh hơn

### 9.2. Khi transfer learning có thể không áp dụng
- **Dữ liệu rất khác miền**: Nếu ảnh NEU khác xa ImageNet, backbone có thể không giúp
- **Backbone không phù hợp**: Chọn sai backbone (ví dụ: backbone quá lớn cho dữ liệu nhỏ)
- **Fine-tune không đúng cách**: Learning rate quá cao -> phá hủy pretrained weights
- **Overfitting lên validation set**: Backbone quá mạnh, overfit trên dữ liệu nhỏ

### 9.3. Kết luận dựa trên thực nghiệm

**Từ kết quả của em:**
- CNN scratch (debug_run) đạt best_val_acc = 97.41%
- ResNet18 finetune đạt best_val_acc = 100%
- Hiệu suất: Transfer learning **tốt hơn** CNN scratch **2.59%**
- **Lý do**: 
  1. **Transfer learning chiến thắng rõ ràng**: ResNet18 đạt 100% accuracy vs 97.41% của CNN
  2. **Pretrained backbone mạnh**: ImageNet weights giúp học đặc trưng ảnh tốt hơn
  3. **Hội tụ nhanh hơn**: ResNet18 đạt 100% sớm, CNN scratch phải đến epoch 16+ để hội tụ
  4. **Tuy nhiên chi phí tính toán cao**: ResNet18 = 49.33s/epoch, CNN = 4.63s/epoch (10.6× chậm hơn)
  5. **Kết luận**: Với dữ liệu nhỏ (NEU-CLS ~1800 ảnh), transfer learning rõ ràng vượt trội
     - Nên dùng transfer learning khi: dữ liệu nhỏ, muốn accuracy cao, không quan tâm tốc độ
     - CNN scratch hữu ích khi: cần inference nhanh, dữ liệu không khác biệt xa ImageNet

---

## 10. Câu hỏi tự kiểm tra

Hãy tự trả lời ngắn gọn các câu hỏi sau để đánh giá mức độ hiểu của bạn:

1. **Vì sao weight sharing làm CNN hiệu quả hơn MLP cho ảnh?**
   - Trả lời: Weight sharing cho phép cùng một filter/kernel được dùng lặp lại trên nhiều vùng ảnh. Nhờ đó CNN có ít parameters hơn MLP rất nhiều (32k vs 1M+), tăng tính hiệu quả. Đặc biệt, CNN học được các pattern/đặc trưng cục bộ (edges, corners, textures) mà có thể tái sử dụng ở bất kỳ chỗ nào trong ảnh.

2. **Receptive field tăng lên như thế nào khi chồng nhiều lớp conv/pooling?**
   - Trả lời: Mỗi lớp conv mở rộng receptive field thêm kích thước kernel (thường 3x3 → +2 pixels). Pooling làm tăng receptive field gấp đôi kích thước pool (thường 2x2 → ×2). Chồng nhiều lớp → receptive field tích lũy → lớp sâu nhìn được toàn bộ ảnh.

3. **Khi nào nên chọn transfer learning thay vì train from scratch?**
   - Trả lời: Transfer learning tốt hơn khi: (1) dữ liệu ít (NEU-CLS ~1800 ảnh), (2) muốn accuracy cao (100% vs 97.41%), (3) không quan tâm inference speed. Không nên dùng transfer khi: (1) dữ liệu khác xa miền source (nhưng NEU có tính chất ảnh không thể biến đổi), (2) cần speed (ResNet slow 10.6×).

4. **Vì sao cần so sánh scratch và transfer trên cùng một tập dữ liệu?**
   - Trả lời: Để loại bỏ biến số khác (dữ liệu, augmentation, hardware), chỉ so sánh được hiệu quả thực sự của hai phương pháp. Nếu dùng dữ liệu khác nhau → kết luận không công bằng. Trong lab này, cùng NEU-CLS → có thể khẳng định transfer thắng vì pretrained backbone tốt, không phải vì dữ liệu khác.

5. **Dấu hiệu nào cho thấy mô hình bị overfitting?**
   - Trả lời: (1) train_loss tiếp tục giảm nhưng val_loss tăng lên, (2) train_acc cao nhưng val_acc không tăng hoặc giảm, (3) khoảng cách train_loss - val_loss ngày càng lớn. Ở lab này, cnn_small_baseline có dấu hiệu overfitting ở epoch 13 (val_loss nhảy từ 0.149 lên 1.145).

6. **W&B giúp ích gì khi phải so sánh nhiều cấu hình?**
   - Trả lời: W&B lưu toàn bộ metrics (loss, accuracy, learning rate, epoch time) + config của mỗi run → dễ so sánh side-by-side trên dashboard. Giúp phát hiện nhanh run nào tốt nhất theo từng tiêu chí (accuracy, speed, stability). Nếu không W&B, phải đọc terminal output dở dang + hình vẽ rời rạc → rất khó so sánh khoa học. 

---

## 11. Checklist nộp bài

- [ ] Có ít nhất 1 run CNN from scratch (chạy thành công, log trên W&B)
- [ ] Có ít nhất 1 run transfer learning (chạy thành công, log trên W&B)
- [ ] Có W&B dashboard với project name `csc4005-lab2-neu-cnn`
- [ ] Có bảng so sánh chi tiết MLP vs CNN scratch vs transfer learning
- [ ] Có learning curves cho train/val loss và accuracy
- [ ] Có confusion matrix cho best model
- [ ] Có phân tích khi nào transfer learning tốt hơn (dựa trên số liệu, không cảm tính)
- [ ] Có test accuracy và classification report của best model
- [ ] Có ví dụ dự đoán đúng/sai từ best model
- [ ] Trả lời đầy đủ 6 câu hỏi tự kiểm tra
- [ ] Code chạy được, README hướng dẫn rõ ràng

---

## Ghi chú
- Lưu lại các outputs: `outputs/<run_name>/` sẽ chứa best_model.pt, history.csv, curves.png, confusion_matrix.png, metrics.json
- W&B project link: https://wandb.ai/quanghaia2005-samsung-electronics/csc4005-lab2-neu-cnn?nw=nwuserquanghaia2005
- Mọi kết luận phải dựa trên **số liệu thực nghiệm**, không dựa vào cảm tính
