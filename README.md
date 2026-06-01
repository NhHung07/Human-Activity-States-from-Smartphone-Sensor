# Phân loại Trạng thái Hoạt động của Con người bằng Cảm biến Smartphone

**Tác giả**: Trần Nhật Hưng

---

## 1. Tóm tắt (Abstract)

Dự án này trình bày một quy trình khoa học dữ liệu hoàn chỉnh để giải quyết bài toán phân loại đa lớp, nhằm mục đích xác định sáu trạng thái hoạt động của con người: đi bộ, đi lên lầu, đi xuống lầu, ngồi, đứng và nằm. Dữ liệu được thu thập từ cảm biến gia tốc kế và con quay hồi chuyển tích hợp trong smartphone. Quy trình bao gồm các bước: phân tích dữ liệu khám phá (EDA), tiền xử lý, lựa chọn đặc trưng, xây dựng và tinh chỉnh tham số cho ba mô hình học máy (Logistic Regression, Random Forest, SVM), và cuối cùng là đánh giá sâu mô hình được chọn. Kết quả cho thấy mô hình Logistic Regression, sau khi được tối ưu hóa, đạt độ chính xác xấp xỉ 93.8% trên tập dữ liệu kiểm thử, chứng tỏ hiệu quả cao trong việc nhận dạng hoạt động.

## 2. Phát biểu bài toán (Problem Statement)

Với sự phổ biến của các thiết bị di động, việc tận dụng dữ liệu từ cảm biến để hiểu và phân tích hành vi con người ngày càng trở nên quan trọng. Bài toán đặt ra là: "Làm thế nào để xây dựng một mô hình dự báo chính xác trạng thái hoạt động của một người (đứng, ngồi, đi, v.v.) chỉ dựa vào các số liệu thô từ cảm biến gia tốc và con quay hồi chuyển của điện thoại thông minh?"

Mục tiêu của dự án là:
- Xây dựng một pipeline xử lý dữ liệu hiệu quả.
- So sánh và lựa chọn mô hình học máy phù hợp nhất cho bài toán.
- Đánh giá mô hình một cách toàn diện để đảm bảo độ tin cậy.
- Đóng gói mô hình để có thể tái sử dụng hoặc triển khai trong một ứng dụng thực tế.

## 3. Phân tích và khám phá dữ liệu (EDA)

Giai đoạn này tập trung vào việc "làm quen" với dữ liệu để phát hiện các mẫu, điểm bất thường và các mối quan hệ tiềm ẩn.

- **Kiểm tra cấu trúc và tổng quan**: Tập `train` có 7352 mẫu và `test` có 2947 mẫu, với 563 đặc trưng.
- **Kiểm tra dữ liệu thiếu và trùng lặp**: Kết quả xác nhận bộ dữ liệu hoàn toàn "sạch", không có giá trị `null` hay các dòng bị trùng lặp.
- **Phân phối của biến mục tiêu (`Activity`)**: Các lớp hoạt động được phân phối tương đối cân bằng, điều này thuận lợi cho việc huấn luyện mô hình mà không cần các kỹ thuật xử lý mất cân bằng phức tạp như SMOTE.
- **Phát hiện ngoại lệ (Outlier Detection)**: Biểu đồ Boxplot cho thấy sự tồn tại của các điểm ngoại lệ ở một số đặc trưng. Tuy nhiên, chúng được giữ lại vì có thể chứa thông tin hữu ích về các chuyển động đột ngột hoặc đặc biệt của người dùng.

    ![Biểu đồ boxplot để quan sát các outlier](/img/image-1.png)

    *Hình 1: Biểu đồ boxplot thể hiện sự phân bố và các điểm ngoại lệ của dữ liệu cảm biến.*
## 4. Tiền xử lý và Lựa chọn đặc trưng

### 4.1. Mã hóa Biến mục tiêu
Biến `Activity` có kiểu dữ liệu `object` (text) được chuyển đổi sang dạng số nguyên bằng `sklearn.preprocessing.LabelEncoder` để các thuật toán học máy có thể xử lý.

### 4.2. Chuẩn hóa dữ liệu (Data Scaling)
Toàn bộ các đặc trưng dự báo được chuẩn hóa bằng `sklearn.preprocessing.StandardScaler`. Quá trình này giúp đưa tất cả các đặc trưng về cùng một thang đo (phân phối chuẩn với trung bình là 0 và độ lệch chuẩn là 1), điều này rất quan trọng đối với các thuật toán nhạy cảm với thang đo như Logistic Regression và SVM.

### 4.3. Lựa chọn đặc trưng (Feature Selection)
Với 561 đặc trưng ban đầu, việc giảm chiều dữ liệu là cần thiết để tránh "lời nguyền số chiều" (curse of dimensionality), giảm thời gian huấn luyện và có thể cải thiện hiệu suất mô hình.
- **Phương pháp**: Sử dụng `sklearn.feature_selection.SelectFromModel` kết hợp với `RandomForestClassifier`.
- **Cơ chế**: Một mô hình Random Forest được huấn luyện trên toàn bộ đặc trưng. Dựa trên độ quan trọng (feature importance) của từng đặc trưng, `SelectFromModel` sẽ tự động giữ lại những đặc trưng có độ quan trọng lớn hơn một ngưỡng (mặc định là trung bình).
- **Kết quả**: Số lượng đặc trưng được giảm từ 561 xuống còn 121, giúp mô hình tập trung vào những tín hiệu cảm biến thực sự có giá trị.

![Biểu đồ barplot thể hiện top 20 tín hiệu cảm biến có mức độ quan trọng cao.](/img/image-2.png)

*Hình 2: Biểu đồ barplot thể hiện top 20 tín hiệu cảm biến có mức độ quan trọng cao.*
## 5. Xây dựng và Huấn luyện Mô hình

Ba mô hình ứng viên đã được lựa chọn để so sánh:

1.  **Logistic Regression**: Một mô hình tuyến tính, nhanh, mạnh mẽ và dễ diễn giải.
2.  **Random Forest**: Một mô hình tập hợp (ensemble) dựa trên cây quyết định, có khả năng xử lý các mối quan hệ phi tuyến phức tạp.
3.  **Support Vector Machine (SVM)**: Một thuật toán mạnh mẽ, hiệu quả trong không gian nhiều chiều.

### Tinh chỉnh tham số (Hyperparameter Tuning)
- **Logistic Regression & SVM**: Sử dụng `GridSearchCV` để tìm kiếm toàn diện trên một lưới các tham số đã định nghĩa.
- **Random Forest**: Sử dụng `RandomizedSearchCV` để tìm kiếm ngẫu nhiên một số lượng kết hợp tham số nhất định, hiệu quả hơn về mặt tính toán khi không gian tìm kiếm lớn.
- **Chiến lược kiểm định chéo (Cross-Validation)**: `StratifiedKFold` với `k=5` được sử dụng để đảm bảo mỗi fold trong quá trình kiểm định chéo có tỷ lệ phân phối lớp tương tự như tập dữ liệu gốc.

## 6. Đánh giá Mô hình

### 6.1. So sánh hiệu suất và Hiện tượng Overfitting

Để lựa chọn mô hình tốt nhất, chúng ta không chỉ nhìn vào độ chính xác trên tập kiểm thử mà còn phải xem xét sự chênh lệch giữa hiệu suất trên tập huấn luyện và tập kiểm thử. Điều này giúp phát hiện hiện tượng học vẹt (Overfitting).

| Model                 | Accuracy (Train set) | Accuracy (Test set) | Chênh lệch | Ghi chú                                                              |
| --------------------- | -------------------- | ------------------- | ---------- | -------------------------------------------------------------------- |
| **Logistic Regression** | **~97.8%**           | **~93.8%**          | **~4.0%**  | **Lựa chọn cuối cùng**. Mức độ overfitting thấp, khả năng tổng quát hóa tốt. |
| SVM                   | ~98.2%               | ~93.0%              | ~5.2%      | Overfitting nhẹ, hiệu suất trên tập test thấp hơn Logistic Regression. |
| Random Forest         | 100%                 | ~91.3%              | ~8.7%      | **Overfitting nghiêm trọng**. Mô hình học thuộc lòng tập train.       |

**Phân tích:**
- **Random Forest** cho thấy dấu hiệu overfitting rõ rệt khi đạt độ chính xác tuyệt đối (100%) trên tập huấn luyện nhưng lại giảm mạnh xuống chỉ còn 91.3% trên tập kiểm thử. Điều này cho thấy mô hình quá phức tạp và đã "học thuộc lòng" dữ liệu huấn luyện, dẫn đến khả năng tổng quát hóa kém.
- **SVM** cũng có một khoảng cách nhất định giữa train và test accuracy, cho thấy một mức độ overfitting nhẹ.
- **Logistic Regression** là mô hình có sự cân bằng tốt nhất. Mặc dù độ chính xác trên tập train không phải cao nhất, nhưng nó lại cho kết quả tốt trên tập test và có sự chênh lệch giữa hai tập là nhỏ nhất (~4.0%). Điều này chứng tỏ mô hình có khả năng tổng quát hóa tốt, không bị học vẹt.

**Lý do lựa chọn Logistic Regression**: Dựa trên phân tích trên, Logistic Regression được chọn không chỉ vì độ chính xác cao trên tập test mà quan trọng hơn là vì khả năng khái quát hóa tốt và ít bị overfitting nhất so với hai mô hình còn lại.

### 6.2. Phân tích sâu mô hình Logistic Regression

Sau khi lựa chọn Logistic Regression, chúng ta tiến hành phân tích sâu hơn để hiểu rõ hiệu suất và hành vi của mô hình.

- **Báo cáo phân loại (Classification Report)**: Báo cáo cung cấp các chỉ số `Precision`, `Recall`, và `F1-score` cho từng lớp.
    - **Precision** (Độ chính xác): Tỷ lệ các dự đoán đúng trên tổng số dự đoán cho một lớp. Ví dụ, precision của `SITTING` là 0.92, nghĩa là trong số tất cả các lần mô hình dự đoán là `SITTING`, 92% là chính xác.
    - **Recall** (Độ nhạy): Tỷ lệ các trường hợp thực tế của một lớp được mô hình nhận dạng đúng. Ví dụ, recall của `SITTING` là 0.85, nghĩa là mô hình đã tìm thấy 85% trong tổng số các hoạt động `SITTING` thực tế.
    - **F1-score**: Trung bình điều hòa của Precision và Recall, là một thước đo tổng hợp về hiệu suất.
    - **Nhận xét**: Các chỉ số cho thấy mô hình hoạt động rất tốt trên các lớp `LAYING`, `WALKING`, `WALKING_DOWNSTAIRS`, và `WALKING_UPSTAIRS` với F1-score > 0.93. Lớp `SITTING` và `STANDING` có F1-score thấp hơn một chút, cho thấy sự nhầm lẫn giữa hai hoạt động này.

    ```
                            precision    recall  f1-score   support

            LAYING             1.00      0.99      1.00       537
           SITTING             0.92      0.85      0.88       491
          STANDING             0.87      0.93      0.90       532
           WALKING             0.94      0.99      0.96       496
  WALKING_DOWNSTAIRS         1.00      0.92      0.96       420
  WALKING_UPSTAIRS           0.92      0.94      0.93       471

          accuracy                                 0.94      2947
         macro avg             0.94      0.94      0.94      2947
      weighted avg             0.94      0.94      0.94      2947
    ```

- **Ma trận nhầm lẫn (Confusion Matrix)**: Biểu đồ trực quan hóa hiệu suất của mô hình.
    - Trục chéo chính (từ trên trái xuống dưới phải) thể hiện số lượng dự đoán đúng cho mỗi lớp.
    - Các ô còn lại thể hiện số lượng dự đoán sai.
    - **Nhận xét**: Từ ma trận nhầm lẫn, ta thấy rõ sự nhầm lẫn lớn nhất xảy ra giữa hai lớp `SITTING` (thực tế) và `STANDING` (dự đoán), và ngược lại. Cụ thể, có tổng cộng 109 trường hợp `SITTING` và `STANDING` bị dự đoán nhầm lần với nhau. Điều này là hợp lý về mặt logic vì cả hai đều là các hoạt động tĩnh, các tín hiệu cảm biến có thể tương tự nhau khi người dùng không di chuyển.

    ![Confusion Matrix](/img/image-3.png)
    
    *Hình 3: Ma trận nhầm lẫn của mô hình Logistic Regression trên tập kiểm thử.*

-   **Brier Score**:
    - **Brier Score** là một thước đo lỗi, tính toán sai số bình phương trung bình giữa xác suất dự đoán và kết quả thực tế. Giá trị càng gần 0 càng tốt. Brier score trung bình (macro avg) là 0.015, một con số rất thấp, khẳng định thêm rằng các dự đoán xác suất của mô hình là đáng tin cậy.
        ```
        Brier score (macro avg): 0.015396691481472352
        LAYING                0.001061
        WALKING               0.010308
        WALKING_DOWNSTAIRS    0.010752
        WALKING_UPSTAIRS      0.018951
        STANDING              0.025239
        SITTING               0.026068
        ```

## 7. Đóng gói và Triển khai

Để mô hình có thể được tái sử dụng, các thành phần của pipeline đã được lưu lại bằng `joblib`:
- `lr_model.pkl`: Mô hình Logistic Regression đã được huấn luyện.
- `scaler.pkl`: Đối tượng `StandardScaler` đã được `fit` trên tập huấn luyện.
- `selector.pkl`: Đối tượng `SelectFromModel` đã được `fit`.

Một ứng dụng demo đơn giản được xây dựng bằng Streamlit (`demo_app.py`) để minh họa cách sử dụng các thành phần đã đóng gói để dự đoán trên dữ liệu mới.

## 8. Kết luận và Hướng phát triển

Dự án đã thành công trong việc xây dựng một mô hình Logistic Regression hiệu quả cao để phân loại hoạt động của con người. Quy trình từ EDA đến đánh giá mô hình đã được thực hiện một cách có hệ thống.

**Hướng phát triển trong tương lai**:
- **Triển khai thời gian thực**: Tích hợp mô hình vào một ứng dụng di động để phân loại hoạt động theo thời gian thực.
- **Mở rộng bài toán**: Thu thập thêm dữ liệu cho các hoạt động phức tạp hơn (ví dụ: chạy xe, đạp xe).

## Phụ lục: Hướng dẫn chạy ứng dụng Demo

1.  **Cài đặt môi trường**:
    ```bash
    pip install streamlit pandas scikit-learn joblib
    ```

2.  **Chạy ứng dụng**:
    ```bash
    streamlit run demo_app.py
    ```

Ứng dụng sẽ khởi chạy một giao diện web, cho phép bạn tải lên một file CSV (tương thích với định dạng đầu vào, ví dụ `data/sample_input.csv`) và nhận về kết quả dự đoán.
