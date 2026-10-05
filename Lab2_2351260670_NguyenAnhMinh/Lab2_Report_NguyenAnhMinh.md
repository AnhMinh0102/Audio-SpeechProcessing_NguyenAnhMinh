# BÁO CÁO THỰC HÀNH LAB 2

## XỬ LÝ ÂM THANH VÀ TIẾNG NÓI

**Đề tài:** Nhận dạng từ đơn bằng MFCC và Dynamic Time Warping (DTW)  
**Sinh viên:** Nguyễn Anh Minh  
**MSSV:** 2351260670

---

## 1. MỤC TIÊU BÀI THỰC HÀNH

Lab 2 chuyển các kiến thức về xử lý tín hiệu tiếng nói thành một hệ nhận dạng từ đơn có thể chạy được. Trọng tâm của bài là phân tích tín hiệu theo frame, sử dụng short-time energy và ZCR để khảo sát tín hiệu và xác định vùng tiếng nói, trích chọn MFCC, tự cài đặt khoảng cách DTW và xây dựng bộ nhận dạng nearest-template. Hệ thống cuối cùng nhận một file WAV chưa biết, trích đặc trưng, so sánh với các template bằng DTW và trả về nhãn dự đoán cùng khoảng cách.

Trong thực nghiệm này, hệ thống được xây dựng cho năm từ tiếng Việt: **không, một, minh, đỏ, trắng**. Đây là bài toán speaker-dependent với cùng một người nói; 3 mẫu đầu của mỗi từ được dùng làm template và 2 mẫu còn lại dùng làm tập kiểm tra.

## 2. DỮ LIỆU VÀ TỔ CHỨC THỰC NGHIỆM

| Từ | File sử dụng | Template | Test |
|---|---|---|---|
| không | khong_01 … khong_05.wav | 01–03 | 04–05 |
| một | mot_01 … mot_05.wav | 01–03 | 04–05 |
| minh | minh_01 … minh_05.wav | 01–03 | 04–05 |
| đỏ | do_01 … do_05.wav | 01–03 | 04–05 |
| trắng | trang_01 … trang_05.wav | 01–03 | 04–05 |
| **Tổng** | **25 file** | **15** | **10** |

25 bản ghi ban đầu được chuyển sang WAV mono 16 kHz. Notebook xác nhận đủ 25 file, mỗi file có sampling rate 16.000 Hz và một kênh.

Thời lượng các file nằm trong khoảng khoảng 1,54–3,84 giây. Ví dụ, `khong_01.wav` dài 3,75 giây, `mot_01.wav` dài 2,82 giây và `trang_04.wav` dài 1,54 giây.

## 3. CẤU HÌNH THAM SỐ

| Tham số | Giá trị thực nghiệm |
|---|---|
| Sampling rate | 16.000 Hz |
| Kênh | Mono |
| Frame length | 25 ms = 400 mẫu |
| Hop length | 10 ms = 160 mẫu |
| Window | Hamming |
| Pre-emphasis | α = 0,97 |
| Mel filters | 24 |
| MFCC | 13 hệ số |
| DTW local distance | Euclidean |
| Template / từ | 3 |
| Test / từ | 2 |

Với tín hiệu 16 kHz, frame 25 ms tương ứng 400 mẫu và hop 10 ms tương ứng 160 mẫu.

## 4. PHÂN TÍCH MIỀN THỜI GIAN

Tín hiệu được chia thành các frame ngắn 25 ms với bước dịch 10 ms. Trên từng frame, notebook tính RMS/short-time energy và Zero-Crossing Rate (ZCR). RMS phản ánh mức năng lượng của tín hiệu trong frame, trong khi ZCR phản ánh mức độ đổi dấu của waveform.

Notebook tạo đồ thị waveform, RMS và ZCR cho ba từ **không, một và minh**. Các đồ thị này được lưu trong thư mục `figures` với tên tương ứng `energy_zcr_không.png`, `energy_zcr_một.png` và `energy_zcr_minh.png`.

Short-time energy/RMS phù hợp để nhận biết vùng có tiếng nói vì năng lượng của speech thường cao hơn khoảng lặng. ZCR bổ sung thông tin về đặc tính thay đổi dấu của tín hiệu, hữu ích khi phân biệt các đoạn có năng lượng tương đối thấp.

## 5. ENDPOINT DETECTION

Notebook xây dựng endpoint detection dựa trên log-RMS energy kết hợp ZCR và ngưỡng thích nghi. Với file `khong_01.wav`, kết quả:

- Start: **0,41 s**
- End: **3,7546875 s**
- Original duration: **3,7546875 s**
- Speech duration sau endpoint: **3,3446875 s**

Sau đó endpoint detection được áp dụng cho toàn bộ 25 file và tạo thư mục `trimmed_dataset`.

Kết quả thống kê:

| Chỉ tiêu | Kết quả |
|---|---:|
| Thời lượng trung bình trước trim | 2,625 s |
| Thời lượng trung bình sau trim | 2,292 s |
| Silence loại bỏ trung bình | 0,332 s |

Như vậy endpoint detection đã làm giảm độ dài trung bình của chuỗi âm thanh, giúp giảm số frame phải xử lý trong các bước MFCC và DTW.

## 6. TRÍCH CHỌN MFCC

Pipeline MFCC trong notebook gồm pre-emphasis, framing, Hamming window, Mel filterbank, log và DCT, sau đó giữ lại 13 MFCC.

Đặc trưng đầu ra có dạng **[T, 13]**, trong đó `T` là số frame và 13 là số hệ số MFCC.

Kết quả thực nghiệm:

| File | Shape MFCC |
|---|---|
| không_01 | (335, 13) |
| một_01 | (231, 13) |

Số frame khác nhau vì mỗi lần phát âm có thời lượng khác nhau. Tuy nhiên số chiều đặc trưng trên mỗi frame vẫn cố định là 13.

Notebook cũng tạo heatmap MFCC cho hai từ **không** và **một**, lưu lần lượt trong `figures/mfcc_không.png` và `figures/mfcc_một.png`.

## 7. DYNAMIC TIME WARPING

DTW được tự cài đặt bằng Dynamic Programming. Trước hết, khoảng cách Euclid được tính giữa từng cặp vector MFCC để tạo local-distance matrix. Sau đó accumulated-cost matrix được tính theo truy hồi DTW và đường căn chỉnh tối ưu được truy vết bằng backtracking.

Công thức tổng quát của local distance được sử dụng là khoảng cách Euclid giữa hai vector đặc trưng.

Ví dụ thực nghiệm cho hai cặp:

| So sánh | Total DTW cost | Normalized DTW | Path length |
|---|---:|---:|---:|
| không_01 ↔ không_02 | 10010,9568 | 29,3576 | 341 |
| không_01 ↔ một_01 | 10926,9544 | 32,0439 | 341 |

Normalized DTW của hai lần nói cùng từ **không** thấp hơn khi so sánh **không** với **một**, phù hợp với kỳ vọng rằng hai utterance cùng từ có cấu trúc đặc trưng gần nhau hơn.

Trong DTW, bước chéo ghép hai frame đồng thời; bước ngang hoặc dọc cho phép một frame ở chuỗi này được căn chỉnh với nhiều frame ở chuỗi kia. Nhờ đó DTW có thể xử lý sự khác nhau về tốc độ phát âm.

## 8. BỘ NHẬN DẠNG NEAREST-TEMPLATE

Mỗi file test được trích MFCC và so sánh bằng DTW với toàn bộ 15 template. Nhãn có khoảng cách DTW nhỏ nhất được chọn làm kết quả nhận dạng.

Một số kết quả Top-3:

| Test | Nhãn thật | Top 1 | DTW | Top 2 | DTW | Top 3 | DTW |
|---|---|---|---:|---|---:|---|---:|
| khong_04.wav | không | không | 23,0357 | trắng | 28,1524 | một | 29,0869 |
| khong_05.wav | không | không | 24,1631 | một | 28,3827 | đỏ | 30,1589 |
| mot_04.wav | một | một | 23,1283 | không | 24,5859 | đỏ | 28,5145 |
| mot_05.wav | một | không | 25,1305 | một | 25,3674 | đỏ | 27,6939 |
| minh_04.wav | minh | minh | 25,5998 | không | 27,9217 | đỏ | 32,0837 |

Trường hợp `mot_05.wav` là một ví dụ khó: khoảng cách đến nhãn **không** là 25,1305, chỉ thấp hơn khoảng cách đến nhãn **một** là 25,3674. Vì vậy hệ thống dự đoán sai mẫu này thành **không**.

## 9. ĐÁNH GIÁ HỆ THỐNG

Confusion matrix thu được:

| Nhãn thật \ Dự đoán | không | một | minh | đỏ | trắng |
|---|---:|---:|---:|---:|---:|
| không | 2 | 0 | 0 | 0 | 0 |
| một | 1 | 1 | 0 | 0 | 0 |
| minh | 0 | 0 | 2 | 0 | 0 |
| đỏ | 2 | 0 | 0 | 0 | 0 |
| trắng | 0 | 0 | 0 | 0 | 2 |

Hệ thống nhận dạng đúng **7/10 mẫu**, đạt:

**Accuracy = 70,00%**

Cụ thể:

- không: đúng 2/2;
- một: đúng 1/2;
- minh: đúng 2/2;
- đỏ: sai 2/2, đều bị nhận thành không;
- trắng: đúng 2/2.

Cặp nhầm nổi bật nhất là **đỏ → không**, với hai lỗi. Đây là điểm cần chú ý khi phân tích nguyên nhân sai của hệ thống.

## 10. THÍ NGHIỆM E1 — CÓ VÀ KHÔNG ENDPOINT

Theo output thực nghiệm hiện có:

| Cấu hình | Accuracy | Mean DTW path |
|---|---:|---:|
| Không Endpoint | 70,00% | 336,8 |
| Có Endpoint | 70,00% | 286,1 |

Accuracy không thay đổi, đều đạt 70,00%, nhưng mean DTW path giảm từ 336,8 xuống 286,1 khi sử dụng endpoint. Điều này cho thấy việc loại bỏ phần silence giúp chuỗi ngắn hơn và giảm lượng frame cần căn chỉnh.

**Lưu ý về tính hợp lệ của E1:** trong notebook hiện tại, phần E1-A sử dụng các biến file đang trỏ tới dữ liệu trong `trimmed_dataset`. Vì vậy kết quả mang tên “Không Endpoint” chưa phải đối chứng hoàn toàn độc lập với dữ liệu gốc. Trước khi nộp báo cáo chính thức, nên sửa E1-A để đọc trực tiếp `dataset` gốc và chạy lại.

## 11. THÍ NGHIỆM E2 — MFCC 13 VS MFCC + Δ

Kết quả thực nghiệm:

| Cấu hình | Accuracy | Thay đổi |
|---|---:|---:|
| 13 MFCC | 70,00% | — |
| 13 MFCC + Δ | 80,00% | +10,00 điểm % |

Khi bổ sung delta feature, accuracy tăng từ **70,00% lên 80,00%**, tức tăng 10 điểm phần trăm trên 10 mẫu test.

Kết quả này cho thấy việc bổ sung thông tin về biến thiên theo thời gian có thể giúp hệ thống phân biệt một số utterance dễ nhầm. Tuy nhiên tập test chỉ có 10 mẫu nên kết quả chưa đủ để khẳng định khả năng tổng quát hóa trên tập dữ liệu lớn.

# 12. TRẢ LỜI 9 CÂU HỎI BÁO CÁO

### Câu 1. Vì sao không nên dùng toàn bộ waveform làm template chính khi hai utterance có thời lượng khác nhau?

Waveform của hai lần nói cùng một từ có thể có số mẫu khác nhau do tốc độ nói khác nhau. Nếu so sánh trực tiếp theo từng mẫu, các điểm tương ứng về mặt âm học có thể bị lệch vị trí. Waveform cũng nhạy với biên độ, pha, thiết bị thu và các sai khác nhỏ khi phát âm. Vì vậy MFCC cung cấp biểu diễn đặc trưng gọn hơn, còn DTW cho phép căn chỉnh hai chuỗi đặc trưng có độ dài khác nhau.

### Câu 2. Giải thích vai trò khác nhau của short-time energy và ZCR trong endpoint detection.

Short-time energy/RMS đo mức năng lượng của tín hiệu trong từng frame, hữu ích để phân biệt silence với vùng speech. ZCR đo tần suất tín hiệu đổi dấu; âm hữu thanh thường có ZCR thấp hơn tương đối, trong khi âm vô thanh có thể có ZCR cao. Vì vậy energy giúp tìm vùng speech thô, còn ZCR bổ sung thông tin ở các đoạn có năng lượng thấp.

### Câu 3. Vì sao Mel filterbank có khoảng cách theo Hz rộng dần khi tần số tăng?

Thang Mel được thiết kế để mô phỏng cách hệ thính giác con người cảm nhận độ cao âm thanh. Ở vùng tần số thấp, tai phân biệt thay đổi tần số nhỏ tốt hơn nên các filter được đặt dày hơn theo Hz. Khi tần số tăng, các filter có thể cách nhau rộng hơn theo Hz.

### Câu 4. Log trong MFCC có tác dụng gì về dynamic range? DCT biến M log-energy thành các hệ số gì?

Phép log nén dynamic range của năng lượng giữa các băng Mel. DCT biến vector M giá trị log-energy của các Mel filter thành các hệ số cepstral biểu diễn biến thiên của phổ log-Mel. Trong Lab, 13 MFCC được giữ lại làm vector đặc trưng cho mỗi frame.

### Câu 5. Trong ma trận DTW, ý nghĩa của bước ngang, bước dọc và bước chéo là gì?

Bước chéo ghép một frame của chuỗi thứ nhất với một frame của chuỗi thứ hai. Bước ngang cho phép một frame của chuỗi thứ nhất tương ứng với nhiều frame liên tiếp ở chuỗi thứ hai; bước dọc làm điều ngược lại. Hai bước ngang/dọc tạo khả năng kéo giãn hoặc nén trục thời gian.

### Câu 6. Tại sao phải chuẩn hóa DTW cost theo path length?

Nếu chỉ dùng tổng DTW cost, cặp utterance có nhiều frame sẽ có xu hướng tích lũy nhiều khoảng cách hơn. Chia tổng cost cho path length tạo chi phí trung bình trên một bước căn chỉnh và giúp so sánh các utterance có độ dài khác nhau công bằng hơn.

### Câu 7. Nêu ít nhất ba nguyên nhân làm cùng một từ có MFCC khác nhau giữa hai lần nói.

Các nguyên nhân chính gồm tốc độ nói khác nhau, biến thiên phát âm tự nhiên và nhiễu/điều kiện thu. Ngoài ra, microphone, phòng thu, khoảng cách tới microphone và endpoint detection cũng có thể làm phổ và MFCC thay đổi.

### Câu 8. Từ confusion matrix, chọn cặp từ dễ nhầm nhất và phân tích.

Cặp nổi bật nhất là **đỏ bị nhận thành không**, xảy ra 2/2 mẫu test. Có thể có các đoạn đặc trưng MFCC gần nhau hoặc việc căn chỉnh DTW làm template “không” cạnh tranh với “đỏ”. Tuy nhiên notebook hiện tại chưa xuất riêng waveform, MFCC và DTW path cho một cặp lỗi “đỏ–không”, vì vậy chưa thể đưa ra kết luận dựa trên hình ảnh cụ thể.

### Câu 9. Nếu muốn nhận dạng người nói mới chưa có template, DTW gặp hạn chế gì? Nội dung Chương 3 nào giải quyết tốt hơn?

Nearest-template DTW phụ thuộc trực tiếp vào các template đã lưu. Với người nói mới, MFCC thay đổi do giọng, cao độ, cách phát âm và kênh thu; nếu không có template đại diện thì khoảng cách có thể không phản ánh đúng nhãn. Các phương pháp mô hình hóa chuỗi như HMM có thể phù hợp hơn khi cần mô hình hóa biến thiên của tín hiệu thay vì phụ thuộc trực tiếp vào một số template cụ thể.

# 13. KẾT LUẬN

Bài thực hành đã xây dựng pipeline nhận dạng từ đơn gồm chuẩn hóa dữ liệu, framing, RMS/ZCR, endpoint detection, MFCC, DTW và nearest-template recognition. Với 25 file của năm từ **không, một, minh, đỏ, trắng**, hệ thống sử dụng 15 template và 10 file test.

Baseline đạt **70,00% accuracy**. Confusion matrix cho thấy lỗi tập trung ở **đỏ → không** và một mẫu **một → không**. Thí nghiệm E2 cho thấy bổ sung delta feature giúp accuracy tăng từ **70,00% lên 80,00%** trên tập test nhỏ. Thí nghiệm E1 cho thấy mean DTW path giảm khi trim, nhưng phần “Không Endpoint” cần chạy lại bằng dữ liệu gốc để đối chứng hoàn toàn hợp lệ.

Nhìn chung, MFCC kết hợp DTW minh họa rõ khả năng nhận dạng từ đơn và xử lý khác biệt về tốc độ nói. Hạn chế chính của phương pháp nearest-template là phụ thuộc vào các template có sẵn và khó tổng quát hóa khi gặp người nói mới.

# 14. TÀI LIỆU THAM KHẢO

[1] X. Huang, A. Acero, and H.-W. Hon, *Spoken Language Processing*, Prentice-Hall, 2001.

[2] L. R. Rabiner and R. W. Schafer, *Theory and Applications of Digital Speech Processing*, Pearson/Prentice Hall, 2011.

[3] D. Jurafsky and J. H. Martin, *Speech and Language Processing*, Prentice-Hall, 2008.

[4] Đề cương chi tiết học phần CSE457 – Xử lý âm thanh và tiếng nói, Trường Đại học Thủy lợi, 2023.
