---
layout: article
meta:
    title: Các mô hình học máy
---

## Linear Support Vector Machine (SVM)
Support vector machine (SVM) là một trong những thuật toán học có giám sát phổ biến và mạnh mẽ, được sử dụng rộng rãi trong các bài toán phân loại và hồi quy. Thuật toán này dựa trên việc tìm ra một siêu phẳng (hyperplane) phân chia tốt nhất giữa các nhóm dữ liệu, sao cho khoảng cách từ các điểm dữ liệu tới siêu phẳng đó là lớn nhất.

Trong thí nghiệm, chúng tôi sử dụng phiên bản soft-margin của SVM. Giả sử các cặp dữ liệu của tập huấn luyện là  $\{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$, với $x_i \in \mathbb{R}^d$ là vector đặc trưng của mẫu thứ \(i\), và $y_i \in \{-1, 1\}$ là nhãn của mẫu đó. Bài toán soft-margin SVM đặt ra là tìm ra siêu phẳng $w^Tx + b = 0$ sao cho:
$$
\begin{aligned}
& \underset{w, b}{\text{minimize}}
& & \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i \\
& \text{subject to}
& & y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad i = 1, 2, \ldots, n \\
& & & \xi_i \geq 0, \quad i = 1, 2, \ldots, n
\end{aligned}
$$

Trong đó $C$ là một hằng số cho trước và các biến $\xi_i$ được gọi là slack variable.

![Hình 1](../assets/svm/1.png)

Ta gọi hai nửa mặt phẳng nằm về hai phía của hai đường margin (đường nét mảnh trong hình vẽ) là vùng an toàn. Các điểm nằm trong vùng an toàn tương ứng với $\xi_i = 0$. Các điểm nằm đúng phía nhưng không nằm trong vùng an toàn tương ứng với $0 < \xi_i < 1$. Các điểm nằm sai phía tương ứng với $\xi_i > 1$.  Các slack variable có thể hiểu là thước đo sự hy sinh giữa việc tìm một siêu phẳng tốt và việc phân loại đúng mọi điểm dữ liệu. Hàm mục tiêu của chúng ta là sự kết hợp giữa việc tối đa margin và tối thiểu sự hy sinh.