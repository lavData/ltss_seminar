# Parallel Hashing
## Linear Hashing
Linear hashing là một thuật toán phân đoạn (segmentation algorithm) dùng để giải quyết vấn đề xung đột (collision) trong bảng băm (hash table). Thuật toán này cho phép ta tăng kích thước của bảng băm một cách linh hoạt, mà không cần phải tái tạo toàn bộ bảng băm.

Thuật toán linear hashing bao gồm các bước sau:

Khởi tạo bảng băm với một kích thước ban đầu và một hàm băm.

Khi một phần tử mới được thêm vào bảng băm và xảy ra xung đột, ta tách phần đó ra thành hai phần: một phần ở vị trí hiện tại, và một phần ở vị trí mới được tính toán bằng cách sử dụng hàm băm với kích thước bảng băm mới.

Tiếp tục thêm phần tử mới vào bảng băm bằng cách sử dụng hàm băm mới và tách phần tử ra nếu xảy ra xung đột.

Nếu số phần tử trong bảng băm đạt đến một ngưỡng nào đó, ta tăng kích thước bảng băm lên và tính toán lại hàm băm.

Dưới đây là một chương trình Python đơn giản để thực hiện thuật toán linear hashing với cấu trúc dữ liệu bảng băm đơn giản:

```python
class LinearHash:
    def __init__(self):
        self.size = 2
        self.count = 0
        self.data = [[] for i in range(self.size)]
        self.threshold = 2

    def add(self, key, value):
        if self.count >= self.size * self.threshold:
            self.size *= 2
            self.rehash()
        index = hash(key) % self.size
        bucket = self.data[index]
        for i in range(len(bucket)):
            if bucket[i][0] == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.count += 1

    def get(self, key):
        index = hash(key) % self.size
        bucket = self.data[index]
        for i in range(len(bucket)):
            if bucket[i][0] == key:
                return bucket[i][1]
        return None

    def rehash(self):
        new_data = [[] for i in range(self.size)]
        for bucket in self.data:
            for key, value in bucket:
                index = hash(key) % self.size
                new_data[index].append((key, value))
        self.data = new_data

    def __str__(self):
        result = ""
        for bucket in self.data:
            for key, value in bucket:
                result += str(key) + " -> " + str(value) + "\n"
        return result
```
Trong chương trình này, chúng ta sử dụng một danh sách các danh sách để lưu trữ phần tử trong bảng băm. Mỗi phần tử được lưu trữ dưới dạng một cặp (key, value).

Phương thức add() được sử dụng để thêm một phần tử mới vào bảng băm. Nếu số phần tử trong bảng băm đã vượt quá một ngưỡng (được đặt là threshold), chúng ta tăng kích thước bảng băm lên và tính toán lại hàm băm. Sau đó, chúng ta tính toán chỉ số của phần tử mới bằng cách sử dụng hàm băm và thêm phần tử mới vào danh sách tại vị trí này.

Phương thức get() được sử dụng để lấy giá trị của một phần tử từ bảng băm. Chúng ta tính toán chỉ số của phần tử bằng cách sử dụng hàm băm và tìm kiếm phần tử trong danh sách tại vị trí này.

Phương thức rehash() được sử dụng để tính toán lại hàm băm và di chuyển các phần tử trong bảng băm sang các vị trí mới tương ứng.

Phương thức str() được sử dụng để hiển thị bảng băm dưới dạng chuỗi.
