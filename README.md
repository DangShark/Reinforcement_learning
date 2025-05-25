# Deep Q-Learning va Double Deep Q-Learning cho LunarLander-v3

Kho mã nguồn này chứa triển khai thuật toán **Deep Q-Learning (DQN)** để giải bài toán **LunarLander-v3** từ thư viện Gymnasium. Dự án thể hiện việc áp dụng học tăng cường (Reinforcement Learning) để huấn luyện một tác nhân (agent) hạ cánh an toàn một mô-đun mặt trăng, vượt qua các thách thức như không gian trạng thái liên tục và đảm bảo tính ổn định trong quá trình huấn luyện.

## Mục lục

* [Deep Q-Learning va Double Deep Q-Learning cho LunarLander-v3](#deep-q-learning-va-double-deep-q-learning-cho-lunarlander-v3)

  * [Mục lục](#mục-lục)
  * [Tổng quan dự án](#tổng-quan-dự-án)
  * [Môi trường](#môi-trường)
  * [Thuật toán](#thuật-toán)
  * [Chi tiết triển khai](#chi-tiết-triển-khai)

    * [Kiến trúc mạng nơ-ron](#kiến-trúc-mạng-nơ-ron)
    * [Bộ nhớ kinh nghiệm](#bộ-nhớ-kinh-nghiệm)
    * [Tác nhân](#tác-nhân)
    * [Siêu tham số](#siêu-tham-số)
  * [Kết quả](#kết-quả)
  * [Cài đặt](#cài-đặt)
  * [Cách sử dụng](#cách-sử-dụng)
  * [Cấu trúc thư mục](#cấu-trúc-thư-mục)
  * [Cải tiến trong tương lai](#cải-tiến-trong-tương-lai)
  * [Tài liệu tham khảo](#tài-liệu-tham-khảo)

## Tổng quan dự án

Mục tiêu của dự án là huấn luyện một tác nhân để giải bài toán **LunarLander-v3**, trong đó tác nhân cần điều khiển một mô-đun mặt trăng hạ cánh an toàn xuống một bãi đáp được chỉ định. Môi trường có **không gian trạng thái liên tục** (8 chiều) và **không gian hành động rời rạc** (4 hành động), phù hợp để áp dụng thuật toán Deep Q-Learning.

Triển khai sử dụng **PyTorch** cho mạng nơ-ron và **Gymnasium** cho môi trường, tích hợp các kỹ thuật như **Experience Replay** và **Target Network** để đảm bảo tính ổn định trong quá trình huấn luyện.

## Môi trường

Môi trường **LunarLander-v3** từ Gymnasium yêu cầu điều khiển một mô-đun mặt trăng để hạ cánh an toàn. Các đặc điểm chính:

* **Không gian trạng thái**: 8 biến liên tục (vị trí, vận tốc, góc, v.v.).
* **Không gian hành động**: 4 hành động rời rạc (không làm gì, kích hoạt động cơ trái, kích hoạt động cơ chính, kích hoạt động cơ phải).
* **Phần thưởng**: Tác nhân nhận phần thưởng dựa trên việc hạ cánh thành công, tiết kiệm nhiên liệu và bị phạt nếu va chạm hoặc di chuyển ra khỏi bãi đáp.

## Thuật toán

Dự án triển khai **Deep Q-Learning (DQN)**, kết hợp Q-Learning với mạng nơ-ron sâu để xấp xỉ hàm giá trị Q cho không gian trạng thái liên tục. Các kỹ thuật chính bao gồm:

* **Experience Replay**: Lưu trữ các kinh nghiệm `(state, action, reward, next_state, done)` vào bộ nhớ đệm và lấy mẫu ngẫu nhiên để phá vỡ tương quan giữa các kinh nghiệm liên tiếp.
* **Target Network**: Sử dụng một mạng mục tiêu riêng để tính toán giá trị Q mục tiêu, giúp ổn định quá trình huấn luyện.
* **Chính sách ε-greedy**: Cân bằng giữa khám phá (chọn hành động ngẫu nhiên) và khai thác (chọn hành động tốt nhất) thông qua chiến lược giảm epsilon.

Hàm giá trị Q được xấp xỉ bởi:

$$
Q(s, a; \theta) \approx r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$

trong đó:

* \$r\$: Phần thưởng tức thời
* \$\gamma\$: Hệ số chiết khấu
* \$\theta\$: Tham số của mạng chính (online network)
* \$\theta^-\$: Tham số của mạng mục tiêu (target network)

Hàm mất mát được định nghĩa là:

$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]
$$

trong đó \$y = r + \gamma \max\_{a'} Q(s', a'; \theta^-)\$.

## Chi tiết triển khai

### Kiến trúc mạng nơ-ron

Lớp `Network` định nghĩa một mạng nơ-ron kết nối đầy đủ (fully-connected) với:

* **Tầng đầu vào**: Nhận vector trạng thái 8 chiều.
* **Tầng ẩn**: Hai tầng với 128 nơ-ron mỗi tầng, sử dụng hàm kích hoạt ReLU.
* **Tầng đầu ra**: Xuất ra giá trị Q cho 4 hành động có thể thực hiện.

```python
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### Bộ nhớ kinh nghiệm

Lớp `ReplayMemory` triển khai bộ nhớ đệm để lưu trữ và lấy mẫu kinh nghiệm:

* **Dung lượng**: 200,000 kinh nghiệm.
* **Hàm push**: Thêm kinh nghiệm mới vào bộ nhớ, xóa kinh nghiệm cũ nhất nếu bộ nhớ đầy.
* **Hàm sample**: Lấy ngẫu nhiên một mini-batch (kích thước 128) để huấn luyện.

### Tác nhân

Lớp `Agent` quản lý quá trình học của DQN:

* **act**: Chọn hành động theo chính sách ε-greedy.
* **step**: Lưu kinh nghiệm vào bộ nhớ đệm và kích hoạt học sau mỗi 4 bước.
* **learn**: Cập nhật mạng chính bằng cách tối thiểu hóa hàm mất mát và thực hiện soft update cho mạng mục tiêu.
* **soft\_update**: Cập nhật dần tham số mạng mục tiêu theo công thức:

$$
\theta_{\text{target}} = \tau \cdot \theta_{\text{local}} + (1 - \tau) \cdot \theta_{\text{target}}
$$

### Siêu tham số

| Tham số                       | Giá trị |
| ----------------------------- | ------- |
| Tốc độ học (Learning Rate)    | 5e-4    |
| Kích thước Mini-batch         | 128     |
| Hệ số chiết khấu (\$\gamma\$) | 0.99    |
| Dung lượng bộ nhớ đệm         | 200,000 |
| Hệ số nội suy (\$\tau\$)      | 1e-3    |

## Kết quả

Tác nhân đã huấn luyện đạt **tỷ lệ hạ cánh thành công 100%** trong quá trình đánh giá, với mức tiêu thụ nhiên liệu trung bình khoảng **53,88 đơn vị mỗi tập**. Video thể hiện hiệu suất của tác nhân đã được lưu trong notebook (`23020356_Bùi Hải Đăng.ipynb`).

## Cài đặt
Để chạy mã nguồn, cần cài đặt các thư viện sau:
```bash
pip install gymnasium torch numpy matplotlib imageio
```

Đảm bảo sử dụng Python phiên bản tương thích (ví dụ: Python 3.8 trở lên).

## Cách sử dụng
1. Sao chép kho mã nguồn:
   ```bash
   git clone https://github.com/DangShark/lunarlander-dqn.git
   cd lunarlander-dqn
   ```
2. Mở Jupyter Notebook:
   ```bash
   jupyter notebook 23020356_Bùi Hải Đăng.ipynb
   ```
3. Chạy toàn bộ các cell để huấn luyện tác nhân hoặc tải mô hình đã huấn luyện trước (`checkpoint_DDQN_newreward.pth`) để đánh giá.
4. Để xem kết quả trực quan, đảm bảo cài đặt các thư viện cần thiết cho việc hiển thị video và xem video được tạo ra.

## Cấu trúc thư mục
```
lunarlander-dqn/
│
├── 23020356_Bùi Hải Đăng.ipynb  # Notebook chính chứa triển khai DQN
├── checkpoint_DDQN_newreward.pth  # Trọng số mô hình đã huấn luyện
├── README.md                      # Tài liệu mô tả dự án
```

## Cải tiến trong tương lai
- Triển khai **Double DQN** để giảm sai lệch trong việc ước lượng giá trị Q.
- Thử nghiệm **Prioritized Experience Replay** để ưu tiên các kinh nghiệm quan trọng.
- Tinh chỉnh siêu tham số (ví dụ: tốc độ học, kiến trúc mạng) để cải thiện hiệu suất.
- Thêm **reward shaping** để tăng tốc độ hội tụ và tối ưu hóa tiêu thụ nhiên liệu.

## Tài liệu tham khảo
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
- Tài liệu Gymnasium: [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- Tài liệu PyTorch: [pytorch.org](https://pytorch.org/)