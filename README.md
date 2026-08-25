# LightGBM baseline cho CIC-DDoS2019 — GitHub + Google Colab + AWS S3

Pipeline huấn luyện baseline LightGBM đa lớp trên bộ CIC-DDoS2019 dạng Parquet, chạy trên
**Google Colab Pro (CPU)**, lấy code từ **GitHub**, và lưu toàn bộ trạng thái trung gian lẫn
kết quả cuối trên **AWS S3**. Đây là bản port của pipeline Kaggle
`Luan-Van-Lightgbm-Parquet-Github-v2` sang Colab; hợp đồng thí nghiệm giữ nguyên.

Colab chắc chắn sẽ ngắt session giữa chừng, nên mọi thứ được thiết kế quanh một nguyên tắc:
**S3 là nguồn sự thật duy nhất**. Colab và GitHub Actions đều chỉ là worker không trạng thái;
cả hai đọc `active_run.json` + `training_state.json` + `history.json` từ S3 rồi tiếp tục đúng
từ boosting iteration kế tiếp.

## Hợp đồng thí nghiệm

- `objective=multiclass`, `learning_rate=0.05`, **đúng 100 boosting iterations**.
- Không early stopping, không tuning, không chọn vòng tốt nhất.
- Không xử lý mất cân bằng: không class/sample weight, không oversampling/undersampling.
- Toàn bộ train split dùng ở cả 100 vòng; validation chỉ theo dõi, test chỉ đánh giá một lần.
- `feature_selection = "none"`: **dùng toàn bộ 80 thuộc tính** còn lại sau tiền xử lý, không
  sàng lọc thêm cột nào. Cơ chế sàng lọc theo gain vẫn còn trong mã nguồn như một van xả RAM,
  nhưng mặc định tắt (xem *Chọn thuộc tính theo gain*). Bốn cột chỉ số/xuất xứ — `Unnamed: 0`,
  `__source_row_id`, `__source_file_id`, `__capture_day` — bị loại ở bước tiền xử lý vì chúng
  rò rỉ nhãn (xem *Cột chỉ số dòng và rò rỉ nhãn*).
- **14 lớp, không phải 19**: các cặp nhãn trùng của hai ngày capture được gộp trước khi chia
  split (xem *Gộp nhãn: 19 → 14 lớp*).
- **Macro-F1 và balanced accuracy là chỉ số chính**; accuracy chỉ là phụ (xem *Chỉ số nào
  được coi là chính*).
- CPU bắt buộc, seed cố định, `deterministic=true`, `force_col_wise=true`.
- Checkpoint mỗi 10 vòng: Booster `.txt` + `training_state.json` + `history.json` append-only.
- Model cuối luôn là `final_model_round_100.txt`.

Toàn bộ tham số nằm trong `config/*.json`, không hard-code rải rác trong mã nguồn.

## Kiến trúc

```text
data.py                    đọc Parquet, profile RAM, split chống rò rỉ, đồng bộ S3
model.py                   sàng lọc thuộc tính, dựng lgb.Dataset, Macro-F1, callback, continue_training
train.py                   huấn luyện/resume đúng 100 vòng, heartbeat, khoá worker
checkpoint.py              checkpoint local/S3, xác minh SHA-256, khoá hợp tác
viz.py                     toàn bộ hàm vẽ, không gọi plt.show()
make_report.py             đánh giá cuối và tái tạo báo cáo từ artifact
colab_runner.ipynb         notebook Colab (sinh bởi scripts/build_colab_notebook.py)
scripts/colab_orchestrator.py   watchdog đọc heartbeat S3
scripts/sync_dataset.py         push/pull dữ liệu đã tiền xử lý
config/                    data / train / train.gha / report / orchestration
.github/workflows/         watchdog.yml, fallback-worker.yml
```

Luồng chạy:

```text
GitHub (public repo)  ──git clone──>  Colab Pro
                                         │
        S3 datasets/<data_version>/  <───┤ data.py (chạy 1 lần, dữ liệu trung gian)
                                         │
                                         v
                                    train.py ── mỗi 10 vòng ──> S3 <run_id>/checkpoints
                                         │                          + metrics/history.json
                                         │                          + figures/learning_curves
                                         v (iteration = 100)
                                  make_report.py ──> metrics / figures / explainability
                                         ^
GitHub Actions watchdog (30 phút/lần) ───┘
   ├─ Colab còn sống          -> không làm gì
   ├─ im lặng > 30 phút       -> mở/cập nhật GitHub Issue nhắc bạn mở lại Colab
   ├─ im lặng > 90 phút       -> tự chạy tiếp trong runner GitHub
   └─ đã đủ 100 vòng, thiếu báo cáo -> chạy make_report.py trong runner
```

## Chuẩn bị (làm thủ công một lần)

| Hạng mục | Giá trị |
|---|---|
| GitHub repo | `richardnguyen1991/Luan-Van-Lightgbm-Parquet-Github-v3-colab` (public) |
| S3 bucket | `my-thesis-checkpoints` |
| S3 prefix | `Luan-Van-Lightgbm-Parquet-Github-v3-colab` |
| S3 region | `us-east-1` — giá trị của `AWS_DEFAULT_REGION` |
| IAM user | `kaggle-checkpoint` (dùng lại từ pipeline Kaggle v2) — policy phải cho phép prefix `Luan-Van-Lightgbm-Parquet-Github-v3-colab/` |

Sửa `repository` trong `config/orchestration.json` nếu bạn dùng tên repo khác, rồi chạy lại
`python scripts/build_colab_notebook.py` để badge "Open in Colab" trỏ đúng chỗ.

### Secret — ba nơi, không nơi nào nằm trong code

**GitHub Secrets** (watchdog + fallback runner):
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `S3_BUCKET`, `S3_PREFIX`.

**Colab Secrets** (biểu tượng chìa khoá ở sidebar, bật *Notebook access* cho từng mục):
năm secret trên, cộng `KAGGLE_USERNAME` và `KAGGLE_KEY` — hai cái này chỉ cần cho lần tiền
xử lý đầu tiên, khi phải tải Parquet gốc về.

**AWS IAM**: user `kaggle-checkpoint` dùng chung với pipeline Kaggle v2, cần
`s3:GetObject/PutObject/DeleteObject` trên
`arn:aws:s3:::my-thesis-checkpoints/Luan-Van-Lightgbm-Parquet-Github-v3-colab/*` và
`s3:ListBucket` trên chính bucket.

Không có dòng code nào biết tên IAM user — pipeline chỉ đọc access key, secret, region, bucket
và prefix. Đổi user chỉ là đổi giá trị secret. **Điều duy nhất phải kiểm** là policy của user
đó có phủ prefix `v3-colab` hay không: nếu nó được viết cho prefix `v2` thì mọi thao tác S3 sẽ
chết với `AccessDenied` ngay ở lần ghi checkpoint đầu tiên. Xem *Kiểm tra quyền S3 trước khi
train* ở cuối README.

Repo là public nhưng notebook **không** chứa khoá: nó đọc qua
`google.colab.userdata.get()`, tức secret nằm trong tài khoản Colab của bạn. Test
`tests/test_orchestration.py` chặn mọi chuỗi `AKIA` hay `AWS_SECRET_ACCESS_KEY=` lọt vào
notebook đã commit.

## Chạy trên Colab

1. Mở `colab_runner.ipynb` bằng link
   `https://colab.research.google.com/github/<owner>/<repo>/blob/main/colab_runner.ipynb`.
2. `Runtime -> Change runtime type -> CPU`, bật **High-RAM**.
3. `Runtime -> Run all`, rồi **giữ nguyên tab**. Colab Pro không có background execution;
   đóng tab là mất session (chỉ mất tối đa 10 vòng gần nhất, vì checkpoint 10 vòng/lần).

Notebook làm đúng 5 việc: clone repo (in ra commit SHA để truy vết), nạp secret, chuẩn bị
hoặc tải dữ liệu, huấn luyện, rồi hiển thị learning curves + confusion matrix +
`summary_metrics.csv` ngay trong notebook. Cell 6 là tuỳ chọn và mặc định tắt: nó quét số
thuộc tính (xem *Giảm số thuộc tính: xếp hạng trên validation*).

**Chọn thí nghiệm ở cell 3.** Ô `EXPERIMENT` có ba giá trị — `A_random_split`,
`B_cross_capture_day`, `C_open_set` (xem *Ba thí nghiệm*). Mỗi lựa chọn dùng một cặp config
riêng, một thư mục prepared riêng (`outputs/data`, `outputs/data-expB`, `outputs/data-expC`),
một hậu tố `S3_PREFIX` riêng và một `run_id` có gắn tag thí nghiệm. Nhờ vậy ba thí nghiệm
chạy được theo thứ tự bất kỳ mà không đè lên nhau — và quan trọng hơn, một dataset 6 lớp
không thể resume nhầm lên checkpoint 14 lớp rồi chết ở guard `feature_schema_hash`.

Dataset gốc lấy từ Kaggle: `dungnguyen28101991/cicddos2019-parquet` (ô `KAGGLE_DATASET` trong
cell 3). Prompt gốc trong `Prompt_Training_LightGBM_Optimized.docx` nhắc tới slug cũ
`...-parquet-per-classes`; slug trong notebook mới là slug đúng đang dùng.

Cả `data.py` lẫn `train.py` đều nhận cùng một ngân sách thời gian session: notebook đặt
`PIPELINE_SESSION_DEADLINE_EPOCH` một lần ở cell secret, và cell dữ liệu truyền thêm
`--maximum-hours`/`--stop-before-minutes` từ `config/train.json`. Nhờ vậy tiền xử lý dừng
đúng biên file nguồn trước khi Colab thu hồi runtime, thay vì bị giết giữa file.

Ý nghĩa exit code:

| Code | Nghĩa |
|---|---|
| `0` | Đã đạt iteration 100 và báo cáo cuối thành công |
| `75` | Dừng an toàn sau checkpoint — chạy lại notebook là tiếp tục, không lặp vòng |
| khác | Lỗi thật, xem log |

## Dữ liệu trung gian trên S3

`data.py` chỉ chạy tiền xử lý **một lần**. Kết quả nằm ở
`s3://<bucket>/<prefix>/datasets/<data_version>/`, trong đó `data_version` là fingerprint của
*công thức* tiền xử lý (`config/data.json` + split + preprocessing + output + hằng số
`SPLIT_ALGORITHM_VERSION` trong `data.py`), cố tình **không** tính `data_dir`. Nhờ vậy Colab, runner GitHub và máy cục bộ mount dữ liệu ở đường dẫn khác
nhau vẫn trỏ về cùng một bộ dữ liệu đã chuẩn bị, và đổi `run_id` không làm tiền xử lý lại.

```bash
python scripts/sync_dataset.py status                    # có dữ liệu trên S3 chưa?
python scripts/sync_dataset.py pull  --output-dir outputs/data
python scripts/sync_dataset.py push  --output-dir outputs/data
```

Từ session thứ hai trở đi, Colab chỉ tải phần đã tiền xử lý; Parquet gốc không cần nữa.

## Chống rò rỉ split: kiểm tra thật, không phải chứng minh suông

`audit.backend` cũ (`deterministic_proof`) trả về `passed: true` mà **không đọc một dòng dữ
liệu nào**, dựa trên lập luận "split code là hàm thuần tuý của identity hash nên một identity
không thể vào hai split". Lập luận đó chỉ đúng khi tiền đề đúng, mà tiền đề chính là thứ một
con bug làm hỏng: nếu cùng một flow được *render khác nhau* ở hai file nguồn (port là `int64`
ở file này, `float64` ở file kia → `"80"` và `"80.0"`; Flow ID dính khoảng trắng thừa) thì nó
băm ra hai identity và rơi vào hai split — đúng thứ rò rỉ mà group-aware split sinh ra để
chặn. Backend này đã bị **gỡ bỏ**; dùng nó sẽ báo lỗi ngay.

Thay thế bằng hai lớp:

- `data.canonical_group_values` chuẩn hoá từng cột group về một dạng chuỗi duy nhất trước khi
  băm, nên `80`, `80.0` và `" 80 "` cho cùng một identity.
- `audit.backend = "sampled_exact"` giữ lại mọi identity có hash (đã trộn bằng muối **độc lập
  với seed split**) chia hết cho `audit.identity_sample_divisor`, rồi **giao tập chính xác**
  giữa ba split. RAM bị chặn bởi divisor chứ không phải bởi kích thước dataset, và state được
  ghi xuống `.leakage_audit/*.u64` sau mỗi file nguồn nên session sau tiếp tục đúng audit đó.

Độ nhạy được ghi thẳng vào `sample_manifest.json`: một rò rỉ ảnh hưởng `m` identity thoát
được audit với xác suất `(1 - 1/K)^m`. Với `K = 64`, một bug hệ thống (ảnh hưởng phần lớn
identity) bị bắt với xác suất gần như 1; một identity lẻ đơn độc thì có thể lọt. Lập luận
theo cấu trúc vẫn được giữ trong trường `constructive_proof`, nhưng nó không còn là căn cứ
để kết luận đạt.

## Pre-flight: phát hiện lớp thiếu trước, không phải sau vài giờ

Yêu cầu `split.require_all_classes_each_split` trước đây chỉ được kiểm ở **câu lệnh cuối
cùng** của quá trình tiền xử lý. Trên dữ liệu đầy đủ, một lớp hiếm (`WebDDoS` chỉ có vài trăm
dòng) rơi hết vào một split đồng nghĩa mất trắng vài giờ mới biết.

Split code chỉ phụ thuộc cột nhãn và cột group, nên `preflight_split_coverage` dựng ma trận
lớp × split bằng cách đọc **đúng 2–6 cột** đó, trong vài phút, trước khi chuyển đổi thuộc
tính hay ghi bất kỳ part nào. Kết quả nằm ở `split_coverage_preflight.json`, được cache theo
fingerprint nên resume không quét lại, và khi kết thúc pipeline so khớp lại pre-flight với
split thực tế — lệch nhau là lỗi.

Mọi `assert` trong nhánh này đã đổi thành `raise` tường minh, vì `assert` bị `python -O` xoá.

## Ngân sách RAM: `require_safe_memory_profile` giờ thực sự chặn

Khoá cấu hình này trước đây có mặt trong cả ba profile train nhưng **không dòng code nào
đọc**. `data_profile.json` chỉ trả lời được câu hỏi "ma trận float32 thô có vừa RAM không",
mà ở quy mô đầy đủ câu trả lời luôn là *không* — đó chính là lý do đường Parquet Sequence tồn
tại. Con số quyết định run có sống sót hay không là thứ khác:

| Khoản | Công thức |
|---|---|
| Dataset đã binning | `(train + val) × features × 1 byte` |
| Score nội bộ LightGBM | `(train + val) × classes × 4` |
| Gradient + hessian | `train × classes × 2 × 4` |
| Buffer dự đoán float64 của Python | `(train + val) × classes × 8` |
| Init score khi resume | `(train + val) × classes × 8 × 2` |

`model.estimate_training_memory` cộng các khoản này, so với
`available × dataset.memory_guard.available_ram_fraction`, ghi toàn bộ vào
`run_config.json → memory_estimate`, và khi `require_safe_memory_profile = true` mà không vừa
thì `train.py` **thoát 75 (dừng an toàn)** chứ không để bị OOM-kill giữa iteration — checkpoint
S3 còn nguyên, watchdog thử lại trên worker lớn hơn.

## Cột chỉ số dòng và rò rỉ nhãn

`Unnamed: 0` là `RangeIndex` của pandas bị ghi ra khi bộ CIC-DDoS2019 gốc được convert từ CSV
sang Parquet. Nó **không** phải một đặc trưng mạng, nhưng bộ lọc tiền xử lý ban đầu để lọt nó
vào tập huấn luyện: nó không phải target, không phải group id, là kiểu `int64` nên qua được
kiểm tra kiểu số, và không khớp pattern nào trong `drop_name_patterns` (các pattern đó chỉ bắt
flow id, src/dst ip/port, timestamp, simillarhttp).

Hậu quả là rò rỉ nhãn nghiêm trọng. Dataset lưu **mỗi file một loại tấn công** và
`label_from_filename_if_missing = true` suy nhãn từ tên file, nên chỉ số dòng và nhãn gần như
tương ứng một-một: mỗi file đánh số lại từ 0 và có độ dài khác nhau, vài ngưỡng trên
`Unnamed: 0` là tách được các lớp. LightGBM tìm ra ngay ở các split đầu tiên và `Unnamed: 0`
chiếm ngôi đầu bảng `feature_importance_gain`. Chia train/val/test theo dòng **không** phá được
quan hệ này — trong mỗi split nó vẫn nguyên vẹn. Metric của một run còn cột này bị thổi phồng
và không nói lên điều gì về hành vi mạng.

`__source_row_id` là **đúng cùng một lỗi, lọt lưới lần thứ hai**. Notebook chuyển CSV sang
Parquet phụ thêm ba cột xuất xứ và ghi rõ ở đầu file rằng chúng *"must never be classifier
features"*: `__capture_day`, `__source_file_id` (cả hai kiểu `string`, nên bị loại tự động vì
không phải số) và `__source_row_id` — một bộ đếm dòng `int64` chạy `0..N-1` **trong từng file
nguồn**. Nó qua được `_arrow_is_numeric`, không khớp pattern nào, không nằm trong
`explicit_drop_columns`, nên trở thành thuộc tính. Trong run `lightgbm_998247ffcfd3c0c0` nó
đứng **hạng 8 về total gain**, và vì mỗi file là một loại tấn công với độ dài rất khác nhau
(TFTP 20.1 triệu dòng, Portmap 191 nghìn, WebDDoS vài trăm) nó là một prior rất mạnh về nhãn.

Bản vá liệt kê cả bốn cột trong mọi `config/data*.json`:

```json
"explicit_drop_columns": [
  "Unnamed: 0",
  "__source_row_id",
  "__source_file_id",
  "__capture_day"
]
```

`__capture_day` vẫn được **đọc** — split theo ngày dùng nó làm khoá — nhưng không bao giờ vào
ma trận thuộc tính. `tests/test_experiments.py` chốt điều này cho cả bốn cấu hình dữ liệu:
mỗi cột phải xuất hiện trong `preprocessing.json → dropped_columns` với lý do
`explicitly excluded by configuration`.

`_drop_reasons` so khớp `casefold()` nên hoa/thường không quan trọng, nhưng đây là so khớp
**đúng tên**, không phải regex: nếu bộ dữ liệu của bạn còn `Unnamed: 0.1` hay `unnamed:_0` thì
thêm từng tên vào danh sách, hoặc thêm một pattern `"^unnamed:?[_\\s]*\\d*(\\.\\d+)?$"` vào
`drop_name_patterns`. Sau khi loại, `preprocessing.json → dropped_columns` phải có mục
`{"column": "Unnamed: 0", "reason": "explicitly excluded by configuration"}` — đó là bằng chứng
kiểm tra trong mục *Nghiệm thu*.

Đổi tập cột làm đổi `feature_schema_hash`, nên đây là **một run mới** (`run_id` mới): guard
resume trong `train.py` sẽ dừng có lỗi chứ không train tiếp lên checkpoint của run cũ. Mọi kết
quả sinh trước bản vá này cần chạy lại, không so sánh trực tiếp được.

## Chọn thuộc tính theo gain (mặc định TẮT)

> **Trạng thái hiện tại:** `feature_selection = "none"` trong cả hai profile — baseline chạy
> với đủ 80 thuộc tính hợp lệ. Trên Colab Pro High-RAM, 80 cột cho đỉnh RAM dự phóng 25,7 GiB
> so với ngân sách 38,4 GiB, nên van xả này không cần dùng tới. Mục dưới đây mô tả cơ chế để
> bạn bật lại khi chạy trên máy nhỏ hơn — và để giải thích vì sao nó từng tồn tại.
>
> Giữ đủ 80 cột còn là điều kiện để so sánh công bằng với các model khác trong luận văn
> (GNN, GRU, MLP, Transformer, LSTM): khác tập đặc trưng thì chênh lệch kết quả không còn quy
> được cho model nữa.
>
> Lưu ý phân biệt hai bước khác nhau: **tiền xử lý** loại cột theo luật cấu hình (target, group
> id, `explicit_drop_columns`, `drop_name_patterns`, cột không phải kiểu số) và luôn chạy;
> **sàng lọc theo gain** mới là thứ đang tắt. 80 là số cột còn lại sau bước thứ nhất.

Bảng RAM ở trên có `features` nằm thẳng trong số hạng lớn nhất — `(train + val) × features ×
1 byte` cho dataset đã binning. Ở quy mô đầy đủ, giữ toàn bộ cột số của CIC-DDoS2019 đẩy con
số đó vượt ngân sách của cả Colab high-RAM lẫn runner GitHub, và `require_safe_memory_profile`
sẽ chặn run ngay trước khi construct Dataset. Giảm số cột là cách duy nhất hạ được số hạng đó
mà **không** đụng vào một dòng nào của hợp đồng học (100 vòng, `learning_rate=0.05`, không
early stopping, không class weight).

`model.select_model_features` làm việc này bằng một model sàng lọc **tách rời hoàn toàn** với
model chính:

| Bước | Chi tiết |
|---|---|
| Lấy mẫu | `balanced_train_rows = 500000` chia đều cho các lớp, rút không hoàn lại bằng `default_rng(seed)` |
| Nguồn | **Chỉ train split.** Validation và test không tham gia — không rò rỉ vào bước chọn cột |
| Model sàng lọc | 30 vòng, `learning_rate=0.1`, không metric, dùng xong là `del` + `gc.collect()` |
| Xếp hạng | `feature_importance("gain")`, tie-break theo tên cột nên thứ tự là toàn phần và ổn định |
| Kết quả | 20 cột đầu bảng, **giữ nguyên thứ tự gốc** của `preprocessing.json` |

Ba tính chất khiến bước này an toàn cho một pipeline bị cắt session liên tục:

1. **Tất định.** Cùng seed + cùng train split ⇒ cùng 20 cột, bất kể chạy trên Colab hay runner
   GitHub. Test `test_every_shipped_train_profile_satisfies_the_contract` khoá
   `feature_screening.seed == seed`.
2. **Không sửa được giữa chừng.** 20 cột đã chọn đi vào `feature_schema_hash`, và `train.py`
   đối chiếu hash này với `training_state.json` ở mỗi lần resume. Đổi `maximum_features` giữa
   một run đang chạy dở thì session sau **dừng có lỗi**, không âm thầm train tiếp trên tập cột
   khác.
3. **Hai worker chọn giống nhau.** `feature_selection` và toàn bộ khối `feature_screening`
   phải trùng nhau giữa `config/train.json` và `config/train.gha.json` — khoá bằng
   `test_runner_profile_differs_only_in_resource_parameters`. Nếu lệch, một run chuyển từ
   Colab sang runner sẽ sàng ra tập cột khác và chết ở guard `feature_schema_hash` (mục 2)
   thay vì resume được.

Toàn bộ vết được ghi lại: `<run_id>/config/feature_selection.json` chứa **bảng xếp hạng gain
của mọi cột ứng viên**, không chỉ 20 cột thắng, kèm số dòng thực dùng và phân bố lớp của mẫu
sàng lọc. `run_config.json → feature_selection_summary` và `preprocessing.json →
feature_selection` giữ bản tóm tắt. Nhờ đó luận văn trả lời được câu "vì sao là 20 cột này"
bằng số liệu, không phải bằng lời.

Khi RAM vẫn không đủ, hạ `dataset.feature_screening.maximum_features` là cần gạt được thiết
kế sẵn — thông báo lỗi của `require_safe_memory_profile` trỏ thẳng vào nó. Lưu ý đây là **bắt
đầu một run mới** (`run_id` mới), vì hash schema đã đổi.

`config/train.smoke.json` để `feature_selection = "none"`: bộ smoke chỉ có vài cột, sàng lọc
không có ý nghĩa gì ở đó ngoài việc làm test chậm đi.

Diễn giải: 20 cột này là kết quả của một model gain 30 vòng trên mẫu cân bằng lớp, nên chúng
là *đủ tốt để training vừa RAM*, **không** phải "20 thuộc tính quan trọng nhất của tấn công
DDoS". Muốn nói về importance thì dùng bốn thước đo ở mục cuối, và nhớ rằng chúng chỉ xếp
hạng trong phạm vi 20 cột đã lọt vào model.

### Trên bộ đầy đủ, giảm `maximum_features` gần như không giúp gì

Thông báo lỗi của `require_safe_memory_profile` gợi ý hạ
`dataset.feature_screening.maximum_features`. Với CIC-DDoS2019 đầy đủ (49,3 triệu dòng train,
10,6 triệu validation, **19 lớp**), lời khuyên đó gần như không có tác dụng — số liệu tính từ
chính `model.estimate_training_memory`:

| `maximum_features` | Đỉnh RAM dự phóng |
|---|---|
| 20 | 22,3 GiB |
| 15 | 22,0 GiB |
| 10 | 21,7 GiB |
| 8 | 21,6 GiB |

Cắt hơn một nửa số cột chỉ tiết kiệm 0,7 GiB. Lý do nằm ở bảng công thức phía trên: khoản phụ
thuộc `features` là dataset đã binning, `(train + val) × features × 1 byte` — chỉ 1,2 GiB ở
mức 20 cột. Mọi khoản còn lại tỉ lệ với **số lớp**, không phải số cột, và 19 lớp × 60 triệu
dòng mới là thứ chiếm chỗ: riêng buffer dự đoán float64 đã là 9,1 GiB.

Nói cách khác, cần gạt duy nhất có tác dụng là **RAM của máy**. Colab tiêu chuẩn (12,7 GB) cho
ngân sách 9,5 GiB — không đủ ở bất kỳ số cột nào. Colab Pro **High-RAM** (~51 GB) cho ngân sách
38,4 GiB, thừa sức chứa 22,3 GiB. Bật High-RAM là điều kiện bắt buộc, không phải tuỳ chọn.

`session.minimum_available_ram_gb = 16.0` chặn trước cả bước này: trên runtime tiêu chuẩn
`train.py` thoát 75 ngay trước `claim_run`, nên **không** có `active_run.json` nào được tạo —
dấu hiệu nhận biết là notebook in "No active run pointer was created in this session".

## Giảm số thuộc tính: xếp hạng trên validation

Câu hỏi "cắt bớt bao nhiêu cột mà không mất độ chính xác" không trả lời được bằng một tập
đánh giá duy nhất, vì hai cách hỏng nằm ở hai chỗ khác nhau.

**Không dùng test để chọn.** `make_report.py` đã tính permutation importance, nhưng trên
`raw/explain_sample.parquet` vốn rút từ **test split**. Bảng đó dùng để *giải thích* một mô
hình đã xong thì tốt; dùng để *chọn* thuộc tính thì hỏng — nó gấp tập giữ lại vào quyết định,
và mọi con số báo cáo sau đó trên chính tập test ấy đều lạc quan một lượng không đo được.

**Gain cũng không phải selector tốt.** `train_gain_top_k` fit trên train split (sạch), nhưng
gain thiên vị cột có nhiều giá trị phân biệt, và với các nhóm tương quan mạnh của
CIC-DDoS2019 (`Fwd Packet Length Max/Min/Mean/Std`, `Flow IAT Mean/Std/Max/Min`) nó chia đều
công trạng khiến cả nhóm rơi xuống giữa bảng.

`feature_ranking.py` bù đúng mắt xích còn thiếu: permutation importance **đo trên validation**.

```bash
python feature_ranking.py --run-dir outputs/runs/<run_id> --prepared-data-dir outputs/data
```

Nó nạp Booster vòng 100, lấy mẫu phân tầng theo lớp từ validation (mặc định 200,000 dòng,
tối thiểu một dòng mỗi lớp, xác định theo seed), đo Macro-F1 và balanced accuracy gốc, rồi với
từng cột: hoán vị ngẫu nhiên cột đó `repeats` lần, dự đoán lại, ghi mức sụt. Kết quả:

| Artifact | Nội dung |
|---|---|
| `explainability/permutation_importance_validation.csv` | mức sụt trung bình + độ lệch chuẩn của từng cột |
| `config/feature_ranking_validation.json` | bảng xếp hạng gọn kèm xuất xứ, để bộ chọn tiêu thụ |

Cột `within_noise` đánh dấu những thuộc tính có mức sụt **không lớn hơn độ lệch chuẩn của
chính phép đo** — tức là không có bằng chứng chúng hữu ích. Nó chỉ đánh dấu chứ không tự cắt:
ngưỡng cắt là quyết định của người làm thí nghiệm, không phải của bảng xếp hạng.

### Dùng bảng xếp hạng để train lại

```json
"feature_selection": "validation_permutation_top_k",
"dataset": {
  "feature_screening": {
    "maximum_features": 30,
    "ranking_file": "outputs/runs/<run_id>/config/feature_ranking_validation.json"
  }
}
```

Bộ chọn **từ chối** một file xếp hạng không được đo trên validation (`scored_split != "validation"`),
và từ chối một file không phủ đúng tập cột ứng viên — đó là dấu hiệu nó đến từ một công thức
tiền xử lý khác. `feature_selection.json` của run rút gọn ghi `fit_split: "validation"` cùng
`run_id` và `feature_schema_hash` của run đã sinh ra bảng xếp hạng, nên chuỗi quyết định truy
vết được đầy đủ.

### Quy trình đề xuất cho luận văn

| Bước | Tập dữ liệu | Việc |
|---|---|---|
| 1 | **A**, train | Chạy đủ 79 thuộc tính, lấy `feature_ranking_validation.json` |
| 2 | **A**, validation | Quét `maximum_features` ∈ {60, 40, 30, 20, 15, 10}, chọn k |
| 3 | **B**, test (một lần) | Xác nhận k trên ngày giữ lại so với baseline 79 cột của B |
| 4 | **C** (tuỳ chọn) | So `mean_predictive_entropy` giữa hai tập cột |

Bước 1 và 2 gói trong một lệnh — `scripts/sweep_feature_count.py` tự chuẩn bị mẫu con, chạy
baseline, sinh bảng xếp hạng, rồi train một mô hình cho mỗi k:

```bash
python scripts/sweep_feature_count.py \
    --data-config config/data.json --train-config config/train.json \
    --output-root outputs/sweep --target-total-rows 7000000 \
    --k 60 40 30 20 15 10 \
    --tolerance-macro-f1 0.005 --tolerance-class-recall 0.02
```

Trên Colab, cell 6 gọi đúng lệnh này (mặc định tắt, bật bằng ô `RUN_SWEEP`).

Kết quả nằm ở `sweep_feature_count.csv` và `sweep_feature_count.json`. `chosen_k` là **k nhỏ
nhất** thoả cả hai ngưỡng; nếu không k nào thoả thì nó là `null` — một kết quả hợp lệ, nghĩa
là ở quy mô này không cắt được cột nào mà không trả giá.

Ba tính chất của bộ điều khiển:

- **Mọi so sánh đo trên validation.** `val_macro_f1` và `val_balanced_accuracy` lấy từ dòng
  cuối `history.json` (LightGBM đã tính trên toàn bộ validation mỗi vòng); recall từng lớp lấy
  từ `feature_ranking.validation_scores`. Không chỗ nào đọc `summary_metrics.csv`, vốn tính
  trên **test** — đọc nó là chọn k bằng chính tập giữ lại. Phép kiểm
  `test_the_sweep_never_reads_the_test_metrics` khoá tính chất này bằng cách chứng minh hai
  con số đó khác nhau.
- **Ngưỡng khai báo trước khi chạy**, truyền qua tham số dòng lệnh chứ không chọn sau khi
  nhìn kết quả.
- **Tiếp tục được.** Run nào đã đủ 100 vòng thì bỏ qua, nên session Colab chết giữa chừng chỉ
  cần chạy lại cell.

Ba điều quyết định quy trình này:

- **A một mình không kết luận được.** Khoảng cách train–validation của A ở vòng 100 là 0.0076
  logloss, nhỏ hơn một sai số chuẩn. Validation của A không phân biệt được giữa "tập cột rút
  gọn thật sự đủ" và "vẫn nội suy được vì các luồng gần trùng nằm ở cả hai phía".
- **B một mình không chọn được.** Nó chỉ có 6/14 lớp; một cột chỉ thiết yếu cho NTP sẽ bị coi
  là vô dụng mà không có cách nào biết. Đây là giới hạn của bộ dữ liệu, cần nêu thành
  *limitation*.
- **Chỉ xác nhận trên B đúng một lần.** Quét k trên test của B sẽ biến nó thành một tập
  validation thứ hai và bạn quay lại đúng vấn đề của A.

Bước 2 chạy đủ quy mô sẽ tốn ~2 giờ mỗi giá trị k. Rẻ hơn: đặt `dataset.target_total_rows`
khoảng 7 triệu (10%) cho vòng quét, chốt k, rồi mới chạy full-scale hai cấu hình (79 và k).

**Đặt ngưỡng dung sai trước, đừng dùng kiểm định ý nghĩa.** Với 10.5 triệu dòng validation,
mọi chênh lệch đều "có ý nghĩa thống kê" kể cả khi vô nghĩa thực tế. Hãy tuyên bố trước, ví
dụ: *chấp nhận nếu Macro-F1 giảm ≤ 0.005 tuyệt đối và không lớp nào mất quá 0.02 recall*.

Cấu hình phép đo nằm ở `config/report.json → validation_permutation`
(`maximum_rows`, `repeats`, `seed`, `predict_chunk_rows`). Chi phí xấp xỉ
`features × repeats` lượt dự đoán: 79 × 5 = 395 lượt trên `maximum_rows` dòng.

## Watchdog và fallback

`.github/workflows/watchdog.yml` chạy mỗi 30 phút, đọc `active_run.json` trên S3 và chọn một
trong: `wait`, `notify`, `fallback_train`, `fallback_report`, `complete`, `stop`.

- `notify` mở **một** GitHub Issue nhãn `colab-watchdog` (cập nhật chính issue đó, không tạo
  issue mới mỗi lần) kèm link Colab và số vòng đã xong.
- `fallback_train` / `fallback_report` kích `fallback-worker.yml`.
- `complete` đóng issue.

`.github/workflows/fallback-worker.yml` chạy trong runner GitHub:

- `mode=report` chỉ đọc `y_true.npy`, `y_prob.npy`, `final_model_round_100.txt` nên luôn chạy
  được. **Đây là giá trị lớn nhất của fallback**: hình và metric cuối không phụ thuộc việc bạn
  có mở được Colab hay không.
- `mode=train` là *best effort*. Runner chỉ có 4 vCPU / 16 GB RAM / ~14 GB đĩa, ít hơn nhiều
  Colab Pro high-RAM, nên job kiểm tra dung lượng trước và thoát sạch (exit 75) thay vì bị
  OOM-kill giữa một boosting iteration.

Hai worker không bao giờ train cùng lúc: `train.py` gọi `CheckpointManager.claim_run()` —
worker nào ghi heartbeat gần nhất thì giữ quyền, worker đến sau thoát 75. Heartbeat được ghi
mỗi `session.heartbeat_seconds` (mặc định 300 giây) bởi một thread nền trong `train.py`, nên
watchdog phân biệt được "đang chạy chậm" với "đã chết".

## Các profile cấu hình

| File | Vai trò |
|---|---|
| `config/data.json` + `config/train.json` | Thí nghiệm A trên Colab Pro |
| `config/train.gha.json` | Thí nghiệm A trên runner GitHub (fallback) |
| `config/data.expB.json` + `config/train.expB.json` | Thí nghiệm B, có bật `monitor_split` |
| `config/data.expC.json` + `config/train.expC.json` | Thí nghiệm C, open-set |
| `config/data.smoke.json` + `config/train.smoke.json` | Chạy thử cục bộ |

`config/train.json` (Colab Pro) và `config/train.gha.json` (runner GitHub) có **mọi
hyperparameter học giống hệt nhau**, kể cả khối `feature_screening` quyết định 20 cột nào vào
model — chỉ khác `num_threads` và ngân sách thời gian/RAM.
LightGBM ghi rõ `deterministic=true` cho kết quả ổn định *kể cả khi `num_threads` khác nhau*,
nên một run có thể chuyển qua lại giữa hai môi trường. Test
`test_runner_profile_differs_only_in_resource_parameters` khoá tính chất này.

Các profile thí nghiệm cũng giữ nguyên hợp đồng học (100 vòng, `learning_rate=0.05`,
`num_leaves=31`, không early stopping): chúng chỉ khác ở **dữ liệu nào được đưa vào** và
**tập nào được theo dõi**, đúng như một thí nghiệm đối chứng phải thế.

## Lưu ý về tái lập khi resume

`use_quantized_grad=true` (kế thừa từ v2, nằm trong hợp đồng tham số) khiến việc resume
**không** bit-exact so với một run chạy liền một mạch. Đây là bản chất của LightGBM chứ không
phải của pipeline: `lightgbm.train(init_model=...)` gốc cũng lệch đúng như vậy. Đo trên bộ
smoke, sai lệch tích luỹ sau 100 vòng vào khoảng `1e-3` với logloss và `1e-2` với Macro-F1.

Nếu luận văn cần kết quả bit-exact bất kể cắt session ở đâu, đặt `use_quantized_grad: false`
(và `quant_train_renew_leaf: false`) trong cả hai profile **và** cập nhật `required_exact`
trong `model.validate_training_config`. Khi tắt quantized gradient, resume tái lập chính xác
tuyệt đối — điều này được khoá bằng test
`ContinueTrainingTest.test_resumed_training_matches_an_uninterrupted_run`.

## Vì sao có `model.continue_training`

Không dùng được `lightgbm.train(init_model=...)` ở đây. Nó gắn model cũ làm predictor
*trước khi* Dataset được construct, rồi gọi `predictor.predict()` trên dữ liệu thô của Dataset
để tính init score. Dữ liệu thô của pipeline này là danh sách Parquet `Sequence` (để giới hạn
RAM), mà `predict()` không đọc được — mọi session resume sẽ chết với
`Cannot convert data list to numpy array`.

`continue_training` làm đúng những bước LightGBM làm, nhưng theo thứ tự tương thích với
Sequence: tự tính init score theo chunk cho cả train lẫn validation, construct Dataset xong
mới gắn model cũ vào để `Booster` merge các cây trước đó. Kết quả trùng khớp tuyệt đối với
`lightgbm.train` trong cả hai tình huống (chạy mới và resume) — xem `ContinueTrainingTest`.

## Chạy cục bộ

Yêu cầu Python 3.10+ và `pip install -r requirements.txt`.

```bash
python -m unittest discover -s tests -v

python data.py  --config config/data.smoke.json  --data-dir <thư-mục-parquet> \
                --output-dir outputs/data-smoke --samples-per-file 2000

python train.py --config config/train.smoke.json --prepared-data-dir outputs/data-smoke \
                --output-dir outputs/runs-smoke --max-rounds-this-session 20   # thoát 75

python train.py --config config/train.smoke.json --prepared-data-dir outputs/data-smoke \
                --output-dir outputs/runs-smoke                                 # tiếp vòng 21

python make_report.py --run-dir outputs/runs-smoke/<run_id> --no-upload-to-s3

python feature_ranking.py --run-dir outputs/runs-smoke/<run_id> \
                          --prepared-data-dir outputs/data-smoke

python scripts/sweep_feature_count.py --data-config config/data.smoke.json \
    --train-config config/train.smoke.json --output-root outputs/sweep-smoke \
    --target-total-rows 20000 --k 40 20 10
```

Ba thí nghiệm trên dữ liệu đầy đủ, chạy cục bộ:

```bash
# A — in-distribution, 14 lớp
python data.py  --config config/data.json  --output-dir outputs/data --full-dataset
python train.py --config config/train.json --prepared-data-dir outputs/data \
                --output-dir outputs/runs

# B — tổng quát hoá liên ngày, 6 lớp giao, có đường cong crossday
python data.py  --config config/data.expB.json  --output-dir outputs/data-expB --full-dataset
python train.py --config config/train.expB.json --prepared-data-dir outputs/data-expB \
                --output-dir outputs/runs

# C — open-set: Portmap chưa từng được huấn luyện
python data.py  --config config/data.expC.json  --output-dir outputs/data-expC --full-dataset
python train.py --config config/train.expC.json --prepared-data-dir outputs/data-expC \
                --output-dir outputs/runs
```

Tái tạo báo cáo từ artifact S3 mà không huấn luyện lại:

```bash
python make_report.py \
  --run-dir s3://my-thesis-checkpoints/Luan-Van-Lightgbm-Parquet-Github-v3-colab/<run_id> \
  --upload-to-s3
```

## Artifact trên S3

```text
s3://<bucket>/<prefix>/
├── active_run.json              con trỏ + heartbeat (watchdog đọc file này)
├── orchestration_state.json     bộ đếm session/stagnant/report
├── datasets/<data_version>/     DỮ LIỆU TRUNG GIAN dùng chung mọi run
│   ├── progress.json, data_profile.json, label_mapping.json,
│   ├── preprocessing.json, sample_manifest.json, dataset_version.json,
│   ├── split_coverage_preflight.json
│   ├── .leakage_audit/{sample,group}_{train,validation,test}.u64
│   └── splits/{train,validation,test}/part-*.parquet
└── <run_id>/
    ├── checkpoints/    last_model.txt, model_round_*.txt,
    │                   final_model_round_100.txt, training_state.json
    ├── metrics/        history.*, test_metrics.json, summary_metrics.csv,
    │                   per_class_metrics.csv, classification_report.txt,
    │                   confusion_matrix*.csv, roc_curves.csv, pr_curves.csv
    ├── figures/        *.png (300 dpi), *.pdf (vector), *.csv dữ liệu hình
    ├── raw/            y_true.npy, y_prob.npy, explain_sample.parquet
    ├── explainability/ gain, split, permutation, SHAP
    └── config/         run_config.json, model_params.json, preprocessing.json,
                        sample_manifest.json, label_mapping.json,
                        feature_selection.json (xếp hạng gain của mọi cột ứng viên)
```

## Gộp nhãn: 19 → 14 lớp

Bộ dữ liệu có hai ngày capture và **đặt tên khác nhau cho cùng một cuộc tấn công**. Đếm trực
tiếp cột `Label` của cả 18 file CSV gốc (tổng 70,427,637 dòng, khớp từng nhãn với
`data_profile.json`) cho kết quả dứt khoát:

| Nhãn | 01-12 | 03-11 |
|---|---:|---:|
| `DrDoS_LDAP` / `LDAP` | 2,179,930 | 1,915,122 |
| `DrDoS_MSSQL` / `MSSQL` | 4,522,492 | 5,787,453 |
| `DrDoS_NetBIOS` / `NetBIOS` | 4,093,279 | 3,657,497 |
| `DrDoS_UDP` / `UDP` | 3,134,645 | 3,867,155 |
| `UDP-lag` / `UDPLag` | 366,461 | 1,873 |

Cách viết **trùng khít với ngày capture**: 01-12 luôn dùng tiền tố `DrDoS_` và `UDP-lag`,
03-11 luôn dùng tên trần và `UDPLag`. Đây là artifact của quy trình gán nhãn, không phải hai
hiện tượng mạng khác nhau. Giữ 19 lớp nghĩa là phạt mô hình vì một lỗi đặt tên: 5 cặp không
thể tách được về mặt đặc trưng luồng, và chúng chiếm ~46% số dòng — đủ để giải thích vì sao
`multi_error` đứng yên quanh 19.8% từ vòng 15 trở đi.

`labels.merge_map` trong `config/data*.json` gộp 9 tên về 5, cho **14 lớp**: BENIGN, DNS,
LDAP, MSSQL, NTP, NetBIOS, Portmap, SNMP, SSDP, Syn, TFTP, UDP, UDPLag, WebDDoS. Phép gộp
chạy **trước** khi split code được gán, nên nó ảnh hưởng cả pre-flight lẫn split thực tế, và
`preprocessing.json → label_policy` ghi lại chính xác ánh xạ đã dùng.

## Ba thí nghiệm

| | A — `data.json` | B — `data.expB.json` | C — `data.expC.json` |
|---|---|---|---|
| Câu hỏi | in-distribution | tổng quát hoá liên ngày | open-set |
| Lớp | 14 | 6 lớp giao | 13 huấn luyện, Portmap giữ lại |
| Split | group-aware ngẫu nhiên 70/15/15 | train/val = 01-12, test = 03-11 | train/val = 01-12, test = chỉ Portmap |
| Chỉ số | Macro-F1, balanced accuracy, per-class | Macro-F1 + recall từng lớp | phân bố dự đoán, độ tin cậy, entropy |
| Train config | `train.json` | `train.expB.json` | `train.expC.json` |

### Vì sao split theo ngày bắt buộc phải gộp nhãn trước

Với 19 nhãn thô, **giao của hai ngày chỉ có `BENIGN` và `Syn`**. Train trên 01-12 rồi test
trên 03-11 sẽ khiến 11/13 lớp đã học không tồn tại trong tập test. Phép gộp 14 lớp không phải
phương án song song với split theo ngày — nó là điều kiện tiên quyết.

Ngay cả sau khi gộp, tập nhãn hai ngày vẫn lệch: `DNS`, `NTP`, `SNMP`, `SSDP`, `TFTP`,
`WebDDoS` chỉ có ở 01-12; `Portmap` chỉ có ở 03-11. Vì vậy thí nghiệm B giới hạn vào
`labels.keep_only` = 6 lớp chung (BENIGN, LDAP, MSSQL, NetBIOS, Syn, UDP) — còn 15.9 triệu
dòng train và 20.2 triệu dòng test. `UDPLag` bị loại có chủ ý dù về danh nghĩa là lớp chung:
phía 03-11 chỉ có 1,873 dòng so với 366,461 dòng phía 01-12, và file
`03-11/UDPLag.csv` thực tế chứa 606,749 dòng `Syn` cùng 112,475 dòng `UDP` — không đủ căn cứ
để coi 1,873 dòng đó là cùng hiện tượng.

### `split.strategy = "by_capture_day"`

```json
"split": {
  "strategy": "by_capture_day",
  "capture_day_column": "__capture_day",
  "capture_day_assignment": {"01-12": "train_validation", "03-11": "test"},
  "validation_fraction_of_train_day": 0.15
}
```

Ngày mang vai trò `test` đi trọn vào test; ngày `train_validation` được cắt 85/15 bằng đúng
group hash cũ, nên ranh giới train↔validation vẫn an toàn ở mức flow. Ngày **không** có vai
trò nào thì mọi dòng bị loại — đó là cơ chế cho phép thí nghiệm C giữ 03-11 ngoài cuộc mà vẫn
nhận riêng lớp Portmap.

**Phạm vi audit rò rỉ thay đổi theo chủ đích.** Với `by_capture_day`, một Flow ID 5-tuple lặp
lại ở ngày kia là đặc tính của mạng chứ không phải rò rỉ, nên audit nhóm chỉ áp cho ranh giới
train↔validation (trong cùng một ngày). Manifest ghi rõ điều này ở
`split.group_audit_scope = ["train", "validation"]`; audit theo `sample_id` vẫn phủ cả ba split.

### Thí nghiệm C: open-set

`labels.open_set_labels = ["Portmap"]` khiến các dòng Portmap:

- **không** vào `label_mapping.json` — mô hình không có output unit cho chúng;
- bị ép vào split `test` bất kể ngày nào;
- được mã hoá `_label = -1` (`OPEN_SET_LABEL_CODE`).

`make_report.py` nhận ra tập test toàn `-1` và chuyển sang nhánh open-set: **không** báo cáo
accuracy, precision, recall, ROC hay confusion matrix — tất cả đều vô định khi lớp đúng không
tồn tại trong không gian đầu ra. Thay vào đó nó xuất
`open_set_prediction_distribution.csv` + `open_set_distribution.png`, cùng
`mean_max_probability` và `mean_predictive_entropy`. Một mô hình dồn lưu lượng lạ vào một lớp
láng giềng với độ tin cậy cao là rủi ro vận hành rất khác với một mô hình rải đều — chỉ bảng
này phân biệt được hai trường hợp.

## Đường cong thứ ba: vì sao train và validation luôn chồng lên nhau

Trên 49.3 triệu dòng train, 1,900 cây × 31 lá là ~1.6 triệu dòng mỗi lá: mô hình không có chỗ
để ghi nhớ dòng riêng lẻ. Cộng với việc validation được rút ngẫu nhiên từ đúng cùng một tổng
thể, khoảng cách train–validation ở vòng 100 chỉ là 0.0076 logloss — nhỏ hơn một sai số chuẩn
của phép đo. **Hai đường trùng khít là bằng chứng về cách chia dữ liệu, không phải về khả năng
tổng quát hoá.**

`dataset.monitor_split` thêm một tập đánh giá thứ ba để biểu đồ nói được điều đó:

```json
"monitor_split": {
  "enabled": true, "split": "test", "name": "crossday",
  "maximum_rows": 2000000, "seed": 2026
}
```

Đây là mẫu phân tầng theo lớp (tối thiểu một dòng mỗi lớp, xác định theo seed) của ngày giữ
lại, được chấm điểm mỗi vòng cùng train và validation. Kết quả là một hình duy nhất có ba
đường: `train` và `validation` chồng lên nhau, `crossday` tách hẳn.

Nó **không** được dùng để chọn mô hình: `early_stopping = false` và số vòng cố định 100, nên
việc theo dõi không làm hỏng tính khách quan. `run_config.json → monitoring` ghi lại điều đó
(`used_for_model_selection: false`). Chi phí RAM của tập thứ ba đã được cộng vào
`model.estimate_training_memory` (bins + score + prediction buffer + lần vật chất hoá float32
duy nhất lúc dựng), nên `require_safe_memory_profile` vẫn chặn đúng.

## Chỉ số nào được coi là chính

TFTP một mình chiếm 28.5% corpus, nên accuracy gần như vô nghĩa: đoán bừa lớp đông nhất đã
được 28.5%. Vì vậy:

- `summary_metrics.csv` mở đầu bằng **Macro F1** rồi **Balanced Accuracy**; Accuracy bị đẩy
  xuống sau MCC và Weighted F1.
- `test_metrics.json` khai báo tường minh `primary_metrics = ["macro_f1",
  "balanced_accuracy"]` và `secondary_metrics = ["accuracy", "weighted_f1"]`.
- Balanced accuracy (= macro recall) được tính **mỗi vòng**, không chỉ ở bảng cuối, nên
  `learning_curves` có panel riêng cho nó.

## Biểu đồ và metric

13 nhóm hình, mỗi nhóm xuất đồng thời PNG 300 dpi + PDF vector + CSV dữ liệu:

- `learning_curves` — **bốn panel**: Multi-logloss, Macro-F1, Balanced Accuracy và Accuracy
  (theo thứ tự đó, accuracy đứng cuối vì nó ít nói nhất). Mỗi panel vẽ `Train`, `Validation`
  và — nếu `monitor_split` bật — `Crossday`, kèm vạch đứt `Resume` tại mỗi lần đổi session.
  CSV đi kèm còn có `val_minus_train_multi_logloss` và `monitor_minus_val_multi_logloss` để
  trích dẫn khoảng cách bằng số thay vì ước lượng bằng mắt. Vẽ lại và upload sau **mỗi**
  checkpoint block.
- `open_set_distribution` — chỉ có ở thí nghiệm C: mô hình gán lớp chưa từng thấy vào đâu.
- `confusion_matrix` (chuẩn hoá theo hàng) và `confusion_matrix_raw` (đếm thô, thang log).
- `roc_curves`, `pr_curves`, `per_class_f1`, `class_distribution`, `iteration_time`,
  `lr_schedule`.
- Bốn nhóm feature importance: gain, split, permutation, SHAP.

`summary_metrics.csv` mở đầu bằng **Macro F1** và **Balanced Accuracy**, rồi Macro
Precision–Recall, MCC, Weighted F1, Accuracy, F1 lớp thiểu số nhỏ nhất, Log Loss, AUC-ROC
(macro-OVR / weighted-OVR / micro), PR-AUC, `final_iteration=100`, `num_trees`,
`model_size_mb`, thời gian train, độ trễ p50/p95, thông lượng và peak RSS.

Ở thí nghiệm C, `summary_metrics.csv` có dạng khác hẳn (`evaluation_mode = open_set`,
`dominant_predicted_class`, `mean_max_probability`, `mean_predictive_entropy`) — đúng như nó
phải thế, vì không có nhãn đúng nào để đo.

## Nghiệm thu

1. `history.json` chứa đúng iteration 1..100, không thiếu hoặc trùng.
2. Có ít nhất hai `session_id`; `learning_curves` hiển thị vạch `Resume` đúng vị trí.
3. Mỗi bản ghi history có trường `environment` (`colab` / `github_actions` / `local`).
4. `make_report.py` tái tạo đủ 13 nhóm hình và toàn bộ CSV chỉ từ artifact S3, chạy lại cho
   kết quả giống hệt.
5. `final_model_round_100.txt` nạp độc lập, `current_iteration() == 100`.
6. `sample_manifest.json` → `leakage_audit` có `method =
   exact_intersection_on_deterministic_identity_subsample`, ba giao điểm bằng 0, và
   `sample_identities_tracked_distinct > 0` (chứng minh audit thực sự đã đọc dữ liệu).
7. Tên và thứ tự thuộc tính của Booster khớp tuyệt đối `preprocessing.json`.
8. `config/feature_selection.json` có `method = "none"` và
   `selected_feature_count == candidate_feature_count` — chứng minh **bước sàng lọc theo gain**
   không loại cột nào (các cột bị tiền xử lý loại đã nằm trong `preprocessing.json →
   dropped_columns` kèm lý do, và `Unnamed: 0` phải có mặt ở đó).
   (Nếu bật lại sàng lọc: `fit_split = "train"` và `ranking_by_gain` phủ hết cột ứng viên,
   chứng minh việc chọn cột không chạm vào validation/test.)
9. Repo public không chứa `AKIA` hay secret trong `.ipynb`.
10. `preprocessing.json → dropped_columns` chứa cả bốn cột `Unnamed: 0`, `__source_row_id`,
    `__source_file_id`, `__capture_day` với lý do `explicitly excluded by configuration`, và
    `feature_importance_gain` **không** còn cột nào trong số đó.
11. `label_mapping.json` có đúng 14 khoá ở thí nghiệm A, không khoá nào bắt đầu bằng `DrDoS_`
    và không có `UDP-lag`.
12. Ở thí nghiệm B: `sample_manifest.json → split.strategy = "by_capture_day"`,
    `split.group_audit_scope = ["train", "validation"]`, và `history.json` có
    `monitor_macro_f1` ở đủ 100 vòng. Trong `learning_curves.csv`,
    `monitor_minus_val_multi_logloss` phải lớn hơn `val_minus_train_multi_logloss` — nếu
    không, split theo ngày đã thôi là split theo ngày.
13. Ở thí nghiệm C: `test_metrics.json → evaluation_mode = "open_set"`, không có khoá
    `accuracy`, và `open_set_prediction_distribution.csv` phủ đủ 13 lớp đã huấn luyện.
14. `history.json` có `train_macro_recall` và `val_macro_recall` ở mọi vòng.
15. Nếu dùng `validation_permutation_top_k`: `feature_selection.json` có
    `fit_split = "validation"`, `ranking_run_id` trỏ đúng run đã sinh bảng xếp hạng, và
    `selected_feature_count == maximum_features`. Bảng xếp hạng phải đến từ
    `feature_ranking.py` — không có đường nào nạp được một bảng đo trên test.

## Kiểm tra quyền S3 trước khi train

Sai policy IAM chỉ lộ ra khi pipeline ghi checkpoint đầu tiên — tức sau khi đã tiền xử lý xong
và tốn hàng giờ. Đoạn dưới kiểm đúng bốn quyền pipeline cần, trên đúng prefix, trong vài giây.
Nó **tự nạp secret** từ Colab Secrets nên dán vào cell mới nào cũng chạy được, không cần chạy
cell nào trước. Thiếu secret thì nó nói thẳng tên secret đó thay vì báo `KeyError`:

```python
import os, boto3
from botocore.exceptions import ClientError

try:
    from google.colab import userdata          # tự nạp secret, không phụ thuộc cell 2
except ImportError:
    userdata = None

NEEDED = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
          "S3_BUCKET", "S3_PREFIX"]
missing = []
for name in NEEDED:
    value = os.environ.get(name, "").strip()
    if not value and userdata is not None:
        try:
            value = (userdata.get(name) or "").strip()
        except Exception:
            value = ""
    if value:
        os.environ[name] = value
    else:
        missing.append(name)
if missing:
    raise SystemExit(
        "Thiếu secret (hoặc chưa bật Notebook access): " + ", ".join(missing))

s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
bucket, prefix = os.environ["S3_BUCKET"], os.environ["S3_PREFIX"].rstrip("/")
key = f"{prefix}/.permission_probe"

def check(name, fn):
    try:
        fn(); print(f"  OK      {name}")
    except ClientError as exc:
        print(f"  DENIED  {name}: {exc.response['Error']['Code']}")

print(f"s3://{bucket}/{prefix}  (region {os.environ['AWS_DEFAULT_REGION']})")
check("ListBucket", lambda: s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1))
check("PutObject",  lambda: s3.put_object(Bucket=bucket, Key=key, Body=b"probe"))
check("GetObject",  lambda: s3.get_object(Bucket=bucket, Key=key))
check("DeleteObject", lambda: s3.delete_object(Bucket=bucket, Key=key))
```

Bốn dòng `OK` nghĩa là đường S3 thông. Bất kỳ `DENIED` nào cũng phải sửa policy trước khi
chạy tiếp — `AccessDenied` ở `PutObject` là dấu hiệu kinh điển của policy viết cho prefix của
một project cũ.

### Thiếu `DeleteObject` thì sao?

Pipeline **vẫn chạy đúng**. Cả hai chỗ gọi `delete_object` (`checkpoint.py`, `make_report.py`)
đều nằm trong `finally` và được bọc `try/except` chỉ ghi cảnh báo — chúng dọn key tạm sau khi
đã `copy_object` sang key cuối và xác minh checksum xong. Mất quyền xoá không làm hỏng artifact
nào.

Cái giá là rác tích luỹ. Key tạm có dạng `<key>.tmp-<uuid4>`, **uuid mới cho mỗi lần upload**,
nên chúng không đè lên nhau: mỗi lần ghi để lại vĩnh viễn một bản sao mồ côi. Một run đầy đủ
của v2 nặng khoảng 1,9 GiB / 306 object, nên thiếu quyền xoá sẽ đẩy nó lên xấp xỉ gấp đôi.

Điều đó không chỉ tốn tiền lưu trữ. `make_report.py --run-dir s3://...` tải **mọi** object dưới
run prefix về trước khi dựng lại báo cáo, kể cả orphan — nên fallback `mode=report` trên runner
GitHub (đĩa ~14 GB) phải tải gấp đôi dữ liệu cần thiết.

Vì vậy: không bắt buộc, nhưng nên cấp. Chỉ cần thêm một statement, không đụng policy sẵn có:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowDeleteInsideV3ColabPrefix",
      "Effect": "Allow",
      "Action": "s3:DeleteObject",
      "Resource": "arn:aws:s3:::my-thesis-checkpoints/Luan-Van-Lightgbm-Parquet-Github-v3-colab/*"
    }
  ]
}
```

Chạy được ở Colab hoặc GitHub Actions. Trên máy Windows có antivirus quét HTTPS (AVG, Avast,
Kaspersky…) thì boto3 có thể chết ở khâu bắt tay TLS với `CERTIFICATE_VERIFY_FAILED` — đó là
lỗi của antivirus, không phải của credentials.

## Diễn giải feature importance

Gain và split là importance phụ thuộc cấu trúc model. SHAP biểu diễn đóng góp vào dự đoán,
không chứng minh quan hệ nhân quả. Các thuộc tính tương quan có thể chia sẻ importance; không
gộp bốn thước đo thành một điểm duy nhất và không dùng chúng để sửa baseline này.
