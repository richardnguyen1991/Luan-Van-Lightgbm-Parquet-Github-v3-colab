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
- `feature_selection = "train_gain_top_k"`: giữ 20 thuộc tính có gain cao nhất, chọn **chỉ từ
  train split** bằng một model sàng lọc riêng biệt (xem *Chọn thuộc tính theo gain*). Đây là
  ràng buộc RAM, không phải tuning: không hyperparameter nào của model chính bị đụng tới.
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
`summary_metrics.csv` ngay trong notebook.

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

## Chọn thuộc tính theo gain: cắt cột nào, và vì sao điều đó không phá baseline

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

## Hai profile huấn luyện

`config/train.json` (Colab Pro) và `config/train.gha.json` (runner GitHub) có **mọi
hyperparameter học giống hệt nhau**, kể cả khối `feature_screening` quyết định 20 cột nào vào
model — chỉ khác `num_threads` và ngân sách thời gian/RAM.
LightGBM ghi rõ `deterministic=true` cho kết quả ổn định *kể cả khi `num_threads` khác nhau*,
nên một run có thể chuyển qua lại giữa hai môi trường. Test
`test_runner_profile_differs_only_in_resource_parameters` khoá tính chất này.

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

## Biểu đồ và metric

13 nhóm hình, mỗi nhóm xuất đồng thời PNG 300 dpi + PDF vector + CSV dữ liệu:

- `learning_curves` — **biểu đồ loss, biểu đồ accuracy** và Macro-F1 (train vs validation),
  kèm vạch đứt `Resume` tại mỗi lần đổi session. Vẽ lại và upload sau **mỗi** checkpoint block.
- `confusion_matrix` (chuẩn hoá theo hàng) và `confusion_matrix_raw` (đếm thô, thang log).
- `roc_curves`, `pr_curves`, `per_class_f1`, `class_distribution`, `iteration_time`,
  `lr_schedule`.
- Bốn nhóm feature importance: gain, split, permutation, SHAP.

`summary_metrics.csv` gồm Accuracy, Balanced Accuracy, Macro/Weighted Precision–Recall–F1,
MCC, F1 lớp thiểu số nhỏ nhất, Log Loss, AUC-ROC (macro-OVR / weighted-OVR / micro), PR-AUC,
`final_iteration=100`, `num_trees`, `model_size_mb`, thời gian train, độ trễ p50/p95, thông
lượng và peak RSS.

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
7. Tên và thứ tự thuộc tính của Booster khớp tuyệt đối danh sách
   `selected_features_in_model_order` trong `config/feature_selection.json`, và danh sách đó
   là tập con đúng thứ tự của `preprocessing.json`.
8. `feature_selection.json` có `fit_split = "train"` và `ranking_by_gain` phủ hết cột ứng
   viên — chứng minh việc chọn cột không chạm vào validation/test.
9. Repo public không chứa `AKIA` hay secret trong `.ipynb`.

## Kiểm tra quyền S3 trước khi train

Sai policy IAM chỉ lộ ra khi pipeline ghi checkpoint đầu tiên — tức sau khi đã tiền xử lý xong
và tốn hàng giờ. Đoạn dưới kiểm đúng bốn quyền pipeline cần, trên đúng prefix, trong vài giây.
Dán vào một cell mới trong Colab, **chạy sau cell 2** (cell nạp secret):

```python
import os, boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
bucket, prefix = os.environ["S3_BUCKET"], os.environ["S3_PREFIX"].rstrip("/")
key = f"{prefix}/.permission_probe"

def check(name, fn):
    try:
        fn(); print(f"  OK      {name}")
    except ClientError as exc:
        print(f"  DENIED  {name}: {exc.response['Error']['Code']}")

print(f"s3://{bucket}/{prefix}")
check("ListBucket", lambda: s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1))
check("PutObject",  lambda: s3.put_object(Bucket=bucket, Key=key, Body=b"probe"))
check("GetObject",  lambda: s3.get_object(Bucket=bucket, Key=key))
check("DeleteObject", lambda: s3.delete_object(Bucket=bucket, Key=key))
```

Bốn dòng `OK` nghĩa là đường S3 thông. Bất kỳ `DENIED` nào cũng phải sửa policy trước khi
chạy tiếp — `AccessDenied` ở `PutObject` là dấu hiệu kinh điển của policy viết cho prefix của
một project cũ.

Chạy được ở Colab hoặc GitHub Actions. Trên máy Windows có antivirus quét HTTPS (AVG, Avast,
Kaspersky…) thì boto3 có thể chết ở khâu bắt tay TLS với `CERTIFICATE_VERIFY_FAILED` — đó là
lỗi của antivirus, không phải của credentials.

## Diễn giải feature importance

Gain và split là importance phụ thuộc cấu trúc model. SHAP biểu diễn đóng góp vào dự đoán,
không chứng minh quan hệ nhân quả. Các thuộc tính tương quan có thể chia sẻ importance; không
gộp bốn thước đo thành một điểm duy nhất và không dùng chúng để sửa baseline này.
