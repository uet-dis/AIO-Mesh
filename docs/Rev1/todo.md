# AIO-Mesh FGCS Major Revision — TODO

## Mục tiêu của vòng revise

Bản revise phải tách rõ hai lớp đóng góp: **(i) runtime safety contract của AIO-Mesh** là nguồn của thuộc tính fail-closed trước actuation; **(ii) HARP** là một policy có thể thay thế, có nhiệm vụ cải thiện chất lượng/hiệu quả của candidate actions. Không tiếp tục diễn đạt PPO, GAT hoặc invalid-action masking như các thuật toán mới. Claim về safety chỉ giới hạn trong các hard constraints đã mô hình hóa (CPU, memory, SRv6 MSD) và trạng thái mà controller quan sát được.

## Trạng thái hiện tại và nguồn sự thật

### Đã có

- `submitted.tex` là **baseline số liệu hiện hành** của paper. Giữ nguyên các số này khi rà soát; chỉ thay sau khi tái chạy, đối chiếu seed/config và xác nhận artifacts.
- `thesis.tex` là **nguồn concept và lý thuyết** để mở rộng paper: mô hình substrate/SFC, hard và soft constraints, MDP/action, GAT/MaskablePPO, migration/MBB, kiến trúc runtime và HIL. Không lấy các số liệu hay claim thực nghiệm trong thesis làm kết quả mới nhất nếu chúng khác submitted.
- `AIO-Mesh.rev1.tex` đã có phần định vị lại novelty theo hướng runtime-first, HARP là policy instance, phân biệt policy mask với admission boundary, cùng nhiều đoạn bổ sung về MSD, safety scope, evaluation questions và limitations. Các phần này là bản nháp, chưa được xem là xác nhận của implementation.
- Benchmark config cũ đã từng ánh xạ `elephant_stress` sang `heavy_tail`; hiện `chaos` đã có nhánh generator riêng gồm burst peak và heavy-tail draw. Đây mới là bước tách profile trong simulation env, chưa phải rerun và chưa có kết quả revision.
- Notebook `notebook/harp-ablation-nsfnet.ipynb` hiện train các ablation variants rồi đánh giá 3 workload bằng 5 seeds; code benchmark cũ nằm lẫn trong notebook nhưng dùng evaluator/action-space khác. Không gộp kết quả hai evaluator vào cùng một bảng.
- Đã kiểm tra thư mục `kaggle/results_kaggle` trong workspace: có benchmark algorithm NSFNET cho 3 workload (`elephant_stress`, `burst_surge`, `chaos`) × 4 thuật toán × 5 seed (60 dòng); ablation có raw NSFNET 3 workload × 6 variant × 5 seed (90 dòng), cùng summary 5-seed cho cả NSFNET/GEANT2/Vietnam (18 hàng/topology). Các artifact này là kết quả lịch sử để đối chiếu, chưa được xác nhận tái lập theo source/protocol Rev1 và không thay thế rerun.
- Snapshot `CoreRouter` trong `kaggle/results_kaggle` còn chứa kết quả testbed 2026-05-17, checkpoint/model, dữ liệu và mã nguồn cũ. Không nhập chúng vào kết quả notebook mới; nếu cần dùng trên Kaggle, gắn folder như dataset riêng để notebook chỉ inventory/hash.
- Đã tạo `notebook/rev1-benchmark-ablation-kaggle.ipynb`: smoke trên ba topology trước, full run bật thủ công sau khi smoke pass; notebook train checkpoint mới từ source snapshot hiện hành, hỗ trợ 10 seed và ghi manifest/hash. Notebook chỉ inventory folder `results_kaggle` nếu folder đó được attach vào `/kaggle/input`; chưa có cell nào được chạy trên Kaggle.
- Code hiện có action simulation dạng cặp `(place, route)` trên không gian (N^2), observation gồm 6 node features và 13 phần tử request/service/global-context, cùng GAT hai lớp. Cần mô tả đúng pipeline cụ thể của từng experiment/model thay vì suy rộng từ thesis.

### Chưa có / chưa xác nhận

- Chưa có lượt rerun nào được hoàn tất cho revision này; chưa có paired protected-baseline study, significance analysis, latency/scalability, K-sensitivity hay long-running HIL mới. Các output đã có trong `kaggle/results_kaggle` vẫn là historical artifacts, chưa audit đủ provenance để thay số submitted.
- Các con số trong thesis khác submitted ở một số phần và không thay thế baseline submitted.
- Semantics chain nhiều VNF so với action `(place, route)`, cách tính full SID list/MSD, mask-empty behavior, telemetry freshness/TOCTOU, và chính xác hybrid gate cần được xác minh từ runtime và artifacts trước khi mô tả như tính năng đã triển khai.
- Draft còn các `REVISION TODO`, kết quả lặp Elephant/Chaos, câu chữ ở thì tương lai và các chỗ cần cập nhật sau khi có số mới.

### Quy tắc làm việc

1. Đối chiếu claim/công thức từ thesis với code; chỉ đưa vào Rev1 khi khớp implementation hoặc ghi rõ đó là mô hình khái quát/limitation.
2. Tái tạo kết quả submitted trước để xác nhận baseline; sau đó chạy các thí nghiệm mới reviewer yêu cầu. Mọi bảng/figure sửa phải truy được về config, seed và result artifact.
3. Không tự điền số ước lượng vào paper. Nếu experiment không thể chạy, nêu rõ giới hạn và thu hẹp claim tương ứng.

## Thứ tự công việc tiếp theo

1. **Audit và khóa protocol**: lập mapping từng bảng/figure submitted tới script, config, model checkpoint, seeds và artifacts; xác định phiên bản code/runtime đang đại diện cho paper.
2. **Chốt semantics hệ thống**: xác minh chain/action/anchor/SID/MSD, admission, mask rỗng, hybrid predicate, feature normalization và telemetry freshness. Sửa mô hình và công thức trong Rev1 cho đúng trước khi chạy experiment phụ thuộc chúng.
3. **Tái lập baseline**: chạy lại kết quả chính submitted theo đúng topology/scenario/baseline hiện có; so sánh với số đã công bố và ghi nguyên nhân mọi sai lệch.
4. **Sửa và chạy evaluation bắt buộc**: tách Elephant/Chaos thành workload thật sự khác nhau hoặc bỏ một profile; chạy cùng admission boundary cho các policy; dùng matched seeds và thống kê; cập nhật load sweep.
5. **Bổ sung operational evidence**: đo decision-latency breakdown, scalability theo topology/chain/concurrency và K-sensitivity; thực hiện HIL dài hạn/controlled over-MSD nếu testbed và thời gian cho phép.
6. **Hoàn thiện paper**: tích hợp concept/công thức đã xác minh, thay toàn bộ số liệu/prose/hình bằng kết quả đã audit, xóa TODO/placeholder, rà abbreviation và rút gọn conclusion.

Các task dưới đây là checklist chi tiết của các bước này. Trạng thái mặc định là **chưa làm trong vòng revision** trừ khi được ghi rõ ở mục “Đã có”.

### Nhật ký thực hiện P0

- **P0.1 — code prep xong, Kaggle run pending:** benchmark algorithm map `chaos` riêng; env `chaos` kết hợp burst peak và heavy-tail draw; regression tests được thêm. Cần chạy cả Elephant và Chaos trên ba topology bằng đúng benchmark evaluator và lưu artifacts. Profile Chaos hiện tại chưa đa dạng service/source-destination như testbed Chaos.
- **P0.2 — code prep xong, Kaggle run pending:** testbed baseline có thêm `greedy_protected` và `decoupled_protected`, đi qua `try_reserve` chung; record có candidate hard-infeasibility và reason. Cần chạy paired matched traces/seeds; đây là software baseline simulation, không phải live HIL.
- **P0.3 — snapshot `kaggle/results_kaggle` đã xác nhận mismatch giữa runtime và simulator/trainer:** `CoreRouter/src/orchestration/jo_vdpr/env.py` khai báo `Discrete(N*N)`; `train_v11.py` học theo env này. Nhưng `CoreRouter/src/ai/dgrl_agent.py` khôi phục checkpoint bằng `MultiDiscrete([10,10])` (và observation shape cố định 73), không phụ thuộc topology. Benchmark loader thì dựng MaskablePPO mới từ env `Discrete(N*N)` rồi nạp `policy.pth`; nếu checkpoint có output layer MultiDiscrete thì kích thước policy sẽ không khớp và loader phải lỗi, không thể coi đây là adapter tương thích. Các file `dgrl_v11_final_*.zip` trong snapshot có metadata `MultiDiscrete`; benchmark config lại ưu tiên checkpoint `results/models/ablation/harp_full` nếu có. Vì không có manifest ghi checkpoint path/hash của run 20260617, chưa thể khẳng định checkpoint cụ thể nào sinh ra CSV benchmark. Notebook Rev1 train checkpoint mới từ cùng source/environment để smoke/rerun pipeline, nhưng không chứng minh runtime agent hay checkpoint v11 cũ tương thích. Cần thống nhất action distribution/mask semantics và test từng topology trước khi dùng kết quả làm bằng chứng cho runtime.
- **P0.4 — audit:** runtime `SFCRequest`/router hiện chọn một placement và một routing anchor, tạo một SID list từ cặp đó; chưa thấy runtime chain-level mapping nhiều VNF như mô hình tổng quát trong thesis/draft. Cần giới hạn claim hoặc xác định phần controller khác thực thi chain trước khi hoàn tất semantics.
- **P0.7 — audit:** hybrid gate hiện dùng `global_utilization=max(mean CPU utilization, mean MSD utilization)`, engage DRL khi >45% hoặc alert bật, release về heuristic khi <35% và không alert. Runtime không thấy averaging window trong `choose_mode`; alert có thể đến từ request/forecast.
- **P0.8 — audit:** node features trong `NetworkStateManager.to_observation` khớp 6 trường đã ghi; normalizer được load từ VecNormalize/NPZ. Admission `try_reserve` dùng lock để kiểm tra và cập nhật local reservation state, nhưng hiện chưa thấy state version/freshness TTL hoặc atomic transaction với Kubernetes/P4.
- **P0.6 — chưa bắt đầu, implementation gap:** chưa tìm thấy K-shortest-path cache trong runtime/source hiện tại. Không thể chạy K-sensitivity bằng cách đổi một biến notebook; trước hết cần implement cache và dùng chung ở mask/admission/controller, hoặc thu hẹp paper claim và giải thích reviewer concern bằng giới hạn hiện tại.

## Kế hoạch chạy lại trên Kaggle (không chạy benchmark local)

### 0. Đóng băng input và smoke check

1. Tạo một Kaggle Dataset/version duy nhất chứa source snapshot và `data/real_telecom_combined.csv`, kèm manifest ghi Git commit/source SHA-256. Notebook mới tự train checkpoint smoke/full; không cần đóng gói checkpoint v11 cũ. Nếu cần đối chiếu lịch sử, attach `kaggle/results_kaggle` thành dataset riêng để inventory/hash, không trộn `results/` vào source snapshot.
2. Dùng một notebook worker cho mỗi topology: `nsfnet`, `vietnam`, `geant2`. Cùng Python/package versions, cùng topology data, cùng workload configs và seeds.
3. Chạy smoke stage trong notebook mới trên ba topology: notebook train các checkpoint ngắn từ source hiện hành, rồi chạy benchmark và strict ablation với 64 request steps. Xác nhận action space/model match, mask output, output CSV có topology/seed/scenario/variant. Smoke check chỉ xác nhận pipeline; không đưa số vào paper.
4. Notebook cũ `notebook/harp-ablation-nsfnet.ipynb` gọi module path cũ `src.analytics.ablation_study` và không dùng cùng evaluator/action interface. Dùng notebook mới `notebook/rev1-benchmark-ablation-kaggle.ipynb`; notebook này train checkpoint smoke/full từ source snapshot hiện hành và không nạp checkpoint v11 cũ. Chỉ bật full stage sau khi smoke pass; historical outputs trong `kaggle/results_kaggle` chỉ dùng đối chiếu sau khi xác nhận provenance.

### 1. Khôi phục benchmark chính (P0.1, P0.3)

- Với checkpoint v11 historical/submitted, vẫn phải sửa/kiểm tra evaluator: ánh xạ action pair `(place, route)` từ `MultiDiscrete([N,N])` sang biểu diễn mà benchmark environment nhận, hoặc cung cấp adapter environment có cùng action space. Notebook mới né phụ thuộc đó bằng cách train checkpoint mới từ cùng source/environment hiện hành; kết quả mới vì vậy cần được mô tả là rerun theo pipeline hiện tại, không khẳng định là tái lập checkpoint v11 cũ.
- Historical results trong `kaggle/results_kaggle` có benchmark NSFNET 3 workload × 5 seed và ablation 3 topology × 3 workload × 5 seed. Trước khi dùng làm đối chứng, truy source/checkpoint/config hash và xác định vì sao benchmark hiện chỉ có NSFNET; các aggregate CSV dạng `mean ± std` không đủ thay raw per-seed results.
- Chạy lại cấu hình submitted để làm đối chứng: scenario gốc, 3 topology, 5 seed cũ `[42,100,2024,8888,9999]`, 3000 steps/seed. So sánh từng seed với `docs/Submit_1/results`; không thay số paper nếu chưa giải thích sai lệch.
- Sau khi baseline tái lập được, chạy P0.1 workload tách biệt: `elephant_stress` và `chaos` (cùng seed set, topology, steps, load, model/checkpoint). Lưu config thực tế sau resolve, input hash, checkpoint hash, source hash, package/hardware info và per-step/per-seed CSV.
- Chạy thống kê paired HARP vs SAF-H trên tối thiểu 10 matched seeds (nên dùng 20 nếu thời gian GPU cho phép). Dùng cùng seed làm cặp trong cùng workload/topology; báo bootstrap 95% CI, Wilcoxon signed-rank, paired rank-biserial, và Holm correction cho family comparisons đã định trước. Giữ riêng 5-seed historical reproduction với 10/20-seed revision run.

### 2. Ablation HARP (P0.3, P1.6)

- Dùng notebook ablation riêng (`notebook/harp-ablation-nsfnet.ipynb`) làm template; mỗi worker chỉ đổi `WORKER_TOPOLOGY`. Train variant checkpoints một lần cho topology với seed/config cố định (hoặc nêu rõ nếu dùng checkpoint đã đóng băng), sau đó evaluate cùng các variants trên workload traces/seeds matched. Không train lại model cho từng evaluation seed.
- Giữ 3000 evaluation steps và ba workload hiện có (`uniform`, `bursty`, `heavy_tail`) để có thể đối chiếu lịch sử. Dùng ít nhất 10 evaluation seeds cho revision; ghi riêng training seed và evaluation seed.
- Giữ benchmark algorithm rerun tách khỏi ablation. Ablation trả lời đóng góp của GAT/mask/penalty; benchmark trả lời HARP so với SAF-H/Greedy/DAI. Không gọi hai script trên cùng checkpoint nếu action space không tương thích.
- Export raw per-seed CSV và summary; thêm candidate count bị mask loại, runtime reject count/reason, admitted hard violations. Không diễn giải ablation “without mask” là bỏ admission trừ variant được cấu hình cụ thể là unchecked admission.

### 3. P0 protected-baseline study

- Trên Kaggle CPU, chạy testbed paired protected study từ cùng workload generator và seeds: `jo_vppm`, `heuristic`, `greedy`, `greedy_protected`, `decoupled`, `decoupled_protected`. So sánh unprotected/protected bằng cùng request trace; protected policies phải gọi cùng `NetworkStateManager.try_reserve`.
- Xuất candidate hard-infeasible rate, acceptance/safe acceptance, rejection theo CPU/MEM/MSD, admitted violations, decision latency median/P95/P99. Ghi rõ đây là offline software simulation; không dùng làm bằng chứng HIL.
- Lưu per-request records, per-seed summaries, metadata JSON và notebook output; không chỉ tải dashboard/aggregate.

### 4. Chặn trước / việc cần giải quyết ngoài Kaggle

- **P0.4:** code runtime hiện chưa chứng minh multi-VNF chain mapping; việc chạy Kaggle không giải quyết được. Thu hẹp paper thành semantics đơn request `(place, route)` hoặc triển khai chain-level runtime + tests trước khi mô tả chain tổng quát.
- **P0.5:** đo latency decision components và scalability là benchmark được, nhưng phải thêm instrumentation trước. Kaggle cần CPU-only profile tách state/mask/inference/admission; nếu đo GPU thì báo riêng và không so trực tiếp.
- **P0.6:** K-sensitivity chỉ sau khi K-path cache được implement. Hiện không có K knob hợp lệ để sweep.
- **P0.7/P0.8:** paper sửa từ code audit trước; Kaggle không tạo ra telemetry freshness/TTL, versioning, hay atomic cloud+P4 reservation nếu runtime chưa hỗ trợ.
- Long-running HIL và controlled BMv2/P4 over-MSD cần testbed tương ứng, không thay thế bằng Kaggle simulation.

### 5. Artifact checklist cho mỗi Kaggle run

- `run_manifest.json`: topology, scenario, workload parameters, train/eval seeds, steps, model/normalizer/source/data SHA-256, action/observation spaces, git commit, Python/library versions, Kaggle accelerator/CPU/RAM.
- `per_request.csv`, `per_seed_summary.csv`, `aggregate_summary.csv`, `paired_statistics.csv` (nếu so sánh), config-resolved JSON, logs và notebook `.ipynb` output.
- Tải archive về và kiểm tra seed coverage/row count/NaN/failed runs trước khi dùng. Giữ artifacts raw bất biến; tạo paper tables/figures bằng script riêng từ các artifacts đã xác nhận.

## P0 — Bắt buộc hoàn thành trước khi resubmit

| ID | Reviewer concern | Công việc phải làm | Đầu ra cần cập nhật trong paper |
|---|---|---|---|
| P0.1 | R2: Elephant và Chaos có số liệu giống hệt nhau | **Đang làm.** Alias đã được xác nhận và Chaos đã được tách trong code thành burst + heavy-tail. Chạy tests, audit seeds/artifacts, rồi rerun Elephant/Chaos trên Vietnam, NSFNET, GEANT2; lưu config/hash cho từng run. | Thay các dòng liên quan trong Table `simulation_acceptance_summary`, cập nhật mọi prose/figure phụ thuộc số liệu. |
| P0.2 | R2: safety chưa được tách khỏi policy | Chạy **cùng admission boundary** cho Greedy, DAI, SAF-H và HARP. Khuyến nghị thiết kế paired: policy × {unprotected, protected} trên cùng workload trace/seed. | Bảng mới: candidate hard-infeasible rate, rejection CPU/MEM/MSD, safe acceptance, admitted violations, decision latency. Nếu mọi protected policy đều 0 violation, ghi rõ safety thuộc runtime contract. |
| P0.3 | R2: HARP vs SAF-H có margin nhỏ, chưa significance | Tăng số matched seeds: tối thiểu 10, ưu tiên 20 nếu chi phí cho phép. Với HARP vs SAF-H dùng paired Wilcoxon signed-rank trên cùng seed; báo 95% bootstrap CI, paired rank-biserial effect size; Holm correction khi thực hiện nhiều comparisons. Không chỉ báo mean±std. | Bổ sung thống kê vào bảng chính hoặc bảng phụ; prose phải dùng “statistically significant” chỉ khi test hỗ trợ. |
| P0.4 | R2: compact action/anchor chưa xác định đầy đủ chain | Đối chiếu code để xác định chính xác `v_place` áp dụng cho function nào/chain-level template nào; mô tả thuật toán dựng `phi_q`, các segment paths và full SID list từ `(v_place,v_anchor,m)`. Tạo test với chain ≥3 VNFs và đối chiếu `h_q` giữa mask/admission/controller. | Hoàn thiện Sec. System Model, HARP MDP, Algorithm 2; không để TODO semantics. |
| P0.5 | R2/R4/R5: không có end-to-end decision latency/scalability | Đo riêng: state construction, feasible-set/mask construction, HARP inference, heuristic inference, final admission validation, controller handoff, total decision latency. Báo median/P95/P99 và throughput decisions/s. Vary topology size + active chains + concurrent requests. | Subsection/figure/table scalability; cập nhật Complexity và Limitations. |
| P0.6 | R2: path cache có thể tạo false infeasibility | Chạy sensitivity với `K={1,3,5,10}` (thêm 20 nếu chi phí thấp). Cùng workload/seeds. Đo safe acceptance, no-feasible-cached-realization rate, mask time, cache memory, average feasible alternatives/action. Không gọi cache miss là “global infeasibility”. | Figure/table K-sensitivity; sửa terminology toàn paper. |
| P0.7 | R2: Hybrid predicate chưa định nghĩa | Trích đúng logic `IsStable(T_t)` từ code: biến telemetry, threshold, window, hysteresis, alert trigger, migration trigger. Nếu logic hiện tại ad-hoc, chuẩn hóa trước khi rerun Hybrid. | Sec. Architecture + Algorithm 1 + Experimental Setup. |
| P0.8 | R5: feature normalization, mask frequency, stale state, TOCTOU | Liệt kê đúng 6 node features + 13 global features, unit/source, normalization/clipping; ghi mask refresh trigger/frequency; telemetry interval; state timestamp/version; freshness TTL. Kiểm tra có resource reservation/version check hay chỉ revalidation. Nếu chưa có atomic reservation, không được claim là đã có. | Table feature definitions; Sec. HARP encoder/masking; Sec. Telemetry Freshness; Limitations. |

## P1 — Nên hoàn thành để vòng major revision thuyết phục

| ID | Reviewer concern | Công việc đề xuất | Đầu ra |
|---|---|---|---|
| P1.1 | R2: load sweep chỉ có một curve | Regenerate load sweep 0.2x–1.5x cho HARP, SAF-H, Greedy+Admission, DAI+Admission trên matched traces. | Thay `VN_load_sweep.pdf`; nên thể hiện safe acceptance + rejection/infeasible pressure; admitted violation có thể ở panel riêng nếu luôn 0. |
| P1.2 | R4/R5: topology nhỏ | Thêm topology synthetic/extended khoảng 50, 100, 200 nodes nếu runtime cho phép; nếu 200 quá nặng, báo giới hạn thực nghiệm và dùng 25/50/100. Giữ normalized load tương đương. | Scalability figure/table. |
| P1.3 | R5: services/concurrent chains | Vary active chains và concurrent requests, ví dụ các mức thấp/trung bình/cao phù hợp implementation; ghi rõ queueing policy. | Scalability table + throughput/latency curves. |
| P1.4 | R5: long-running stability | Chạy HIL liên tục tối thiểu vài giờ; ưu tiên 24 h nếu khả thi. Ghi total requests, admitted/rejected, rejection by constraint, migrations, controller/deployment failures, telemetry/mask refresh failures, latency drift, admitted MSD violations. | Subsection “Extended HIL stability”. Không gọi là production case study nếu không chạy trên carrier production. |
| P1.5 | R5: practical MSD consequence | Tạo controlled over-MSD test: cố ý tạo SID list > MSD, ghi controller/admission outcome và BMv2/P4 counter/log. Phân biệt rõ target-specific behavior; không suy rộng BMv2 thành mọi ASIC. | 1 đoạn motivation + 1 row/table/log summary; có thể dùng appendix nếu dài. |
| P1.6 | R2/R5: masking role | Từ ablation hiện có, báo thêm number/rate of candidates eliminated by mask và runtime rejections with/without mask. | Cập nhật ablation table/prose để chứng minh masking = efficiency/quality, admission = safety. |
| P1.7 | R4: abbreviations | Audit toàn bộ source: expand first occurrence only cho AIOps, MLOps, SFC, VNF, SRv6, MSD, PPO, GAT... | Clean source; không lặp long form sau first definition. |

## Hình cần sửa/bổ sung

1. **`figs/overview.pdf`**: làm rõ ba khối theo thứ tự `Policy/Heuristic -> independent runtime admission -> actuation`. Gắn nhãn HARP là “replaceable policy instance”; không vẽ masking như nguồn duy nhất của safety. Thêm state version/freshness vào đường telemetry nếu implementation có.
2. **`figs/VN_load_sweep.pdf`**: thay single-method curve bằng multiple protected baselines; ưu tiên 2 panels: (a) safe acceptance; (b) runtime rejection/hard-infeasible pressure. Nếu admitted violation của tất cả protected methods đều 0 thì có thể biểu diễn bằng annotation thay vì curve chồng nhau.
3. **`figs/testbed.pdf`**: thể hiện rõ protected vs bypass path; mọi protected policy phải đi qua cùng Admission Boundary. Nếu thêm protected Greedy/DAI, figure không cần đổi logic nhưng caption phải nói common boundary.
4. **Hình mới — K sensitivity**: x=`K`; y1=safe acceptance, y2=mask latency/cache memory hoặc dùng bảng nếu figure quá dày.
5. **Hình mới — Scalability/latency**: x=number of substrate nodes; đường riêng state/mask/inference/validation/total hoặc dùng stacked bars cho latency decomposition. Không gộp actuation time vào decision latency; báo actuation riêng.
6. **Hình mới tùy chọn — Long-run HIL**: time series decision latency + cumulative admitted violations/rejections; nếu không có biến động đáng kể, dùng summary table thay figure.

## Protocol thực nghiệm khuyến nghị

- Dùng **matched workload traces và matched random seeds** giữa HARP và SAF-H để paired tests có ý nghĩa.
- Lưu `seed`, topology version, workload config, `K`, model checkpoint hash, admission config, MSD profile, telemetry period, hardware/software version cùng mỗi run.
- Tách **decision latency** khỏi **actuation latency**. Decision latency kết thúc ở controller handoff; Kubernetes readiness/Tekton/SRv6 install được báo riêng.
- Với scalability, báo cả `|V|`, `|E|`, `|M|`, `K`, số active chains, concurrency, CPU/GPU của máy chạy policy, RAM, Python/PyTorch version.
- Với significance: primary comparison nên là HARP vs SAF-H; tránh test quá nhiều cặp nếu không cần. Báo exact `p`, CI và effect size, không chỉ dấu `*`.
- Không dùng cụm “globally infeasible” khi chỉ tìm trên `K` cached paths. Dùng “no feasible cached realization”.
- Không claim “hardware-safe by construction” theo nghĩa tuyệt đối. Dùng “fail-closed pre-actuation safety for the modeled CPU/memory/MSD constraints under controller-visible state”.

## Các điểm text cần hoàn thiện sau khi có số liệu

- Abstract: thay câu mô tả kết quả bằng các số chính sau rerun; giữ HARP là policy instantiation, runtime boundary là safety source.
- Introduction/Contributions: không đưa PPO/GAT/masking vào danh sách methodological novelty.
- Table related work: kiểm tra từng cột “Independent admission” và “Explicit PDP/MSD” trước submission; tránh đánh dấu tuyệt đối nếu paper liên quan chỉ hỗ trợ một phần.
- Experimental Setup: đổi “5 runs” sang số runs thực tế và ghi statistical protocol.
- Table `simulation_acceptance_summary`: thay toàn bộ dòng bị ảnh hưởng bởi audit Elephant/Chaos.
- HIL table: thêm protected Greedy/DAI hoặc tạo bảng mới nếu layout hiện tại quá chật.
- Operational Interpretation: chèn measured latency/scalability; bỏ từ “practical” nếu chưa có số đo hỗ trợ.
- Conclusion: sau khi hoàn tất experiments, thay câu mang tính kế hoạch bằng 1–2 quantitative findings mạnh nhất; giữ conclusion khoảng 2 đoạn.
