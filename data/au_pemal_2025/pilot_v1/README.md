# AU-PEMal-2025 dynamic-behavior pilot v1

這是一份以 AU-PEMal-2025 V2 為來源的小型可行性資料集。資料單位分成
sample、sandbox trace 與 normalized event 三層；標籤、來源特徵與 event
embedding 分開保存，模型訓練不需要重新連線或重做文字前處理。

## 收集方式

1. 從作者 GitHub commit
   `22f3ca6f58e8cd777a7781b06d6bd4d5bc338d59` 下載
   `AU-PEMal-2025-V2.csv`。檔案 SHA-256 為
   `8e3f3df56877178271878726b54264fbe342669672eb286535314b3a1a8958c6`。
2. 排除 24 個同一 SHA-1 具有衝突標籤的程式，再以 seed `20261002` 建立
   100 筆候選清單，並依 SHA-1 固定切分 train/validation/test。
3. 僅使用 VirusTotal v3 的既有資料：先以 SHA-1 查詢 file object，再以
   canonical SHA-256 查詢最多 40 份 behavior reports。沒有上傳、下載、重新
   掃描或在本機執行任何 PE 檔案。
4. 將 API、command、process、file、registry、service、module 與 network
   行為正規化成 label-free canonical text。每個 sample 保存全部取得的 reports，
   但只對品質最佳的 canonical report 產生正式 events。
5. 使用 `sentence-transformers/all-MiniLM-L6-v2` revision
   `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，以
   `normalize_embeddings=True` 對每個 event 產生 384 維 float embedding。

可重現指令：

```powershell
uv sync --dev --extra embedding --extra eda
uv run pe-research data fetch
uv run pe-research data select-pilot
uv run pe-research data enrich-vt
uv run pe-research data materialize
uv run pe-research data embed
uv run pe-research data validate --require-minimum
uv run pe-research data eda
```

`enrich-vt` 需要把已輪替的 VirusTotal key 放在執行環境的 `VT_API_KEY`，不得
寫入專案、設定檔或命令列。

## 惡意標籤如何取得

本 pilot 的良惡、malware category 與 family 都直接來自作者發布的
`AU-PEMal-2025-V2.csv`，不是由本專案重新掃描、VirusTotal verdict 或模型預測
產生。來源欄位與輸出對應如下：

| AU-PEMal 來源欄位 | 本資料集欄位 | 說明 |
|---|---|---|
| `Class` | `sample_labels.y` | `Benign` 映射為 `0`，`Malware` 映射為 `1` |
| `Class` | `sample_labels.y_known` | 來源有標籤時為 `1` |
| `Category` | `sample_labels.category` | RAT、Ransomware、Stealer、Trojan；良性為 Benign |
| `Family` | `sample_labels.family` | 例如 Remcos、Formbook、Qbot、LockBit；良性為 Benign |

`label_source` 記為 `AU-PEMal-2025-V2`，`label_confidence` 記為
`published_dataset`，表示這是繼承自已發布資料集的標籤，而不是本研究人工複核後
宣稱的絕對真值。為降低 label noise，同一 SHA-1 若出現不同 Class、Category 或
Family 就整筆排除；本版共排除 24 個衝突 SHA-1，詳情在
`artifacts/source_validation.json`。

Pilot 的 20 個 malicious samples 依固定候選順序收集，四個 Category 各 5 筆，
並盡量分散 Family。目前包含 18 個惡意 Family；Agenttesla 與 Formbook 各 2 筆，
其餘 Family 各 1 筆。這些 family labels 只用於標籤與視覺化，不會寫入 event
canonical text 或 embedding。

## Events 如何取得與正規化

Events 來自 VirusTotal 已存在的 sandbox behavior reports，不是 AU-PEMal CSV
本身提供的逐事件 sequence。每個候選程式採用以下唯讀查詢流程：

1. `GET /api/v3/files/{sha1}`：以 AU-PEMal 的 SHA-1 找到 VT file object，取得
   canonical SHA-256。
2. `GET /api/v3/files/{sha256}/behaviours?limit=40`：取得最多 40 份既有 sandbox
   behavior objects。
3. 完整 JSON 保存在 `raw/vt/files/` 與 `raw/vt/behaviours/`。本版 40 個成功
   samples 共取得 231 份 reports；沒有下載或執行 PE。
4. 同一 sample 的 reports 依「有效 behavior 欄位覆蓋率、可正規化事件數、分析
   日期、sandbox 名稱」排序，選出一份 canonical report。全部 reports 都保留在
   `traces.csv`，但 `events.csv` 與正式 embeddings 只使用 canonical report。

從 behavior object 的下列欄位建立事件：

| VT behavior 欄位 | `event_type` | 典型 `operation` |
|---|---|---|
| `calls_highlighted` | `api` | `call` |
| `command_executions` | `command` | `execute` |
| `processes_created/terminated/killed/injected` | `process` | `create`、`terminate`、`kill`、`inject` |
| `files_opened/written/deleted/files_attribute_changed` | `file` | `open`、`write`、`delete`、`change_attribute` |
| `registry_keys_opened/set/deleted` | `registry` | `open`、`set`、`delete` |
| `services_opened/created/started/stopped/deleted` | `service` | `open`、`create`、`start`、`stop`、`delete` |
| `modules_loaded` | `module` | `load` |
| `dns_lookups`、`http_conversations`、`ip_traffic` | `network` | `dns_lookup`、`http`、`connect` |

正規化時會遮罩 Windows/Linux 使用者名稱、password/token/API key、PID、UUID、
hash 與 IP 角色，並移除 URL query。每個事件形成類似下列的文字：

```text
event_type=file operation=write object_role=file object=C:\Users\<USER>\AppData\...\payload.exe
```

`canonical_text` 不包含 SHA、AU-PEMal Class/Category/Family、VT verdict 或資料來源。
MiniLM 只看正規化後的行為文字，因此 label 與 embedding 輸入保持分離。

VT behavior summary 大多沒有逐事件 timestamp。本版 40 個 canonical traces 的
`ordering_known` 與 `sequence_eligible` 全部為 `0`；`event_index` 只是穩定輸出索引，
不是程式的真實執行順序。`calls_highlighted` 也是高階 API 摘要，不能解讀成完整的
kernel system-call trace。

## ATT&CK concept labels 如何取得

`concept_labels.csv` 直接解析 canonical VT behavior object 內的
`mitre_attack_techniques`，沒有額外呼叫 ATT&CK relationship API，也沒有根據 event
文字自行推論技術。報告明確提供的 technique 才寫成 `value=1`、`known_mask=1`，並
記錄 `evidence_source=virustotal_behavior`。某 technique 沒有出現在 report 時視為
unknown，而不是負例；tactic 未由 report 提供時同樣以 `tactic_known=0` 保存。

## 收集結果

| 項目 | 數量 |
|---|---:|
| 原始 AU-PEMal 列數 | 21,703 |
| 唯一 SHA-1 | 13,904 |
| 重複 SHA-1 | 2,465 |
| 排除的衝突 SHA-1 | 24 |
| Pilot candidates | 100 |
| VT 已處理 candidates | 43 |
| VT file 404 | 3 |
| 有效 dynamic samples | 40 |
| Benign / malicious | 20 / 20 |
| RAT / Ransomware / Stealer / Trojan | 各 5 |
| 全部 sandbox reports | 231 |
| Canonical traces | 40 |
| Events / embeddings | 11,161 / 11,161 |
| Unique canonical event texts | 8,234 |
| Train / validation / test dynamic samples | 26 / 5 / 9 |
| 每 sample event 數 | 6–1,362；平均 279.03，中位數 240.5 |
| 已知真實事件順序的 canonical traces | 0 |
| VirusTotal requests | 83 |

Event type 分布：file 6,015、registry 3,656、module 787、network 280、
process 249、service 98、command 55、API 21。所有 11,161 個 embedding 均為
有限值，L2 norm 約為 1。

## 目錄與資料欄位

### `raw/`

- `AU-PEMal-2025-V2.csv`：不可修改的來源表。
- `source_provenance.json`：來源 commit、URL 與 checksum。
- `vt/files/{sha1}.json`：VT file object 完整回應。
- `vt/behaviours/{sha256}.json`：VT behavior reports 完整回應。

### `artifacts/samples.csv`

| 欄位 | 說明 |
|---|---|
| `sample_key` | 以來源 SHA-1 建立的穩定 join key |
| `source_sha1` | AU-PEMal 提供的 SHA-1 |
| `sha256` | VT 回傳的 canonical SHA-256 |
| `split` | 固定的 `train`、`validation` 或 `test` |
| `lookup_status` | `success`、`not_found`、`pending` 等查詢狀態 |
| `source_dataset` | 來源資料集名稱 |
| `schema_version` | 本資料快照 schema 版本 |

### `artifacts/source_features.csv`

每列一個 candidate，以 `sample_key` join。其餘 35 欄是 AU-PEMal 原始的數值
特徵，包含 process/file/registry/network/API/DLL 彙總，以及 PE header、image、
section size/address 等靜態特徵。此檔已移除 SHA、Class、Category、Family 和 y，
避免將 label 或 identity 混入模型輸入。

### `artifacts/traces.csv`

| 欄位 | 說明 |
|---|---|
| `trace_id` | sample 與 VT behavior report 衍生的穩定 ID |
| `sample_key` | 對應 sample |
| `split` | 繼承 sample split |
| `vt_behavior_id` | VT behavior object ID |
| `sandbox_name`、`analysis_date` | Sandbox 與分析日期 |
| `event_count`、`field_coverage` | 可正規化事件數與有效 behavior 欄位數 |
| `is_canonical` | 是否為此 sample 選定的主要 report |
| `trace_quality` | `valid` 或 `insufficient_events` |
| `ordering_known` | 是否所有事件都有可信 timestamp |
| `sequence_eligible` | 是否允許作為時間序列模型輸入 |
| `raw_report_path` | 對應的快取 JSON 相對路徑 |
| `network_policy` | 原 sandbox network policy；未知時為 `unknown` |

### `artifacts/events.csv`

| 欄位 | 說明 |
|---|---|
| `trace_id`、`event_index` | Event join key；`event_index` 不一定代表時間 |
| `event_type` | API、command、process、file、registry、service、module、network |
| `operation` | open、write、create、connect 等正規化操作 |
| `object_role` | 操作對象的角色 |
| `result` | 有提供時的結果或狀態 |
| `canonical_text` | MiniLM 的實際輸入文字，不含 labels、SHA 或 VT verdict |
| `observed_at` | 原始事件時間；來源未提供時為空 |
| `ordering_known` | 此 trace 是否具有可信排序 |
| `object_known`、`result_known` | 欄位 known masks |

### `artifacts/event_embeddings_minilm384.csv`

包含 `trace_id`、`event_index` 及 `embedding_000`–`embedding_383`。不包含 y、
category、family、SHA、sandbox 或來源欄位。與 `events.csv` 為一對一 join。

### Label files

- `sample_labels.csv`：`sample_key`、binary `y/y_known`、`category`、`family`、
  `label_source`、`label_confidence`。`y=0` 為 benign，`y=1` 為 malicious。
- `concept_labels.csv`：trace-level ATT&CK technique/tactic、known mask、value、
  evidence source/scope。缺少 evidence 時是 unknown，不視為負例。

### 品質與可重現資訊

- `embedding_metadata.json`：模型 revision、套件版本、normalization 版本與文字 checksum。
- `quality_report.json`：joins、missingness、ordering、VT 狀態與最低門檻結果。
- `snapshot.json`：來源、設定及所有 artifact checksums。
- `source_validation.json`：重複與衝突 SHA-1 詳細資料。

## EDA 與互動式 3D UMAP

EDA 產物位於 `artifacts/eda/`：

- [Sample-level：良性／惡意](artifacts/eda/umap_sample_binary_3d.html)
- [Sample-level：惡意 family，良性灰色](artifacts/eda/umap_sample_family_3d.html)
- [Event-level：良性／惡意](artifacts/eda/umap_event_binary_3d.html)
- [Event-level：惡意 family，良性灰色](artifacts/eda/umap_event_family_3d.html)
- [EDA 報告](artifacts/eda/EDA_REPORT.md)

HTML 使用同目錄的 `plotly.min.js`，可離線旋轉、縮放、選取 legend 及查看 hover
資訊。UMAP 固定 seed `20261002`、3 維、cosine metric、`n_neighbors=15`、
`min_dist=0.1`。Sample-level 向量是同一 sample 所有 event embeddings 的平均。

## 限制

- 所有 40 個 canonical traces 都是 `ordering_known=0`；`event_index` 與 UMAP 軸
  都不代表執行時間，不應直接作為 GRU/LSTM 的真實時間序列。
- 40 筆 dynamic samples 只適合 feasibility pilot，不足以支持可靠的 family-level
  泛化結論；多數惡意 family 只有一筆 sample。
- 每個 sample 的事件量差異很大。Event-level 圖會讓事件多的 sample 具有較高視覺
  權重；評估分類時應以 sample 為切分與計分單位。
- UMAP 是非線性視覺化，圖上的群聚或距離不可直接視為分類效能。
