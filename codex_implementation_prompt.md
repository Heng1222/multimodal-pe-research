# 交給 Codex 的實作 Prompt：良性優先多模態 PE 資料管線

以下內容可整段貼給 Codex。若專案中已有 `AGENTS.md`、程式架構或工具鏈，應先遵循既有規範，並以最小變更整合。

---

## Role

你是此研究專案的資深資料工程與資安研究實作工程師。請在目前 repository 中建立一套「良性優先、之後可直接接入惡意 PE」的可重現資料收集 pipeline。

## Goal

目前尚未取得正式惡意 PE，但已可開始建立良性資料集與共用 pipeline。請完成第一個可用版本，使我們能：

1. ingest 良性 PE；
2. 計算 hash、去重並驗證 PE；
3. 保存來源、標籤、工具版本與分析狀態；
4. 產生結構化靜態分析結果；
5. 產生可重現的 manifest、品質報告與 dataset snapshot；
6. 預留 assembly、VirusTotal 與 sandbox 動態分析 adapter；
7. 未來拿到惡意 PE 後，不改 schema 即可用同一流程處理。

這一階段不要實作或訓練解纏模型。重點是資料規格、可重現性、資料品質與安全邊界。

## Context

研究最終希望分離：

- malicious behavior；
- benign/background/noise；
- modality／observation-specific information；
- 後續的 family／type／residual；
- missing modality 與 MITRE ATT&CK 對映。

因此，良性與惡意樣本必須使用相同 pipeline、工具版本、輸出 schema 與分析環境。每個 sample 以小寫 SHA-256 作為唯一 `sample_id`，所有衍生模態透過該 ID 對應。

## First actions

1. 先閱讀 repository、`AGENTS.md`、README、現有 dependencies、tests 與目錄結構。
2. 回報你發現的既有架構、可重用元件與會影響實作的限制。
3. 提出短而具體的實作計畫，列出將新增或修改的檔案、資料流、驗證方式與安全邊界。
4. 除非發現會實質改變 schema、安全性或既有架構的阻斷問題，否則在同一個任務中繼續實作，不要只停在計畫。
5. 若 repository 幾乎為空，建立最小、清楚、可測試的 Python 3.11+ 專案；若已有技術棧，優先沿用。

## Required architecture

建立可插拔、可恢復、idempotent 的 stage pipeline，至少包含：

1. `init`：初始化 dataset workspace、設定與 manifest。
2. `ingest`：讀取輸入 PE、計算 SHA-256／SHA-1／MD5、去重、保存 provenance。
3. `validate-pe`：檢查 magic、PE header、parser 狀態與基本架構。
4. `analyze-static`：輸出 PE headers、sections、imports、exports、resources、strings、signature 狀態、entropy 與疑似 packer 狀態。
5. `verify-benign`：保存良性標籤依據、來源、簽章與可選的多引擎查詢摘要；不得只靠「檔案在本機」判定良性。
6. `analyze-assembly`：以 adapter 介面呼叫 Ghidra 或等價工具；外部工具未安裝時要明確回報 skipped／unavailable，不可使整條 pipeline 崩潰。
7. `lookup-intel`：VirusTotal adapter 預設只用 hash lookup；未命中時禁止自動上傳。
8. `submit-sandbox`／`collect-sandbox`：只建立 adapter、job state、mock 與結果 schema；不得在開發主機直接執行未知 PE。
9. `validate-dataset`：驗證 schema、artifact、hash 一致性、必要欄位與 stage 狀態。
10. `report`：產生樣本數、重複率、成功率、錯誤類型、來源、簽章與模態完整率報告。
11. `snapshot`：保存 schema version、sample IDs、篩選規則、config hash、pipeline commit 與產生時間。

CLI 名稱可依 repository 慣例調整，但功能與邊界必須保留。若使用 Python CLI，可考慮 Typer 或既有框架。

## Canonical storage model

請採取以下概念，實際路徑可配合現有專案調整：

```text
dataset_workspace/
  config/
  manifest.sqlite
  raw/
    sha256_prefix/sample_id/original.pe
  artifacts/
    sample_id/
      static/pe_metadata.json
      static/imports.json
      static/exports.json
      static/sections.json
      static/strings.txt
      assembly/
      intel/
      dynamic/
  reports/
  snapshots/
  logs/
```

要求：

- raw store 視為 immutable；分析工具不得原地修改原始 PE。
- derived artifacts 必須可由 raw PE、config 與工具版本重建。
- canonical manifest 使用 SQLite；提供 JSONL 匯出供交換與版本紀錄。
- 不把真實 PE bytes 存進 git。
- 不把 API keys、完整本機來源路徑或敏感資訊寫進 git、log 或公開報告。

## Minimum schema

請建立有版本的 schema，至少涵蓋：

### Sample identity

- `sample_id`（等於小寫 SHA-256）
- `sha256`、`sha1`、`md5`
- `original_filename`
- `file_size`
- `is_valid_pe`
- `created_at`、`updated_at`

### Provenance

- `source_type`
- `source_reference`
- `collected_at`
- `license_or_usage_note`
- 同一 SHA-256 可以有多筆 provenance，不得被後來的 ingest 覆蓋。

### Labels

- `binary_label`：`benign | malicious | unknown`
- `label_source`
- `label_confidence`：`high | medium | low | unknown`
- `family_label`
- `family_vendor`，canonical 預設為 Microsoft
- `malware_type`
- `mitre_techniques` 預留欄位

### Static metadata

- machine type、subsystem、原始 compile timestamp
- signed status 與可選憑證摘要
- sections、entropy、imports、exports、resources
- packer status：`packed | unpacked | suspected | unknown`
- artifact paths 與 artifact hashes

### Reproducibility and stage state

- `schema_version`
- `pipeline_version` 或 git commit
- `config_hash`
- `tool_versions`
- `analysis_environment_id`
- 每個 stage 的 `pending | running | success | failed | skipped`
- attempt count、started／finished time、error code、error message

## Stage behavior

- 每個 stage 必須可單獨執行、批次執行及安全重跑。
- 相同輸入、設定與工具版本重跑時不得產生重複 sample 或不必要地覆蓋有效產物。
- 若設定或工具版本改變，應能辨識產物過期並選擇重新產生。
- stage failure 應記錄後繼續處理其他樣本，不得讓單一壞檔終止整批工作。
- timeout、crash、parser error、tool unavailable、no behavior 都應有穩定 error/status code。
- 所有寫入採原子方式；中斷後可恢復，不留下看似成功的半成品。
- raw 與 artifact 的 hash 必須可驗證。

## Feature safety and leakage controls

下列欄位要保存供稽核，但預設標記為不可輸入模型：

- original filename／source path；
- source type；
- signed status／certificate vendor；
- VirusTotal detection count／vendor labels；
- collection time、OS image、sandbox profile；
- stage success／failure；
- 直接包含答案的 report 欄位。

請在 schema 或 feature registry 中明確區分：

- `model_eligible`；
- `audit_only`；
- `sensitive`；
- `label_or_leakage_risk`。

## Security constraints

這是惡意程式研究 pipeline，即使目前先處理良性資料，也必須預先遵守：

- 絕不在開發主機直接執行待分析 PE。
- 動態分析只能透過明確設定的隔離 sandbox adapter。
- sandbox 未設定時，動態 stage 必須安全失敗或標記 skipped。
- 不自行停用主機安全機制。
- 不連接個人帳號、校園內網、共享資料夾或剪貼簿。
- 網路預設封鎖；若未來需要模擬網路，只接受明確設定的受控 sinkhole／模擬服務。
- dropped files 預設只 hash 與建立關係，不遞迴執行。
- VirusTotal 未命中時不得自動上傳。
- 測試只能使用合成 bytes、最小安全 PE fixture 或 repository 已明確標記的 benign fixture。
- 真實 PE、hash inventory 和安全情報資料預設不提交 git，除非 repository 已有經審核規則。

如果任何現有需求與上述安全邊界衝突，停止衝突部分，完成其他安全範圍，並清楚列出 blocker。

## Configuration

建立可版本控制、可驗證的設定檔，至少包括：

- workspace paths；
- enabled stages；
- timeouts；
- parser／disassembler／sandbox adapter；
- tool executable paths；
- analysis environment ID；
- artifact retention；
- VirusTotal hash lookup 是否啟用；
- retry policy；
- log level。

Secrets 僅從環境變數或 repository 既有 secret provider 取得；範例設定只能放 placeholder。

## Dependencies

- 優先使用成熟、維護中的 PE parser；若 repository 已使用 `pefile`、LIEF 或其他工具，沿用既有選擇。
- 外部工具必須透過 adapter 與 dependency detection 封裝。
- 不為尚未啟用的重量級工具強迫加入執行期依賴。
- dependencies 應鎖定版本或遵循 repository 現有 lockfile。

## Tests

至少完成：

### Unit tests

- hash 計算與 `sample_id` 規則；
- duplicate ingest 與多 provenance；
- schema validation；
- stage state transition；
- config hash；
- artifact 原子寫入與 hash 驗證；
- parser error、timeout、tool unavailable；
- feature registry 的 leakage 分類。

### Integration tests

- 初始化 workspace；
- ingest 安全 fixture；
- 靜態分析；
- 重跑 idempotency；
- 單一壞檔不終止 batch；
- 產生 report 與 snapshot；
- mock VirusTotal hash hit／miss；
- mock sandbox submission／collection，不執行任何 fixture。

### Migration tests

- schema version 可辨識；
- 不相容版本給出明確錯誤；
- 若實作 migration，必須有 round-trip 或 fixture 測試。

## Documentation deliverables

請新增或更新：

1. README：快速開始、架構、CLI 範例與安全警告。
2. `docs/schema.md`：欄位、型別、允許值、model eligibility 與版本政策。
3. `docs/pipeline.md`：stage、資料流、狀態轉移、重試與恢復。
4. `docs/security.md`：原始 PE、動態分析、網路、secrets 與 git 規則。
5. `docs/benign_collection.md`：良性來源、驗證、provenance 與偏誤注意事項。
6. 範例 config 與不含真實樣本的操作範例。

## Initial implementation scope

本次應完整實作：

- Phase A：schema、config、manifest、CLI、stage state、logging、fixture tests。
- Phase B：ingest、hash、dedup、PE validation、靜態分析、良性標籤與品質報告。
- Phase C 的安全骨架：assembly、VirusTotal、sandbox adapters 與 mocks；外部系統未設定時可降級。

本次不要：

- 執行未知 PE；
- 建立惡意能力或修改樣本；
- 自動上傳任何檔案到外部服務；
- 實作模型訓練；
- 假造真實 family／MITRE labels；
- 因尚未取得惡意資料而把 schema 寫死成 benign-only。

## Acceptance criteria

完成前必須滿足：

1. 新環境可依 README 初始化 workspace。
2. 可 ingest 一批安全 fixture，並以 SHA-256 去重。
3. 同一檔案從不同來源 ingest 時只建立一個 sample，但保留多筆 provenance。
4. 可產生結構化靜態資料、stage 狀態與 artifact hashes。
5. pipeline 中斷或重跑不會破壞已完成資料。
6. 外部工具或 API 未設定時會產生明確狀態，不影響可用 stage。
7. 可產生 JSONL manifest export、品質報告與可重現 snapshot。
8. feature registry 清楚排除來源、簽章、VT label 等預設 leakage 欄位。
9. 沒有真實 PE、secret、完整本機路徑或敏感報告被提交 git。
10. 所有新增測試通過；並執行 repository 適用的 lint、type check 與最小 smoke test。
11. 文件清楚說明目前只完成良性資料與共用 pipeline，尚未完成良惡模型研究。

## Final response

完成後請回報：

- 實作結果摘要；
- 新增／修改的重要檔案；
- CLI 使用範例；
- 執行過的 tests、lint、type check 與結果；
- 尚未設定的外部工具或環境；
- 已知限制與下一步；
- 是否有任何安全或資料品質 blocker。

若無法完成某項驗證，說明原因與下一個最小可行檢查。不要以未驗證的成功敘述代替實際結果。

---

## 使用者在交付 Codex 前可補充的資訊（非必要）

- repository 路徑或 GitHub repository；
- 偏好的 Python dependency manager；
- 已安裝的 PE parser／Ghidra／sandbox；
- 預計存放 dataset workspace 的磁碟；
- 是否已有安全的良性 PE 測試樣本；
- VirusTotal 是否只有查詢權限；
- 是否已有 Cuckoo／CAPE 隔離環境。

若未提供，Codex 應採安全預設、建立 adapter 與 mock，不應自行假設外部系統可用。
