# Versioned dataset artifacts

`data/au_pemal_2025/pilot_v1/` 與
`data/api_traces_malware_detection/pilot_v1/` 是平行、獨立的 dataset snapshots。此 repository
只版本化能直接供分析/訓練使用的重要輸出；raw downloads、VirusTotal responses、原始 selected
traces、candidate cache、model cache、logs、NumPy cache、checkpoint 與任何 `.part` 檔皆維持
本機 ignored。

## Git 與 Git LFS 的分工

一般 Git 追蹤 samples、labels、traces、metadata、quality report、snapshot、正規化 events 及
精簡 EDA tables。下列大型或高頻生成的 matrix 使用 Git LFS：

- AU-PEMal `event_embeddings_minilm384.csv`
- API Traces `events.csv`
- API Traces `sample_embeddings_gte_modernbert768.csv`
- 兩份 dataset 的互動式 UMAP HTML 與其本機 `plotly.min.js`

使用 Git LFS 可以避免把大型 blob 永久塞入一般 Git history。LFS 檔每次修改都會上傳完整的新
object，因此 dataset snapshot 應盡量 immutable；要做新版時建立 `pilot_v2/`，不要反覆覆寫
`pilot_v1/`。

## 在另一台主機取得資料

```powershell
git lfs install
git clone https://github.com/Heng1222/multimodal-pe-research.git
cd multimodal-pe-research
git lfs pull
git lfs ls-files
```

如果 repository 已存在，使用：

```powershell
git pull
git lfs pull
```

安裝 Git LFS 的正常 clone/pull 通常會自動下載 LFS objects；額外執行 `git lfs pull` 可確保
不是只有 pointer files。`git lfs checkout` 可在 object 已下載但 working tree 仍是 pointer
時還原實際內容。

互動圖使用 `include_plotlyjs="directory"` 產生；HTML 與 `plotly.min.js` 必須留在同一個
`artifacts/eda/` 目錄。Pull 完成後可直接用瀏覽器開啟 HTML，不需要 CDN 或網路。也可以用
對應 pipeline 的 `eda` command，從已追蹤的 embedding、labels 與固定 seed 重新產生。

## 更新 snapshot

先完成對應 pipeline 的 `validate`，確認 `quality_report.json` 的 `valid` 為 `true`，再執行：

```powershell
git add .gitattributes data/.gitignore data/README.md
git add data/au_pemal_2025/pilot_v1
git add data/api_traces_malware_detection/pilot_v1
git lfs status
git status
```

不得強制加入 ignored 的 `raw/`、`cache/`、`logs/` 或 `.part`。目前 API Traces 的 frozen GTE
embedding 若仍只有 `.part`，代表工作尚未完整結束；正式 CSV、coverage、metadata、quality
report 與 snapshot 應在 embedding 和 validate 完成後一起提交。
