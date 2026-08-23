# Configurations

可重現且不含秘密資訊的設定放在此處，依責任分為：

- `data/`：dataset workspace、pipeline stages、parser、adapter 與 retry policy。
- `model/`：feature set、architecture、optimizer 與 training policy。
- `experiment/`：dataset snapshot、seed、runs、evaluation 與比較設定。

API keys 與 credentials 僅能由環境變數或外部 secret provider 注入。
