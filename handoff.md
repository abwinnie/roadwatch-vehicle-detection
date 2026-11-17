# 交接紀錄

## 後續工作規則

- 每次完成專案內的檔案、程式碼、依賴或環境設定變更時，必須在同一個工作階段更新本檔，再結束回覆。
- 每筆紀錄包含日期（Asia/Taipei）、變更摘要、受影響檔案，以及執行過的安裝或檢查；未執行的檢查也要明確註明。
- 新紀錄追加在本檔底部，保留既有紀錄。若該次變更只修改本檔，也要在該次紀錄註明。
- 開始後續工作前先閱讀本檔，並在每次變更完成後補上紀錄。

## 專案現況

- 專案目錄：`D:\夜市小霸王_CODEX`
- 目標：讀取網路影像串流，辨識並計算車輛數量，提供統計與畫面顯示。
- 已建立 Python 3.12 專案設定，使用 `uv` 管理虛擬環境與鎖定依賴。
- 執行環境：`uv` 0.13.8、Python 3.12.16、專案虛擬環境 `.venv`。
- 主要依賴：`numpy`、`opencv-python`、`onnxruntime`、`requests`、`python-dotenv`、`tzdata`；精確解析版本記錄於 `uv.lock`。
- `vehicle_counter.py` 使用 YOLOX-S ONNX 逐影格辨識；不使用追蹤器，只統計當前影格數量並顯示在各面板左上角。
- 目前左右並排顯示兩路監視器，每路以等比例縮放完整顯示於 320×240 面板；統計 HUD 與車輛框旁的類別/信心文字採較小字級。
- 目前監視器設定只辨識/顯示汽車組（COCO car/bus/truck，紅框）與摩托車組（motorcycle/bicycle，綠框）；其他類別不進入畫面。
- 按 Enter 將已標註畫面存到 `01_監視器\yyyymmddhhmmss.jpg`；按 Esc 離開。時間戳採 Asia/Taipei。
- 公路局 `multipart/x-mixed-replace` MJPEG 網址由 `requests` 持續讀取，再逐張 JPEG 解碼供 YOLOX 辨識。
- 網路串流中斷時每 3 秒自動重連，重連期間可按 Esc 離開。
- 目前雙監視器辨識工作已啟動，Codex exec session ID `14770`；YOLOX-S 權重為 `weights\yolox_s.onnx`。OpenCV 視窗標題使用 ASCII：`YOLOX - Dual Camera Vehicle Detection`。
- 新增純瀏覽器網站草稿於 `docs/`：ONNX Runtime Web/WASM 在瀏覽器端推論，支援本機影片及具 CORS 許可的 multipart MJPEG；GitHub Pages workflow 與 `scripts/build_pages.py` 已備妥，但尚未推送或部署。訪客無需登入；要提供公開網址，GitHub 儲存庫/Pages 必須公開。
- 工作目錄目前不是 Git 儲存庫（根目錄沒有 `.git`）；尚無法確認遠端、預設分支或實際 Pages 設定。
- NVIDIA GPU/CUDA 尚未設定或確認。

## 變更紀錄

### 2026-11-17（Asia/Taipei）

- 建立專案開發環境：安裝 `uv`，由 `uv` 安裝 Python 3.12.16，建立 `.venv` 並執行 `uv sync`。
- 建立專案依賴設定、Python 版本指定、忽略規則、串流網址範本及開發說明。
- 新增本交接檔，記錄目前狀態並訂定後續變更紀錄規則。
- 受影響檔案：`pyproject.toml`、`uv.lock`、`.python-version`、`.gitignore`、`.env.example`、`README.md`、`handoff.md`。
- 檢查：`uv sync` 成功，安裝 47 個套件；未執行串流辨識或計數流程。

### 2026-11-17（Asia/Taipei）

- 新增 `vehicle_counter.py`：可讀取 `.env` 的 `STREAM_URL` 或命令列來源，使用 Ultralytics YOLO 偵測車輛、Trackers ByteTrack 保留追蹤 ID，並以 Supervision 計算自訂計數線的穿越次數。
- 畫面顯示當前追蹤車輛數、IN/OUT 累計穿越次數、總穿越次數及平均 FPS；離開時在終端機列出本次摘要。支援 RTSP/HTTP 來源、影片路徑及攝影機編號。
- CLI 已驗證計數線座標與信心閾值格式，拒絕超出範圍及非有限數值。
- 更新 README 執行方式、計數邏輯、參數與限制；新增 `trackers`、`python-dotenv` 依賴並更新鎖定檔。
- 受影響檔案：`vehicle_counter.py`、`pyproject.toml`、`uv.lock`、`README.md`、`handoff.md`。
- 檢查：`uv add trackers python-dotenv` 成功並更新虛擬環境；未執行程式測試或端到端串流辨識（尚無可用串流網址）。

### 2026-11-17（Asia/Taipei）

- 依使用者指定調整辨識目標為汽車/卡車/巴士及摩托車/腳踏車，分別使用紅色、綠色外框，並移除其餘類別的顯示與計數。
- 新增 Enter 截圖與 Esc 離開；截圖以臺北時間命名並存入專案根目錄 `01_監視器`。
- 更新 README 的分類、操作方式、截圖位置及計數說明。
- 監視器來源：`https://trafficvideo.tainan.gov.tw/786ae899`。HTTP 回應為 200，Content-Type 為 `multipart/x-mixed-replace`（MJPEG）。
- 受影響檔案：`vehicle_counter.py`、`README.md`、`handoff.md`。
- 執行狀態：已執行 `uv run python vehicle_counter.py --source "https://trafficvideo.tainan.gov.tw/786ae899"`，首次下載 `yolo26n.pt` 權重約 5.3 MB，程式輸出「辨識已啟動」，Codex exec session ID 為 `69737`；可在辨識視窗按 Enter 截圖、按 Esc 結束。
- 檢查：確認監視器 HTTP 串流標頭及程式進入辨識執行狀態；未進行準確率評估或長時間穩定性測試。

### 2026-11-17（Asia/Taipei）

- 使用者要求持續使用串流影像。確認原生 OpenCV 讀取此 multipart MJPEG 網址時只處理 1 個影格後中斷；改為用 `requests` 持續讀取 multipart 回應，拆出 JPEG 影格，再交給 OpenCV 解碼與 YOLO 辨識。
- 新增直接依賴 `numpy`、`requests` 並更新 `uv.lock`；README 說明此串流的逐影格讀取方式。
- 重新啟動指定監視器，HTTP 串流探測確認可讀取多個 JPEG 影格；辨識視窗持續執行，已產生截圖 `01_監視器\20261117200037.jpg`，畫面含汽車紅框、摩托車/腳踏車綠框與分組統計。
- 受影響檔案：`vehicle_counter.py`、`pyproject.toml`、`uv.lock`、`README.md`、`handoff.md`；執行產物：`01_監視器\20261117200037.jpg`。
- 執行狀態：命令 `uv run python vehicle_counter.py --source "https://trafficvideo.tainan.gov.tw/786ae899"` 仍在執行，Codex exec session ID `95089`；按 Enter 截圖、按 Esc 離開。
- 檢查：Requests 串流探測讀到 2 個 JPEG 影格；持續辨識執行中並成功輸出標註截圖；未進行準確率評估。

### 2026-11-17（Asia/Taipei）

- 使用者指定新來源 `https://cctv-ss04.thb.gov.tw/T1-92K+450`，並要求改 YOLOX、移除追蹤、只在左上角顯示目前數量。
- 檢查端點後發現 HTTPS 回應 404，HTTP `http://cctv-ss04.thb.gov.tw/T1-92K+450` 回應 200 並提供 `multipart/x-mixed-replace` 連續 MJPEG；因此將預設來源設為同一監視器的 HTTP 串流端點。
- 將推論改為官方 YOLOX-S ONNX + ONNX Runtime，按每一影格計算偵測數，不用追蹤器。汽車/卡車/巴士紅框，摩托車/腳踏車綠框；畫面只保留這兩組及左上角數量。
- 保留 Enter 截圖與 Esc 離開；更新 README 與 `.env.example`。
- 移除 `ultralytics`、`supervision`、`trackers` 依賴，加入 `onnxruntime` 並更新鎖定檔；官方 `yolox_s.onnx` 已下載到 `weights\`。
- 受影響檔案：`vehicle_counter.py`、`pyproject.toml`、`uv.lock`、`README.md`、`.env.example`、`handoff.md`；模型產物：`weights\yolox_s.onnx`。
- 執行狀態：`uv run python vehicle_counter.py` 已成功啟動 YOLOX-S 串流辨識，Codex exec session ID `91588`；按 Enter 截圖、按 Esc 離開。
- 檢查：來源 HTTP 標頭為 200/MJPEG；HTTPS 端點為 404；模型下載成功，程式載入模型並進入持續辨識。未進行準確率評估。

### 2026-11-17（Asia/Taipei）

- 使用者再次要求使用串流影像。檢查發現前次 YOLOX-S 執行已因串流讀取中斷而退出。
- 為 HTTP/RTSP/RTMP 等網路來源加入自動重連：串流讀取失敗後每 3 秒重新連線，保留即時視窗，重連等待期間仍可按 Esc 結束。
- README 補上自動重連說明，重新執行 YOLOX-S 辨識。
- 受影響檔案：`vehicle_counter.py`、`README.md`、`handoff.md`。
- 執行狀態：`uv run python vehicle_counter.py` 已重新啟動，Codex exec session ID `81704`；按 Enter 截圖、按 Esc 離開。
- 檢查：新執行載入 YOLOX-S 並持續執行超過 40 秒，期間未回報串流中斷；未進行準確率評估。

### 2026-11-17（Asia/Taipei）

- 修正 Enter 截圖會因 Windows 缺少 `Asia/Taipei` 時區資料而結束辨識的問題：加入 `tzdata` 依賴，並將截圖錯誤限制在截圖操作內，串流保持運作。
- 依使用者要求縮小左上角統計字樣並讓面板依影格尺寸自動調整，最大限制於畫面左上四分之一；另縮小車輛框旁類別與信心值文字。
- 重新啟動 YOLOX-S 串流辨識，Codex exec session ID `39964`。
- 受影響檔案：`vehicle_counter.py`、`pyproject.toml`、`uv.lock`、`handoff.md`。
- 檢查：更新後執行成功；Enter 截圖成功後串流仍持續，自動重連也成功；HUD/標籤調整後程序持續執行，未進行準確率評估。

### 2026-11-17（Asia/Taipei）

- 加入臺南 AED99D50 監視器 `https://trafficvideo3.tainan.gov.tw/aed99d50`，與公路局台1線 92K+450 串流同時辨識；兩路各自讀取、失聯後獨立重連。
- 將兩路來源分別設定為左、右面板；在原始解析度執行偵測，再等比例縮至 320×240，並排組成完整雙路畫面。Enter 截圖改存左右合成畫面。
- 新增 `--left-source`、`--right-source` 及 `STREAM_URL_THB`、`STREAM_URL_TAINAN` 設定；更新 `.env.example` 與 README。
- 依使用者截圖修正 OpenCV 視窗標題中文字顯示亂碼，改為 ASCII 標題 `YOLOX - Dual Camera Vehicle Detection`。
- 受影響檔案：`vehicle_counter.py`、`.env.example`、`README.md`、`handoff.md`；執行產物：`01_監視器\20261117203348.jpg`。
- 執行狀態：目前 `uv run python vehicle_counter.py` 持續執行，Codex exec session ID `14770`；視窗標題已由 Windows 程序資訊確認為 ASCII 文字，Enter 合成畫面截圖成功。
- 檢查：雙路視窗啟動成功，截圖存檔成功，Windows 程序資訊確認標題為 ASCII 文字。左側串流仍間歇回報 `stream ended` 並自動重連；雙路視窗保持運作。未做車輛偵測準確率評估。

### 2026-11-17（Asia/Taipei）

- 新增純靜態瀏覽器版 `docs/`：左右兩面板、本機影片上傳、CORS 允許的 multipart MJPEG 讀取、ONNX Runtime Web/WASM YOLOX-S 推論、車種框/即時影格數及合成截圖；訪客影片留在瀏覽器。
- 新增 `scripts/build_pages.py`，從本機權重複製模型；GitHub runner 沒有未追蹤權重時，改從 YOLOX 官方 release 下載約 36 MB 模型到 `dist/models/`。網站與模型皆使用相對路徑，`dist/` 加入忽略規則。
- 新增 `.github/workflows/pages.yml`，建置 Pages artifact 並部署；只準備 workflow，未 push、未執行 Actions、未部署。公開免登入存取要求 GitHub 儲存庫與 Pages 公開；本工作目錄目前沒有 `.git`。
- README 說明純靜態架構、建置方式及限制。瀏覽器端無伺服器可替攝影機加 CORS 標頭；本次臺南端點回應 200/MJPEG，但未回傳 `Access-Control-Allow-Origin`；公路局 HTTP 端點本次 8 秒探測逾時，且在 HTTPS Pages 可能受 Mixed Content 限制。網站因此明確提示遠端串流僅在符合瀏覽器來源規則時才能推論；本機影片可作為前端辨識來源。
- 受影響檔案：`.gitignore`、`README.md`、`docs/index.html`、`docs/styles.css`、`docs/app.js`、`scripts/build_pages.py`、`.github/workflows/pages.yml`、`handoff.md`；建置產物為忽略的 `dist/`。
- 檢查：`build_pages.py` 成功產生網站及 35,858,002-byte 模型；`node --check docs/app.js`、Python 編譯檢查、HTML 解析、相對資源/模型存在檢查成功；本機 HTTP server 對頁面、JS、CSS、模型皆回應 200；ONNX Runtime Web 1.30.0 JS/WASM CDN 與官方模型 release HEAD 皆回應 200。桌面瀏覽器畫面檢查工具初始化失敗，因此尚未完成瀏覽器內實際模型推論檢查；Pages workflow 也尚未執行。

### 2026-11-17（Asia/Taipei）

- 在公開 GitHub repository 準備前，將 `01_監視器/` 加入忽略規則，避免把本機留存的監視器截圖一併發佈。
- 透過本機 Pages 預覽，在瀏覽器成功載入 YOLOX-S 模型並以瀏覽器 WASM CPU 推論一段由既有截圖製成的 3 秒測試影片；畫面顯示 Cars 14、Moto/Bike 0。這只確認推論流程可執行，不代表偵測準確率。
- 更新 Pages workflow 的 build 權限為 `contents: read`、`pages: read`，並使用 `actions/deploy-pages@v5`。
- 受影響檔案：`.gitignore`、`.github/workflows/pages.yml`、`handoff.md`。
- 檢查：瀏覽器模型狀態為「模型就緒 · WASM CPU」，測試影片播放期間顯示偵測框與數量；未評估準確率或效能。
