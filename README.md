# YOLOX 公路監視器車輛辨識

使用官方預訓練 YOLOX-S ONNX 模型同時辨識兩路即時監視器串流，左右並排顯示，各自等比例縮小至 320×240 面板，完整呈現兩個畫面。只處理兩組 COCO 目標：

- 汽車、卡車、巴士：紅色外框。
- 摩托車、腳踏車：綠色外框。

畫面左上角顯示兩組目前影格數量，其餘類別不顯示。此版本不使用追蹤器；畫面數量是逐影格偵測結果，不代表整段串流中不重複車輛的累計數。

預設左側為公路局台1線 92K+450，右側為臺南市 AED99D50 監視器。每路串流獨立讀取及自動重連，其中一路中斷不會停止另一路。

## 直接執行

```powershell
uv sync
uv run python vehicle_counter.py
```

預設左右來源：

```text
左：  http://cctv-ss04.thb.gov.tw/T1-92K+450
右：  https://trafficvideo3.tainan.gov.tw/aed99d50
```

程式會持續解析 multipart MJPEG 影格，並在每一路串流獨立解碼後交由同一個模型推論。預設會自動下載官方 `yolox_s.onnx` 模型到 `weights/`，使用 ONNX Runtime CPU 推論。

辨識視窗按 **Enter** 將含偵測框與左上角數量的畫面存至 `01_監視器`，按 **Esc** 離開。截圖檔名為臺北時間 `yyyymmddhhmmss.jpg`；同秒重複截圖會加上序號以免覆寫。

網路串流若中斷，程式會每隔 3 秒嘗試重新連線並繼續辨識；重連期間也可按 **Esc** 離開。

## 設定

可以複製 `.env.example` 為 `.env` 修改串流來源，或以命令列分別指定左右畫面：

```powershell
uv run python vehicle_counter.py --left-source "http://camera-left/stream" --right-source "https://camera-right/stream"
uv run python vehicle_counter.py --confidence 0.35
```

`STREAM_URL_THB` 設定左側來源，`STREAM_URL_TAINAN` 設定右側來源；舊有的 `STREAM_URL` 仍可作為左側來源覆寫。`--model` 可指定其他 YOLOX ONNX 模型；`--confidence` 預設為 `0.3`。YOLOX-S 比 nano/tiny 型號計算量較高，實際影格率會依 CPU 效能而變化。

## 純瀏覽器版（GitHub Pages）

公開網站：[Roadwatch 車輛辨識](https://abwinnie.github.io/roadwatch-vehicle-detection/)。訪客免安裝、免登入即可使用。`docs/` 是靜態網站版本，使用 ONNX Runtime Web/WASM 在訪客瀏覽器本機執行 YOLOX-S；可載入兩段本機影片，或讀取有允許跨來源（CORS）的 MJPEG 網址。影片不會上傳到網站。

Pages 建置會把模型放入網站產物，不需要把大型 ONNX 檔加入 Git：

```powershell
python scripts/build_pages.py
python -m http.server 8000 --directory dist
```

瀏覽 `http://localhost:8000` 可預覽。`.github/workflows/pages.yml` 設定為 `main` 分支更新時建置並部署，也能從 Actions 手動觸發。若 Git 儲存庫預設分支不是 `main`，需先在 workflow 調整分支名稱。

### 純靜態版限制

GitHub Pages 只能提供靜態檔案，不能代替監視器伺服器加入 CORS 標頭或轉送串流。瀏覽器需要以 `fetch` 取得 MJPEG 影格才能辨識；來源沒有允許 Pages 網域的 `Access-Control-Allow-Origin` 時，可能只能透過 `<img>` 預覽，不能讀取影格辨識。以目前端點探測，臺南來源回應 multipart MJPEG 但未看到 CORS 允許標頭；公路局來源使用 HTTP，而 Pages 是 HTTPS，瀏覽器可能依 Mixed Content 規則封鎖。兩路預設來源因此不能保證在 Pages 直接完成推論；本機影片上傳可用於免後端操作與展示。

網站模型由 `scripts/build_pages.py` 優先複製本機 `weights/yolox_s.onnx`，若建置環境沒有該檔則從官方 YOLOX release 下載。建置流程會將模型切成小型靜態檔，網站逐段載入並在片段中斷時自動重試；固定版本的 ONNX Runtime Web/WASM 也會隨網站一併發佈，避免訪客瀏覽器另連 jsDelivr。首次造訪需下載約 36 MB 模型及瀏覽器執行元件。若要讓目前兩支監視器網址在公開網站上都可直接辨識，需提供具授權、HTTPS 與 CORS 設定的串流服務，或另建受控的後端轉送服務。
