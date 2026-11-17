(() => {
  "use strict";

  const MODEL_URL = new URL("./models/yolox_s.onnx", document.baseURI).href;
  const INPUT_SIZE = 640;
  const CLASS_COUNT = 80;
  const TARGET_CARS = new Set([2, 5, 7]);
  const TARGET_TWO_WHEEL = new Set([1, 3]);
  const CLASS_NAMES = { 1: "BIKE", 2: "CAR", 3: "MOTO", 5: "BUS", 7: "TRUCK" };
  const STRIDES = [8, 16, 32];
  const IOU_THRESHOLD = 0.45;
  const PANEL_WIDTH = 640;
  const PANEL_HEIGHT = 480;
  const cameras = [createCamera("left", "THB 92K+450"), createCamera("right", "TAINAN AED99D50")];

  let session = null;
  let schedulerStarted = false;
  let scratchCanvas = null;
  let scratchContext = null;
  let gridPoints = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function createCamera(id, title) {
    return {
      id,
      title,
      canvas: byId(`${id}-canvas`),
      context: byId(`${id}-canvas`).getContext("2d", { willReadFrequently: true }),
      fallback: byId(`${id}-preview`),
      video: byId(`${id}-video`),
      empty: byId(`${id}-empty`),
      status: byId(`${id}-status`),
      sourceLabel: byId(`${id}-source-label`),
      urlInput: byId(`${id}-url`),
      fileInput: byId(`${id}-file`),
      mode: "idle",
      generation: 0,
      controller: null,
      reader: null,
      objectUrl: null,
      latestBlob: null,
      lastInferenceAt: 0,
      lastInferenceMs: 0,
      detections: [],
    };
  }

  function setStatus(camera, text, kind = "") {
    camera.status.textContent = text;
    camera.status.className = `status-chip${kind ? ` ${kind}` : ""}`;
  }

  function setModelState(text, kind = "") {
    byId("model-state").textContent = text;
    byId("model-indicator").className = `model-indicator${kind ? ` ${kind}` : ""}`;
  }

  function drawIdle(camera, message) {
    const context = camera.context;
    context.fillStyle = "#0b1113";
    context.fillRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT);
    context.fillStyle = "#b9ed7b";
    context.font = "500 18px 'DM Mono', monospace";
    context.fillText(camera.title, 22, 36);
    context.fillStyle = "#9aacab";
    context.font = "14px 'Noto Sans TC', sans-serif";
    context.fillText(message, 22, PANEL_HEIGHT / 2);
  }

  function updateEmptyState(camera) {
    const hasPreviewImage = !camera.fallback.hidden;
    camera.empty.hidden = camera.mode !== "idle" || hasPreviewImage;
  }

  function stopCamera(camera, resetView = true) {
    camera.generation += 1;
    if (camera.controller) camera.controller.abort();
    if (camera.reader) {
      camera.reader.cancel().catch(() => {});
      camera.reader = null;
    }
    camera.controller = null;
    camera.latestBlob = null;
    camera.mode = "idle";
    camera.video.pause();
    camera.video.removeAttribute("src");
    camera.video.load();
    if (camera.objectUrl) URL.revokeObjectURL(camera.objectUrl);
    camera.objectUrl = null;
    camera.fallback.hidden = true;
    camera.fallback.removeAttribute("src");
    camera.sourceLabel.textContent = "尚未選擇來源";
    setStatus(camera, "待連線");
    document.querySelector(`.playback-toggle[data-camera="${camera.id}"]`).disabled = true;
    updateEmptyState(camera);
    if (resetView) drawIdle(camera, "連線監視器或載入影片");
  }

  function makeGridPoints() {
    const points = [];
    for (const stride of STRIDES) {
      const size = INPUT_SIZE / stride;
      for (let row = 0; row < size; row += 1) {
        for (let column = 0; column < size; column += 1) {
          points.push({ x: column, y: row, stride });
        }
      }
    }
    return points;
  }

  async function readModelWithProgress(response) {
    const total = Number(response.headers.get("content-length")) || 0;
    if (!response.body) {
      const buffer = await response.arrayBuffer();
      byId("model-progress").style.width = "100%";
      return new Uint8Array(buffer);
    }

    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      if (total > 0) {
        const percent = Math.min(100, Math.round((loaded / total) * 100));
        byId("model-progress").style.width = `${percent}%`;
        setModelState(`下載模型中 ${percent}%`);
      } else {
        setModelState(`下載模型中 ${(loaded / (1024 * 1024)).toFixed(1)} MB`);
      }
    }

    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return bytes;
  }

  async function loadModel() {
    if (session) return;
    const button = byId("load-model");
    const progressWrap = byId("model-progress-wrap");
    let loadStage = "推論引擎";
    button.disabled = true;
    button.textContent = "載入中…";
    progressWrap.hidden = false;
    byId("model-progress").style.width = "0%";
    setModelState("準備載入瀏覽器推論引擎…");

    try {
      if (!window.ort) throw new Error("推論引擎未載入，請重新整理網站後再試。");
      window.ort.env.wasm.wasmPaths = new URL("./vendor/", document.baseURI).href;
      window.ort.env.wasm.numThreads = 1;
      window.ort.env.wasm.proxy = false;

      loadStage = "模型下載";
      setModelState("下載 YOLOX-S 模型…");
      const response = await fetch(MODEL_URL, { cache: "force-cache" });
      if (!response.ok) throw new Error(`模型下載失敗（HTTP ${response.status}）。`);
      const modelBytes = await readModelWithProgress(response);
      if (modelBytes.byteLength < 1_000_000) throw new Error("模型檔案大小不正確。");

      loadStage = "WASM 初始化";
      setModelState("初始化 WASM 推論…");
      session = await window.ort.InferenceSession.create(modelBytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      if (!session.inputNames.length || !session.outputNames.length) throw new Error("模型輸入或輸出節點不存在。");
      scratchCanvas = document.createElement("canvas");
      scratchCanvas.width = INPUT_SIZE;
      scratchCanvas.height = INPUT_SIZE;
      scratchContext = scratchCanvas.getContext("2d", { willReadFrequently: true });
      gridPoints = makeGridPoints();
      byId("screenshot").disabled = false;
      byId("model-progress").style.width = "100%";
      setModelState("模型就緒 · WASM CPU", "ready");
      button.textContent = "模型已就緒";
      startScheduler();
    } catch (error) {
      session = null;
      const message = error instanceof Error ? error.message : String(error);
      const networkFailure = error instanceof TypeError && /failed to fetch/i.test(message);
      setModelState(
        networkFailure ? `網路載入失敗（${loadStage}），請檢查連線後重試。` : message,
        "error",
      );
      button.disabled = false;
      button.textContent = "重試載入模型";
    } finally {
      window.setTimeout(() => { progressWrap.hidden = true; }, 650);
    }
  }

  function preprocess(source) {
    const sourceWidth = source.videoWidth || source.naturalWidth || source.width;
    const sourceHeight = source.videoHeight || source.naturalHeight || source.height;
    if (!sourceWidth || !sourceHeight) throw new Error("目前影格沒有可讀取的尺寸。");

    const ratio = Math.min(INPUT_SIZE / sourceHeight, INPUT_SIZE / sourceWidth);
    const resizedWidth = Math.max(1, Math.floor(sourceWidth * ratio));
    const resizedHeight = Math.max(1, Math.floor(sourceHeight * ratio));
    scratchContext.fillStyle = "rgb(114, 114, 114)";
    scratchContext.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    scratchContext.drawImage(source, 0, 0, resizedWidth, resizedHeight);
    const rgba = scratchContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const planeSize = INPUT_SIZE * INPUT_SIZE;
    const tensorData = new Float32Array(planeSize * 3);
    for (let pixel = 0; pixel < planeSize; pixel += 1) {
      const offset = pixel * 4;
      // Match the Python OpenCV BGR-to-NCHW preprocessing used by this project.
      tensorData[pixel] = rgba[offset + 2];
      tensorData[planeSize + pixel] = rgba[offset + 1];
      tensorData[planeSize * 2 + pixel] = rgba[offset];
    }
    return {
      tensor: new window.ort.Tensor("float32", tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]),
      ratio,
      sourceWidth,
      sourceHeight,
    };
  }

  function intersectionOverUnion(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
    const union = areaA + areaB - intersection;
    return union > 0 ? intersection / union : 0;
  }

  function applyNms(candidates) {
    const kept = [];
    for (const classId of [...TARGET_CARS, ...TARGET_TWO_WHEEL]) {
      const sorted = candidates.filter((item) => item.classId === classId).sort((a, b) => b.score - a.score);
      const selected = [];
      while (sorted.length) {
        const best = sorted.shift();
        selected.push(best);
        for (let index = sorted.length - 1; index >= 0; index -= 1) {
          if (intersectionOverUnion(best, sorted[index]) > IOU_THRESHOLD) sorted.splice(index, 1);
        }
      }
      kept.push(...selected);
    }
    return kept;
  }

  function decodeOutput(output, ratio, sourceWidth, sourceHeight, confidence) {
    const data = output.data;
    const dimensions = output.dims;
    const predictionCount = dimensions[dimensions.length - 2];
    const attributeCount = dimensions[dimensions.length - 1];
    if (predictionCount !== gridPoints.length || attributeCount < 5 + CLASS_COUNT) {
      throw new Error(`不支援的 YOLOX 輸出形狀：${dimensions.join("×")}`);
    }

    const candidates = [];
    for (let prediction = 0; prediction < predictionCount; prediction += 1) {
      const base = prediction * attributeCount;
      const objectness = data[base + 4];
      let bestClass = -1;
      let bestScore = 0;
      for (let classId = 0; classId < CLASS_COUNT; classId += 1) {
        const score = objectness * data[base + 5 + classId];
        if (score > bestScore) {
          bestScore = score;
          bestClass = classId;
        }
      }
      if (bestClass < 0 || bestScore < confidence || (!TARGET_CARS.has(bestClass) && !TARGET_TWO_WHEEL.has(bestClass))) continue;

      const grid = gridPoints[prediction];
      const centerX = (data[base] + grid.x) * grid.stride;
      const centerY = (data[base + 1] + grid.y) * grid.stride;
      const width = Math.exp(Math.max(-20, Math.min(20, data[base + 2]))) * grid.stride;
      const height = Math.exp(Math.max(-20, Math.min(20, data[base + 3]))) * grid.stride;
      candidates.push({
        x1: Math.max(0, Math.min(sourceWidth - 1, (centerX - width / 2) / ratio)),
        y1: Math.max(0, Math.min(sourceHeight - 1, (centerY - height / 2) / ratio)),
        x2: Math.max(0, Math.min(sourceWidth - 1, (centerX + width / 2) / ratio)),
        y2: Math.max(0, Math.min(sourceHeight - 1, (centerY + height / 2) / ratio)),
        score: bestScore,
        classId: bestClass,
      });
    }
    return applyNms(candidates);
  }

  async function detect(source) {
    const prepared = preprocess(source);
    const inputName = session.inputNames[0];
    const results = await session.run({ [inputName]: prepared.tensor });
    const output = results[session.outputNames[0]];
    const confidence = Math.max(0, Math.min(1, Number(byId("confidence").value) || 0));
    return decodeOutput(output, prepared.ratio, prepared.sourceWidth, prepared.sourceHeight, confidence);
  }

  function drawPanel(camera, source, detections) {
    const context = camera.context;
    const width = source.videoWidth || source.naturalWidth || source.width;
    const height = source.videoHeight || source.naturalHeight || source.height;
    const scale = Math.min(PANEL_WIDTH / width, PANEL_HEIGHT / height);
    const drawnWidth = Math.round(width * scale);
    const drawnHeight = Math.round(height * scale);
    const offsetX = Math.floor((PANEL_WIDTH - drawnWidth) / 2);
    const offsetY = Math.floor((PANEL_HEIGHT - drawnHeight) / 2);

    context.fillStyle = "#070b0c";
    context.fillRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT);
    context.drawImage(source, offsetX, offsetY, drawnWidth, drawnHeight);
    camera.empty.hidden = true;
    camera.fallback.hidden = true;

    const cars = detections.filter((item) => TARGET_CARS.has(item.classId)).length;
    const twoWheel = detections.filter((item) => TARGET_TWO_WHEEL.has(item.classId)).length;
    for (const item of detections) {
      const color = TARGET_CARS.has(item.classId) ? "#ff4b4f" : "#50e282";
      const x1 = offsetX + item.x1 * scale;
      const y1 = offsetY + item.y1 * scale;
      const x2 = offsetX + item.x2 * scale;
      const y2 = offsetY + item.y2 * scale;
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = `${CLASS_NAMES[item.classId]} ${item.score.toFixed(2)}`;
      context.font = "500 15px 'DM Mono', monospace";
      const labelWidth = context.measureText(label).width + 8;
      const labelY = Math.max(0, y1 - 20);
      context.fillStyle = color;
      context.fillRect(x1, labelY, labelWidth, 19);
      context.fillStyle = "#101515";
      context.fillText(label, x1 + 4, labelY + 14);
    }

    context.fillStyle = "rgba(8, 12, 13, .78)";
    context.fillRect(14, 14, 230, 65);
    context.fillStyle = "#ff7176";
    context.font = "600 22px 'DM Mono', monospace";
    context.fillText(`Cars: ${cars}`, 25, 41);
    context.fillStyle = "#83f09b";
    context.fillText(`Moto/Bike: ${twoWheel}`, 25, 68);
    context.fillStyle = "rgba(8, 12, 13, .72)";
    context.fillRect(PANEL_WIDTH - 195, PANEL_HEIGHT - 30, 181, 20);
    context.fillStyle = "#e3ece8";
    context.font = "500 11px 'DM Mono', monospace";
    context.fillText(camera.title, PANEL_WIDTH - 188, PANEL_HEIGHT - 16);
  }

  function drawPreviewOnly(camera, message) {
    drawIdle(camera, message);
    camera.empty.hidden = true;
    camera.fallback.hidden = false;
    camera.fallback.onload = () => setStatus(camera, "預覽中 · 無推論", "warn");
    camera.fallback.onerror = () => {
      camera.fallback.hidden = true;
      setStatus(camera, "連線受限", "bad");
      camera.sourceLabel.textContent = "瀏覽器無法讀取此串流";
      camera.empty.hidden = false;
    };
  }

  function findMarker(buffer, first, second, from = 0) {
    for (let index = from; index < buffer.length - 1; index += 1) {
      if (buffer[index] === first && buffer[index + 1] === second) return index;
    }
    return -1;
  }

  function appendBytes(left, right) {
    const merged = new Uint8Array(left.length + right.length);
    merged.set(left, 0);
    merged.set(right, left.length);
    return merged;
  }

  async function readMjpeg(camera, url, generation) {
    if (location.protocol === "https:" && new URL(url).protocol === "http:") {
      throw new Error("HTTPS 網頁無法直接讀取 HTTP 串流（Mixed Content）。");
    }

    const controller = new AbortController();
    camera.controller = controller;
    const response = await fetch(url, {
      method: "GET",
      mode: "cors",
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`串流回應 HTTP ${response.status}。`);
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.toLowerCase().includes("multipart/x-mixed-replace")) {
      throw new Error("來源不是 multipart MJPEG 串流。");
    }
    if (!response.body) throw new Error("瀏覽器未提供串流讀取介面。");

    camera.reader = response.body.getReader();
    camera.mode = "mjpeg";
    camera.fallback.hidden = true;
    camera.empty.hidden = true;
    setStatus(camera, "串流中", "live");
    camera.sourceLabel.textContent = "MJPEG · 瀏覽器本機推論";
    let buffer = new Uint8Array(0);

    while (generation === camera.generation) {
      const { value, done } = await camera.reader.read();
      if (done) throw new Error("串流已結束，請重新連線。");
      if (value) buffer = appendBytes(buffer, value);

      while (buffer.length >= 4) {
        const start = findMarker(buffer, 0xff, 0xd8);
        if (start < 0) {
          buffer = buffer.slice(Math.max(0, buffer.length - 1));
          break;
        }
        if (start > 0) buffer = buffer.slice(start);
        const end = findMarker(buffer, 0xff, 0xd9, 2);
        if (end < 0) break;
        camera.latestBlob = new Blob([buffer.slice(0, end + 2)], { type: "image/jpeg" });
        buffer = buffer.slice(end + 2);
      }
      if (buffer.length > 8 * 1024 * 1024) throw new Error("串流影格緩衝區過大，連線已中止。");
    }
  }

  async function connectStream(camera) {
    const url = camera.urlInput.value.trim();
    let parsed;
    try {
      parsed = new URL(url);
      if (!new Set(["http:", "https:"]).has(parsed.protocol)) throw new Error("僅支援 HTTP/HTTPS 網址。");
    } catch (error) {
      setStatus(camera, "網址無效", "bad");
      camera.sourceLabel.textContent = error instanceof Error ? error.message : "請輸入有效的串流網址。";
      return;
    }

    stopCamera(camera, false);
    const generation = camera.generation;
    camera.mode = "connecting";
    camera.sourceLabel.textContent = "檢查串流與 CORS…";
    camera.empty.hidden = true;
    setStatus(camera, "連線中…");

    try {
      await readMjpeg(camera, parsed.href, generation);
    } catch (error) {
      if (generation !== camera.generation || (error instanceof DOMException && error.name === "AbortError")) return;
      const reason = error instanceof Error ? error.message : String(error);
      camera.reader = null;
      camera.controller = null;
      camera.mode = "preview";
      camera.sourceLabel.textContent = reason.includes("CORS") || reason.includes("fetch")
        ? "串流伺服器未允許此網頁讀取畫素；以下僅嘗試預覽。"
        : reason;
      camera.fallback.removeAttribute("crossorigin");
      camera.fallback.src = parsed.href;
      drawPreviewOnly(camera, "串流僅能預覽，無法執行辨識");
      setStatus(camera, "檢查 CORS…", "warn");
    }
  }

  async function loadLocalVideo(camera, file) {
    if (!file) return;
    stopCamera(camera, false);
    const generation = camera.generation;
    camera.objectUrl = URL.createObjectURL(file);
    camera.video.src = camera.objectUrl;
    camera.mode = "video-loading";
    camera.fallback.hidden = true;
    camera.empty.hidden = true;
    camera.sourceLabel.textContent = file.name;
    setStatus(camera, "載入影片…");
    camera.video.onloadeddata = async () => {
      if (generation !== camera.generation) return;
      camera.mode = "video";
      document.querySelector(`.playback-toggle[data-camera="${camera.id}"]`).disabled = false;
      try {
        await camera.video.play();
        setStatus(camera, "影片播放中", "live");
      } catch {
        setStatus(camera, "按播放開始", "warn");
      }
    };
    camera.video.onerror = () => {
      if (generation !== camera.generation) return;
      camera.mode = "idle";
      setStatus(camera, "格式不支援", "bad");
      camera.sourceLabel.textContent = "瀏覽器無法解碼此影片格式。";
      camera.empty.hidden = false;
    };
    camera.video.load();
  }

  async function inferCamera(camera, timestamp) {
    let source = null;
    let bitmap = null;
    if (camera.mode === "video" && camera.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !camera.video.paused) {
      source = camera.video;
    } else if (camera.mode === "mjpeg" && camera.latestBlob) {
      bitmap = await createImageBitmap(camera.latestBlob);
      source = bitmap;
    }
    if (!source) return;

    try {
      if (!session) {
        drawPanel(camera, source, []);
        drawNotReadyOverlay(camera);
        return;
      }
      const started = performance.now();
      const detections = await detect(source);
      camera.lastInferenceMs = performance.now() - started;
      camera.detections = detections;
      camera.lastInferenceAt = timestamp;
      drawPanel(camera, source, detections);
    } catch (error) {
      setStatus(camera, "推論錯誤", "bad");
      camera.sourceLabel.textContent = error instanceof Error ? error.message : String(error);
    } finally {
      if (bitmap) bitmap.close();
    }
  }

  function drawNotReadyOverlay(camera) {
    const context = camera.context;
    context.fillStyle = "rgba(8, 12, 13, .84)";
    context.fillRect(14, 14, 258, 38);
    context.fillStyle = "#f2d78f";
    context.font = "500 14px 'Noto Sans TC', sans-serif";
    context.fillText("請先載入模型以開始辨識", 24, 39);
  }

  function startScheduler() {
    if (schedulerStarted) return;
    schedulerStarted = true;
    const loop = async () => {
      while (schedulerStarted) {
        const now = performance.now();
        for (const camera of cameras) {
          if (now - camera.lastInferenceAt < 200) continue;
          if (camera.mode === "video" || (camera.mode === "mjpeg" && camera.latestBlob)) {
            await inferCamera(camera, now);
            if (camera.mode === "mjpeg") camera.latestBlob = null;
          }
        }
        await new Promise((resolve) => window.setTimeout(resolve, 20));
      }
    };
    loop().catch((error) => {
      setModelState(error instanceof Error ? error.message : String(error), "error");
      schedulerStarted = false;
    });
  }

  function taipeiTimestamp() {
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: "Asia/Taipei",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hourCycle: "h23",
    }).formatToParts(new Date());
    const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
    return `${values.year}${values.month}${values.day}${values.hour}${values.minute}${values.second}`;
  }

  function downloadScreenshot() {
    const output = document.createElement("canvas");
    output.width = PANEL_WIDTH * 2 + 6;
    output.height = PANEL_HEIGHT;
    const context = output.getContext("2d");
    context.fillStyle = "#121a1c";
    context.fillRect(0, 0, output.width, output.height);
    context.drawImage(cameras[0].canvas, 0, 0);
    context.drawImage(cameras[1].canvas, PANEL_WIDTH + 6, 0);
    context.fillStyle = "#526162";
    context.fillRect(PANEL_WIDTH, 0, 6, PANEL_HEIGHT);
    const link = document.createElement("a");
    link.download = `roadwatch_${taipeiTimestamp()}.png`;
    link.href = output.toDataURL("image/png");
    link.click();
  }

  for (const camera of cameras) {
    drawIdle(camera, "連線監視器或載入影片");
    byId(`${camera.id}-file`).addEventListener("change", (event) => {
      loadLocalVideo(camera, event.target.files && event.target.files[0]);
    });
  }

  document.querySelectorAll(".connect-stream").forEach((button) => {
    button.addEventListener("click", () => {
      const camera = cameras.find((item) => item.id === button.dataset.camera);
      connectStream(camera);
    });
  });
  document.querySelectorAll(".playback-toggle").forEach((button) => {
    button.addEventListener("click", async () => {
      const camera = cameras.find((item) => item.id === button.dataset.camera);
      if (camera.video.paused) {
        await camera.video.play();
        setStatus(camera, "影片播放中", "live");
      } else {
        camera.video.pause();
        setStatus(camera, "影片已暫停", "warn");
      }
    });
  });
  byId("load-model").addEventListener("click", loadModel);
  byId("screenshot").addEventListener("click", downloadScreenshot);
  byId("stop-all").addEventListener("click", () => cameras.forEach((camera) => stopCamera(camera)));
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const editing = target instanceof HTMLElement && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
    if (editing || target instanceof HTMLButtonElement) return;
    if (event.key === "Escape") cameras.forEach((camera) => stopCamera(camera));
    if (event.key === "Enter" && session) downloadScreenshot();
  });
})();
