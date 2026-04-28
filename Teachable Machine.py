from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import os
import io
import joblib
import uuid
import inspect
import sys
from typing import Optional, List, Any, Dict
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
WEB_DIR = os.path.join(BASE_DIR, "web")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

app = FastAPI(title="Learnix")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

LEGACY_INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Teachable Machine</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --line: #d8e0ec;
      --text: #1f2a3a;
      --muted: #5a6778;
      --accent: #0f6dff;
      --accent2: #0a56c9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, #e8efff, var(--bg) 40%);
    }
    .wrap {
      max-width: 960px;
      margin: 22px auto;
      padding: 0 14px;
    }
    .title {
      margin: 0 0 6px;
      font-size: 30px;
      font-weight: 700;
    }
    .sub {
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 16px;
    }
    label {
      display: block;
      font-size: 12px;
      margin: 8px 0 4px;
      color: var(--muted);
    }
    input[type="text"], input[type="number"], textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 14px;
      background: #fff;
    }
    textarea {
      min-height: 130px;
      resize: vertical;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      line-height: 1.35;
    }
    input[type="file"] {
      width: 100%;
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .row3 {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      align-items: end;
    }
    button {
      border: 0;
      border-radius: 8px;
      padding: 10px 12px;
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
    }
    button:hover {
      background: var(--accent2);
    }
    .secondary {
      background: #2f3f58;
    }
    .status {
      margin: 10px 0 0;
      font-size: 12px;
      color: var(--muted);
      min-height: 18px;
    }
    pre {
      margin: 0;
      border-radius: 8px;
      background: #0f172a;
      color: #d5e2ff;
      padding: 10px;
      min-height: 220px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.4;
    }
    .toprow {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }
    .pill {
      padding: 5px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
      background: #fff;
    }
    @media (max-width: 540px) {
      .row, .row3 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">Teachable Machine</h1>
    <p class="sub">Upload a CSV, train a model, and predict from new rows.</p>
    <div class="toprow">
      <span id="health" class="pill">Checking server...</span>
      <a href="/docs" class="pill" style="text-decoration:none;">Open API docs</a>
    </div>

    <div class="grid">
      <section class="panel">
        <h2>1) Upload CSV</h2>
        <label for="csvFile">CSV file</label>
        <input id="csvFile" type="file" accept=".csv" />
        <label for="target">Target column name</label>
        <input id="target" type="text" placeholder="Example: label or Exited" />
        <button id="uploadBtn">Upload</button>
        <div id="uploadStatus" class="status"></div>
      </section>

      <section class="panel">
        <h2>2) Train Model</h2>
        <div class="row3">
          <div>
            <label for="trainSize">Train size</label>
            <input id="trainSize" type="number" step="0.01" value="0.8" />
          </div>
          <div>
            <label for="randomState">Random state</label>
            <input id="randomState" type="number" step="1" value="42" />
          </div>
          <div>
            <label for="incremental">Incremental</label>
            <select id="incremental" style="width:100%;border:1px solid var(--line);border-radius:8px;padding:9px 10px;">
              <option value="false" selected>false</option>
              <option value="true">true</option>
            </select>
          </div>
        </div>
        <button id="trainBtn" style="margin-top:10px;">Train</button>
        <div id="trainStatus" class="status"></div>
      </section>

      <section class="panel">
        <h2>3) Predict</h2>
        <label for="predictBody">JSON body</label>
        <textarea id="predictBody">{
  "rows": [
    { "age": 30, "city": "A", "income": 62000 }
  ]
}</textarea>
        <div class="row">
          <button id="predictBtn">Predict</button>
          <button id="modelsBtn" class="secondary">Show Models</button>
        </div>
        <div id="predictStatus" class="status"></div>
      </section>

      <section class="panel">
        <h2>Output</h2>
        <pre id="output">{ "ready": true }</pre>
      </section>
    </div>
  </div>

  <script>
    const output = document.getElementById("output");
    const uploadStatus = document.getElementById("uploadStatus");
    const trainStatus = document.getElementById("trainStatus");
    const predictStatus = document.getElementById("predictStatus");
    const health = document.getElementById("health");

    function show(obj) {
      output.textContent = JSON.stringify(obj, null, 2);
    }

    async function checkHealth() {
      try {
        const res = await fetch("/health");
        const data = await res.json();
        health.textContent = data.status === "ok" ? "Server: healthy" : "Server: unknown";
      } catch (e) {
        health.textContent = "Server: not reachable";
      }
    }

    document.getElementById("uploadBtn").addEventListener("click", async () => {
      uploadStatus.textContent = "";
      const file = document.getElementById("csvFile").files[0];
      const target = document.getElementById("target").value.trim();
      if (!file || !target) {
        uploadStatus.textContent = "Choose a CSV file and target column.";
        return;
      }
      const formData = new FormData();
      formData.append("file", file);
      formData.append("target", target);
      try {
        const res = await fetch("/upload_csv", { method: "POST", body: formData });
        const data = await res.json();
        show(data);
        uploadStatus.textContent = res.ok ? "Upload complete." : "Upload failed.";
      } catch (e) {
        uploadStatus.textContent = "Upload failed.";
      }
    });

    document.getElementById("trainBtn").addEventListener("click", async () => {
      trainStatus.textContent = "";
      const ts = encodeURIComponent(document.getElementById("trainSize").value || "0.8");
      const rs = encodeURIComponent(document.getElementById("randomState").value || "42");
      const inc = encodeURIComponent(document.getElementById("incremental").value || "false");
      const url = "/train?train_size=" + ts + "&random_state=" + rs + "&incremental=" + inc;
      try {
        const res = await fetch(url, { method: "POST" });
        const data = await res.json();
        show(data);
        trainStatus.textContent = res.ok ? "Training complete." : "Training failed.";
      } catch (e) {
        trainStatus.textContent = "Training failed.";
      }
    });

    document.getElementById("predictBtn").addEventListener("click", async () => {
      predictStatus.textContent = "";
      const bodyText = document.getElementById("predictBody").value;
      try {
        JSON.parse(bodyText);
      } catch (e) {
        predictStatus.textContent = "Invalid JSON in Predict body.";
        return;
      }
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: bodyText
        });
        const data = await res.json();
        show(data);
        predictStatus.textContent = res.ok ? "Prediction complete." : "Prediction failed.";
      } catch (e) {
        predictStatus.textContent = "Prediction failed.";
      }
    });

    document.getElementById("modelsBtn").addEventListener("click", async () => {
      predictStatus.textContent = "";
      try {
        const res = await fetch("/models");
        const data = await res.json();
        show(data);
        predictStatus.textContent = res.ok ? "Loaded models." : "Could not load models.";
      } catch (e) {
        predictStatus.textContent = "Could not load models.";
      }
    });

    checkHealth();
  </script>
</body>
</html>
"""

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Teachable Machine Studio</title>
  <style>
    :root {
      --bg-main: #f4f8fb;
      --bg-soft: #eaf2f8;
      --panel: #ffffff;
      --line: #d7e4ed;
      --text: #172130;
      --muted: #5b6b7b;
      --accent: #0d899f;
      --accent-dark: #0b6f82;
      --warn: #cc7648;
      --ok: #19885f;
      --shadow: 0 20px 35px rgba(13, 33, 54, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI Variable", "Trebuchet MS", Tahoma, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 8% 12%, #d7eef8 0, transparent 32%),
        radial-gradient(circle at 92% 86%, #ffe7d9 0, transparent 30%),
        linear-gradient(170deg, var(--bg-main), var(--bg-soft));
    }
    .app {
      max-width: 1180px;
      margin: 14px auto;
      padding: 0 12px 16px;
      animation: rise 350ms ease-out;
    }
    .topbar {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.84);
      backdrop-filter: blur(4px);
      box-shadow: var(--shadow);
      padding: 12px 14px;
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    .title {
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .tools {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      padding: 6px 10px;
    }
    .chip strong { color: var(--text); }
    .chip.link { color: #0f6577; text-decoration: none; }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(0, 0.85fr);
      gap: 12px;
    }
    .stack { display: grid; gap: 12px; min-width: 0; }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .h2 {
      margin: 0;
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 16px;
    }
    .badge {
      width: 24px;
      height: 24px;
      border-radius: 999px;
      background: var(--accent);
      color: #fff;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      font-weight: 700;
    }
    .state {
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 12px;
      padding: 5px 9px;
    }
    .timeline {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }
    .node {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      color: var(--muted);
      font-size: 12px;
      background: #fff;
    }
    .node.active {
      color: #0d6174;
      border-color: #97c1cd;
      background: #f2fbfd;
    }
    .node.done {
      color: #0f704c;
      border-color: #a8dcca;
      background: #eefbf6;
    }

    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin: 8px 0 5px;
    }
    input[type="text"], input[type="number"], select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 11px;
      font-size: 14px;
      color: var(--text);
      background: #fff;
      outline: none;
    }
    input:focus, select:focus, textarea:focus {
      border-color: #98bfca;
      box-shadow: 0 0 0 3px rgba(13, 137, 159, 0.12);
    }
    textarea {
      min-height: 170px;
      resize: vertical;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      line-height: 1.4;
    }
    .drop {
      border: 1px dashed #98becc;
      border-radius: 8px;
      background: linear-gradient(180deg, #f9fdff, #f2f9fc);
      padding: 16px;
      text-align: center;
      cursor: pointer;
      transition: 150ms ease;
    }
    .drop:hover {
      border-color: var(--accent);
      background: #eef9fc;
    }
    .file { margin-top: 7px; font-size: 12px; color: var(--muted); min-height: 16px; }
    .hidden { display: none !important; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .tools-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }

    .slider-line { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; }
    input[type="range"] { width: 100%; accent-color: var(--accent); margin: 6px 0 2px; }

    button {
      border: 0;
      border-radius: 8px;
      padding: 10px 12px;
      font-weight: 600;
      color: #fff;
      background: var(--accent);
      cursor: pointer;
      transition: transform 110ms ease, background 160ms ease;
    }
    button:hover { background: var(--accent-dark); transform: translateY(-1px); }
    button:disabled { background: #a8c5cc; transform: none; cursor: wait; }
    button.secondary { background: #34495c; }
    button.warn { background: var(--warn); }
    button.warn:hover { background: #b9653a; }

    .status { margin-top: 10px; min-height: 16px; font-size: 12px; color: var(--muted); }
    .status.ok { color: var(--ok); }
    .status.error { color: #b44444; }

    .columns {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-top: 8px;
      min-height: 22px;
    }
    .pill {
      border: 1px solid #d5e2ea;
      border-radius: 999px;
      background: #f9fcff;
      color: #495b71;
      font-size: 11px;
      padding: 3px 8px;
      max-width: 210px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .table-wrap {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: auto;
      max-height: 220px;
      background: #fff;
    }
    table {
      width: 100%;
      min-width: 500px;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid #edf2f6;
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
      max-width: 180px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: #f4fafc;
      color: #38516b;
      z-index: 1;
    }

    .summary {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .sbox {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fcfeff;
      padding: 9px;
    }
    .slabel { display: block; color: var(--muted); font-size: 11px; margin-bottom: 3px; }
    .sval { font-size: 15px; font-weight: 700; overflow-wrap: anywhere; }

    pre {
      margin: 0;
      min-height: 360px;
      border-radius: 8px;
      background: #121b2b;
      color: #dbe8ff;
      padding: 12px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.4;
    }
    .small { margin: 10px 0 0; font-size: 11px; color: var(--muted); }

    .toast {
      position: fixed;
      right: 14px;
      bottom: 14px;
      max-width: min(94vw, 340px);
      border: 1px solid rgba(214, 232, 247, 0.2);
      border-radius: 8px;
      background: #152235;
      color: #e7f1ff;
      padding: 9px 12px;
      font-size: 12px;
      box-shadow: var(--shadow);
      transform: translateY(10px);
      opacity: 0;
      pointer-events: none;
      transition: opacity 140ms ease, transform 140ms ease;
    }
    .toast.show { opacity: 1; transform: translateY(0); }

    @keyframes rise {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 1040px) {
      .layout { grid-template-columns: 1fr; }
      pre { min-height: 260px; }
    }
    @media (max-width: 660px) {
      .row, .summary { grid-template-columns: 1fr; }
      .title { font-size: 22px; }
      .app { margin-top: 10px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div>
        <h1 class="title">Teachable Machine Studio</h1>
        <p class="subtitle">Upload CSV, train, and predict with live guidance.</p>
      </div>
      <div class="tools">
        <span id="healthChip" class="chip">Server: checking</span>
        <span class="chip">Host: <strong id="hostText">local</strong></span>
        <a href="/docs" class="chip link">API Docs</a>
      </div>
    </header>

    <div class="layout">
      <div class="stack">
        <section class="panel">
          <div class="head">
            <h2 class="h2"><span class="badge">1</span>Upload Dataset</h2>
            <span id="uploadState" class="state">Waiting</span>
          </div>
          <div class="timeline">
            <span id="node1" class="node active">Upload</span>
            <span id="node2" class="node">Train</span>
            <span id="node3" class="node">Predict</span>
          </div>

          <label for="csvFile">CSV file</label>
          <label class="drop" for="csvFile">
            <input id="csvFile" class="hidden" type="file" accept=".csv" />
            <div><strong>Select a CSV</strong> or drag and drop here</div>
            <div id="fileName" class="file">No file selected</div>
          </label>

          <div class="row">
            <div>
              <label for="targetInput">Target column</label>
              <input id="targetInput" type="text" list="columnList" placeholder="Choose or type column name" />
              <datalist id="columnList"></datalist>
            </div>
            <div>
              <label for="rowCountInput">Preview rows</label>
              <input id="rowCountInput" type="number" value="6" min="2" max="12" />
            </div>
          </div>

          <div class="tools-row">
            <button id="uploadBtn">Upload To Server</button>
          </div>
          <div id="uploadStatus" class="status"></div>
          <div id="columnPills" class="columns"></div>
          <div id="previewWrap" class="table-wrap hidden">
            <table id="previewTable"></table>
          </div>
        </section>

        <section class="panel">
          <div class="head">
            <h2 class="h2"><span class="badge">2</span>Train Model</h2>
            <span id="trainState" class="state">Locked</span>
          </div>
          <div class="slider-line"><span>Train size</span><strong id="trainSizeValue">0.80</strong></div>
          <input id="trainSize" type="range" min="0.5" max="0.95" step="0.01" value="0.8" />
          <div class="row">
            <div>
              <label for="randomState">Random state</label>
              <input id="randomState" type="number" value="42" step="1" />
            </div>
            <div>
              <label for="incremental">Incremental mode</label>
              <select id="incremental">
                <option value="false" selected>false</option>
                <option value="true">true</option>
              </select>
            </div>
          </div>
          <div class="tools-row"><button id="trainBtn">Train Model</button></div>
          <div id="trainStatus" class="status"></div>
        </section>

        <section class="panel">
          <div class="head">
            <h2 class="h2"><span class="badge">3</span>Predict</h2>
            <span id="predictState" class="state">Locked</span>
          </div>
          <label for="predictBody">Prediction request JSON</label>
          <textarea id="predictBody">{
  "rows": [
    { "age": 30, "city": "A", "income": 62000 }
  ]
}</textarea>
          <div class="tools-row">
            <button id="templateBtn" class="secondary">Use First Row Template</button>
            <button id="predictBtn">Predict Now</button>
            <button id="modelsBtn" class="warn">Show Saved Models</button>
          </div>
          <div id="predictStatus" class="status"></div>
        </section>
      </div>

      <div class="stack">
        <section class="panel">
          <h2 class="h2" style="margin:0 0 10px;"><span class="badge" style="background:#34495c;">i</span>Session Summary</h2>
          <div class="summary">
            <div class="sbox"><span class="slabel">File</span><span id="sumFile" class="sval">-</span></div>
            <div class="sbox"><span class="slabel">Rows</span><span id="sumRows" class="sval">-</span></div>
            <div class="sbox"><span class="slabel">Target</span><span id="sumTarget" class="sval">-</span></div>
            <div class="sbox"><span class="slabel">Task</span><span id="sumTask" class="sval">-</span></div>
            <div class="sbox"><span class="slabel">Metric</span><span id="sumMetric" class="sval">-</span></div>
            <div class="sbox"><span class="slabel">Model ID</span><span id="sumModelId" class="sval">-</span></div>
          </div>
          <p class="small">Tip: target column is the output you want the model to learn.</p>
        </section>

        <section class="panel">
          <h2 class="h2" style="margin:0 0 10px;"><span class="badge" style="background:#34495c;">{ }</span>Live Output</h2>
          <pre id="output">{ "ready": true }</pre>
        </section>
      </div>
    </div>
  </div>

  <div id="toast" class="toast"></div>

  <script>
    const state = {
      columns: [],
      previewRows: [],
      uploaded: false,
      trained: false,
      lastModelId: null,
    };

    const fileInput = document.getElementById("csvFile");
    const fileName = document.getElementById("fileName");
    const targetInput = document.getElementById("targetInput");
    const rowCountInput = document.getElementById("rowCountInput");
    const columnList = document.getElementById("columnList");
    const columnPills = document.getElementById("columnPills");
    const previewWrap = document.getElementById("previewWrap");
    const previewTable = document.getElementById("previewTable");
    const output = document.getElementById("output");
    const toast = document.getElementById("toast");

    const uploadStatus = document.getElementById("uploadStatus");
    const trainStatus = document.getElementById("trainStatus");
    const predictStatus = document.getElementById("predictStatus");
    const uploadState = document.getElementById("uploadState");
    const trainState = document.getElementById("trainState");
    const predictState = document.getElementById("predictState");
    const node1 = document.getElementById("node1");
    const node2 = document.getElementById("node2");
    const node3 = document.getElementById("node3");
    const healthChip = document.getElementById("healthChip");
    const hostText = document.getElementById("hostText");

    const sumFile = document.getElementById("sumFile");
    const sumRows = document.getElementById("sumRows");
    const sumTarget = document.getElementById("sumTarget");
    const sumTask = document.getElementById("sumTask");
    const sumMetric = document.getElementById("sumMetric");
    const sumModelId = document.getElementById("sumModelId");

    const trainSize = document.getElementById("trainSize");
    const trainSizeValue = document.getElementById("trainSizeValue");
    const randomState = document.getElementById("randomState");
    const incremental = document.getElementById("incremental");
    const predictBody = document.getElementById("predictBody");

    const uploadBtn = document.getElementById("uploadBtn");
    const trainBtn = document.getElementById("trainBtn");
    const predictBtn = document.getElementById("predictBtn");
    const modelsBtn = document.getElementById("modelsBtn");
    const templateBtn = document.getElementById("templateBtn");

    function show(data) {
      output.textContent = JSON.stringify(data, null, 2);
    }

    function setStatus(el, text, kind) {
      el.textContent = text || "";
      el.classList.remove("ok", "error");
      if (kind) el.classList.add(kind);
    }

    function setBusy(btn, busy, busyText, idleText) {
      btn.disabled = busy;
      btn.textContent = busy ? busyText : idleText;
    }

    function showToast(text) {
      toast.textContent = text;
      toast.classList.add("show");
      clearTimeout(showToast.t);
      showToast.t = setTimeout(() => toast.classList.remove("show"), 1800);
    }

    function splitCsvLine(line) {
      const out = [];
      let cur = "";
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
          const next = line[i + 1];
          if (inQuotes && next === '"') {
            cur += '"';
            i++;
          } else {
            inQuotes = !inQuotes;
          }
        } else if (ch === "," && !inQuotes) {
          out.push(cur.trim());
          cur = "";
        } else {
          cur += ch;
        }
      }
      out.push(cur.trim());
      return out;
    }

    function parseCsv(text, maxRows) {
      const lines = text.split(/\\r?\\n/).filter((l) => l.trim().length > 0);
      if (!lines.length) return { headers: [], rows: [] };
      const headers = splitCsvLine(lines[0]).map((x) => x.replace(/^"|"$/g, ""));
      const rows = [];
      for (let i = 1; i < lines.length && rows.length < maxRows; i++) {
        const cells = splitCsvLine(lines[i]).map((x) => x.replace(/^"|"$/g, ""));
        const row = {};
        for (let c = 0; c < headers.length; c++) row[headers[c]] = cells[c] ?? "";
        rows.push(row);
      }
      return { headers, rows };
    }

    function renderColumns(cols) {
      columnList.innerHTML = "";
      columnPills.innerHTML = "";
      cols.forEach((col) => {
        const op = document.createElement("option");
        op.value = col;
        columnList.appendChild(op);

        const pill = document.createElement("span");
        pill.className = "pill";
        pill.textContent = col;
        columnPills.appendChild(pill);
      });
    }

    function renderPreview(headers, rows) {
      if (!headers.length || !rows.length) {
        previewWrap.classList.add("hidden");
        previewTable.innerHTML = "";
        return;
      }
      let html = "<thead><tr>";
      headers.forEach((h) => { html += "<th>" + h + "</th>"; });
      html += "</tr></thead><tbody>";
      rows.forEach((row) => {
        html += "<tr>";
        headers.forEach((h) => { html += "<td>" + (row[h] ?? "") + "</td>"; });
        html += "</tr>";
      });
      html += "</tbody>";
      previewTable.innerHTML = html;
      previewWrap.classList.remove("hidden");
    }

    function updateFlow() {
      node1.className = "node active";
      node2.className = state.uploaded ? "node active" : "node";
      node3.className = state.trained ? "node active" : "node";
      if (state.uploaded) node1.className = "node done";
      if (state.trained) {
        node2.className = "node done";
        node3.className = "node done";
      }
      uploadState.textContent = state.uploaded ? "Uploaded" : "Waiting";
      trainState.textContent = state.uploaded ? (state.trained ? "Trained" : "Ready") : "Locked";
      predictState.textContent = state.trained ? "Ready" : "Locked";
    }

    function buildTemplateFromFirstRow() {
      if (!state.previewRows.length || !state.columns.length) {
        showToast("Upload a CSV first");
        return;
      }
      const target = targetInput.value.trim();
      const src = state.previewRows[0];
      const row = {};
      state.columns.forEach((col) => {
        if (col === target) return;
        const raw = src[col];
        const num = Number(raw);
        row[col] = raw !== "" && Number.isFinite(num) ? num : raw;
      });
      predictBody.value = JSON.stringify({ rows: [row] }, null, 2);
      showToast("Prediction template updated");
    }

    async function checkHealth() {
      hostText.textContent = window.location.host;
      try {
        const res = await fetch("/health");
        const data = await res.json();
        if (res.ok && data.status === "ok") healthChip.innerHTML = "Server: <strong>healthy</strong>";
        else healthChip.innerHTML = "Server: <strong>issue</strong>";
      } catch (_) {
        healthChip.innerHTML = "Server: <strong>offline</strong>";
      }
    }

    fileInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      fileName.textContent = file.name;
      sumFile.textContent = file.name;
      const maxRows = Math.max(2, Math.min(12, Number(rowCountInput.value || 6)));
      const text = await file.text();
      const parsed = parseCsv(text, maxRows);
      state.columns = parsed.headers;
      state.previewRows = parsed.rows;
      renderColumns(parsed.headers);
      renderPreview(parsed.headers, parsed.rows);
      if (parsed.headers.length) targetInput.value = parsed.headers[parsed.headers.length - 1];
      buildTemplateFromFirstRow();
      showToast("CSV preview loaded");
    });

    rowCountInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      const maxRows = Math.max(2, Math.min(12, Number(rowCountInput.value || 6)));
      const parsed = parseCsv(await file.text(), maxRows);
      state.columns = parsed.headers;
      state.previewRows = parsed.rows;
      renderPreview(parsed.headers, parsed.rows);
    });

    trainSize.addEventListener("input", () => {
      trainSizeValue.textContent = Number(trainSize.value).toFixed(2);
    });

    templateBtn.addEventListener("click", buildTemplateFromFirstRow);

    uploadBtn.addEventListener("click", async () => {
      setStatus(uploadStatus, "", null);
      const file = fileInput.files[0];
      const target = targetInput.value.trim();
      if (!file || !target) {
        setStatus(uploadStatus, "Select CSV and target column first.", "error");
        return;
      }

      setBusy(uploadBtn, true, "Uploading...", "Upload To Server");
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("target", target);
        const res = await fetch("/upload_csv", { method: "POST", body: formData });
        const data = await res.json();
        show(data);
        if (!res.ok) {
          setStatus(uploadStatus, data.detail || "Upload failed.", "error");
          return;
        }
        state.uploaded = true;
        state.trained = false;
        sumRows.textContent = data.rows ?? "-";
        sumTarget.textContent = data.target ?? target;
        sumTask.textContent = data.inferred_task ?? "-";
        sumMetric.textContent = "-";
        sumModelId.textContent = "-";
        setStatus(uploadStatus, "Upload successful. Ready to train.", "ok");
        updateFlow();
        showToast("Dataset uploaded");
      } catch (_) {
        setStatus(uploadStatus, "Upload failed. Check server.", "error");
      } finally {
        setBusy(uploadBtn, false, "Uploading...", "Upload To Server");
      }
    });

    trainBtn.addEventListener("click", async () => {
      setStatus(trainStatus, "", null);
      if (!state.uploaded) {
        setStatus(trainStatus, "Upload a dataset first.", "error");
        return;
      }

      const params = new URLSearchParams({
        train_size: String(trainSize.value || "0.8"),
        random_state: String(randomState.value || "42"),
        incremental: String(incremental.value || "false"),
      });

      setBusy(trainBtn, true, "Training...", "Train Model");
      try {
        const res = await fetch("/train?" + params.toString(), { method: "POST" });
        const data = await res.json();
        show(data);
        if (!res.ok) {
          setStatus(trainStatus, data.detail || "Training failed.", "error");
          return;
        }
        state.trained = true;
        state.lastModelId = data.modelId || null;
        sumModelId.textContent = state.lastModelId || "-";
        if (data.metadata?.task) sumTask.textContent = data.metadata.task;
        const m = data.metadata?.metric;
        const n = data.metadata?.metric_name || "metric";
        sumMetric.textContent = m === undefined ? "-" : n + ": " + Number(m).toFixed(4);
        setStatus(trainStatus, "Training complete. Predict is unlocked.", "ok");
        updateFlow();
        showToast("Model trained");
      } catch (_) {
        setStatus(trainStatus, "Training failed. Try again.", "error");
      } finally {
        setBusy(trainBtn, false, "Training...", "Train Model");
      }
    });

    predictBtn.addEventListener("click", async () => {
      setStatus(predictStatus, "", null);
      if (!state.trained) {
        setStatus(predictStatus, "Train model before predict.", "error");
        return;
      }
      let parsed;
      try {
        parsed = JSON.parse(predictBody.value);
      } catch (_) {
        setStatus(predictStatus, "Predict JSON is invalid.", "error");
        return;
      }
      setBusy(predictBtn, true, "Predicting...", "Predict Now");
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(parsed),
        });
        const data = await res.json();
        show(data);
        if (!res.ok) {
          setStatus(predictStatus, data.detail || "Prediction failed.", "error");
          return;
        }
        setStatus(predictStatus, "Prediction complete.", "ok");
        showToast("Prediction ready");
      } catch (_) {
        setStatus(predictStatus, "Prediction failed. Check input.", "error");
      } finally {
        setBusy(predictBtn, false, "Predicting...", "Predict Now");
      }
    });

    modelsBtn.addEventListener("click", async () => {
      setBusy(modelsBtn, true, "Loading...", "Show Saved Models");
      try {
        const res = await fetch("/models");
        const data = await res.json();
        show(data);
        if (!res.ok) setStatus(predictStatus, "Could not load models.", "error");
        else setStatus(predictStatus, "Loaded models list.", "ok");
      } catch (_) {
        setStatus(predictStatus, "Could not load models.", "error");
      } finally {
        setBusy(modelsBtn, false, "Loading...", "Show Saved Models");
      }
    });

    updateFlow();
    checkHealth();
  </script>
</body>
</html>
"""

# Use camelCase consistently to avoid Pydantic v2 protected namespace conflicts
STORE: Dict[str, Any] = {
    "last_dataset": None,
    "last_target": None,
    "last_task": None,
    "last_modelId": None,
}

# ---------- Schemas ----------

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]
    modelId: Optional[str] = None  # renamed from model_id

# ---------- Helpers ----------

def infer_task(y: pd.Series) -> str:
    if y.dtype.name in ["object", "category"]:
        return "classification"
    try:
        unique = y.nunique()
    except Exception:
        unique = len(y.unique())
    return "classification" if unique <= 20 else "regression"

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    ohe_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(**ohe_kwargs))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", cat_transformer, cat_cols)
        ],
        remainder="drop"
    )

def choose_model(task: str, incremental: bool = False):
        if task == "classification":
            return SGDClassifier(max_iter=1000, tol=1e-3) if incremental else RandomForestClassifier(n_estimators=100)
        else:
            return SGDRegressor(max_iter=1000, tol=1e-3) if incremental else RandomForestRegressor(n_estimators=100)

def save_artifact(model_pipeline: Pipeline, metadata: dict) -> str:
    modelId = str(uuid.uuid4())
    path = os.path.join(ARTIFACTS_DIR, f"modelId_{modelId}.joblib")
    joblib.dump({"pipeline": model_pipeline, "metadata": metadata}, path)
    return modelId

def _artifact_path_from_id(modelId: str) -> str:
    new_path = os.path.join(ARTIFACTS_DIR, f"modelId_{modelId}.joblib")
    if os.path.exists(new_path):
        return new_path
    legacy_path = os.path.join(ARTIFACTS_DIR, f"model_{modelId}.joblib")
    return legacy_path

def load_artifact(modelId: str) -> dict:
    path = _artifact_path_from_id(modelId)
    if not os.path.exists(path):
        raise FileNotFoundError("Model not found")
    return joblib.load(path)

# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(content=INDEX_HTML)

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), target: str = Form(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in CSV")
    STORE["last_dataset"] = df
    STORE["last_target"] = target
    STORE["last_task"] = infer_task(df[target])
    return {"status": "ok", "rows": len(df), "target": target, "inferred_task": STORE["last_task"]}

@app.post("/train")
def train(train_size: float = 0.8, random_state: int = 42, incremental: bool = False):
    df = STORE.get("last_dataset")
    target = STORE.get("last_target")
    if df is None or target is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Use /upload_csv first.")

    X = df.drop(columns=[target])
    y = df[target]

    task = infer_task(y)
    STORE["last_task"] = task

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=random_state)

    preprocessor = build_preprocessor(X_train)
    model = choose_model(task, incremental=incremental)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    if incremental and hasattr(model, "partial_fit"):
        if task == "classification":
            classes = y_train.unique()
            chunk_size = max(1, int(len(X_train) / 5))
            for i in range(0, len(X_train), chunk_size):
                X_chunk = X_train.iloc[i:i+chunk_size]
                y_chunk = y_train.iloc[i:i+chunk_size]
                X_proc = preprocessor.fit_transform(X_chunk)
                model.partial_fit(X_proc, y_chunk, classes=classes)
        else:
            X_proc = preprocessor.fit_transform(X_train)
            model.partial_fit(X_proc, y_train)
    else:
        pipeline.fit(X_train, y_train)

    if incremental and hasattr(model, "predict"):
        y_pred = model.predict(preprocessor.transform(X_val))
    else:
        y_pred = pipeline.predict(X_val)

    metric = accuracy_score(y_val, y_pred) if task == "classification" else mean_squared_error(y_val, y_pred, squared=False)
    metadata = {"task": task, "metric": float(metric), "metric_name": "accuracy" if task == "classification" else "rmse", "n_rows": len(df)}

    if incremental:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    modelId = save_artifact(pipeline, metadata)
    STORE["last_modelId"] = modelId

    return {"status": "trained", "modelId": modelId, "metadata": metadata}

@app.post("/predict")
def predict(req: PredictRequest):
    modelId = req.modelId or STORE.get("last_modelId")
    if modelId is None:
        raise HTTPException(status_code=400, detail="No model specified and no trained model available.")

    try:
        data = load_artifact(modelId)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    pipeline_obj = data.get("pipeline")
    rows_df = pd.DataFrame(req.rows)

    try:
        preds = pipeline_obj.predict(rows_df)
    except:
        try:
            pre = pipeline_obj.named_steps["preprocessor"]
            mdl = pipeline_obj.named_steps["model"]
            preds = mdl.predict(pre.transform(rows_df))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return {"predictions": preds.tolist(), "modelId": modelId}

@app.get("/models")
def list_models():
    models = []
    for fname in os.listdir(ARTIFACTS_DIR):
        if fname.endswith(".joblib"):
            metadata = joblib.load(os.path.join(ARTIFACTS_DIR, fname)).get("metadata", {})
            modelId = fname.split("modelId_")[-1].split(".joblib")[0] if fname.startswith("modelId_") else fname.split("model_")[-1].split(".joblib")[0]
            models.append({"modelId": modelId, "metadata": metadata})
    return {"models": models}

@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    port = int(os.getenv("TEACHABLE_MACHINE_PORT", port))

    uvicorn.run(app, host="127.0.0.1", port=port)
