# ⚡ QuickStart — Hugging Face Repo Quickstart Kit

QuickStart is a clean **Gradio** app that turns Hugging Face **repo URLs** or **Repo IDs** into copy-ready first-run artifacts: install commands, runnable Python snippets, download recipes, file previews, lightweight risk hints, and an exportable scaffold ZIP.

[![UI](https://img.shields.io/badge/UI-Gradio-FF7A18)](https://www.gradio.app/)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-orange)
[![CI](https://github.com/tarekmasryo/QuickStart/actions/workflows/ci.yml/badge.svg)](https://github.com/tarekmasryo/QuickStart/actions/workflows/ci.yml)
[![Live Space](https://img.shields.io/badge/Live-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/tarekmasryo/QuickStart)

**Live demo:** https://huggingface.co/spaces/tarekmasryo/QuickStart

---

## 🖼️ Preview

![QuickStart UI](assets/Example.png)

---

## 🎯 Why QuickStart?

Hugging Face repositories are not all started the same way. Models, datasets, and Spaces often require different install paths, download commands, runtime assumptions, and safety checks.

QuickStart standardizes the **first 5 minutes** of repo exploration into a repeatable workflow.

---

## ✨ Features

- 🧭 **Type-aware parsing** for models, datasets, and Spaces
- 🔗 Accepts plain repo IDs and Hugging Face repo URLs
- 🧪 Generates a **best-effort runnable Python snippet** from repo metadata
- 📦 Generates full snapshot download code using `snapshot_download()`
- 🖥️ Generates modern CLI download commands using `hf download`
- 📁 Shows a limited file table with quick filtering
- 🛡️ Flags filename-based risk hints for secrets, keys, and common model artifacts
- 🧰 Exports a minimal scaffold ZIP with `run.py`, `download.py`, `requirements.txt`, `.env.example`, and `README.md`
- 🔒 Keeps server-side token usage disabled by default

---

## ✅ Supported inputs

### Repo ID

```text
<repo>
<owner>/<repo>
```

### URLs

```text
https://huggingface.co/<repo>
https://huggingface.co/<owner>/<repo>
https://hf.co/<owner>/<repo>
https://huggingface.co/<owner>/<repo>/tree/main
https://huggingface.co/<owner>/<repo>/blob/main/file.py
https://huggingface.co/<owner>/<repo>/discussions/1
https://huggingface.co/datasets/<owner>/<repo>
https://huggingface.co/datasets/<owner>/<repo>/viewer/default/train
https://huggingface.co/spaces/<owner>/<repo>
```

### Short typed paths

```text
datasets/<owner>/<repo>
spaces/<owner>/<repo>
```

---

## 🚀 Run locally

```bash
git clone https://github.com/tarekmasryo/QuickStart.git
cd QuickStart

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
python app.py
```

---

## 🔐 Private / gated repos

Public repos work without a token.

Private or gated repos require an explicitly enabled, owner-scoped server token:

```bash
ALLOW_SERVER_TOKEN=1
HF_TOKEN=YOUR_TOKEN
TOKEN_ALLOWED_OWNERS=owner1,owner2
```

`TOKEN_ALLOWED_OWNERS` is required when server-token mode is enabled. This fail-closed behavior prevents an accidentally enabled token from being used against arbitrary repos.

On Hugging Face Spaces:

1. Open **Space Settings → Secrets**
2. Add `HF_TOKEN`
3. Add `ALLOW_SERVER_TOKEN=1` only if you intentionally want server-token access
4. Add `TOKEN_ALLOWED_OWNERS` for safer scoping

---

## 📦 Export ZIP contract

The exported ZIP contains:

```text
run.py
download.py
requirements.txt
.env.example
README.md
```

The scaffold is intentionally small. It is a first-run starter, not a full production project generator.

---

## 🛡️ Risk hints — important limitation

QuickStart provides **filename-based hints only**.

It can flag names such as:

- `.env`
- `token`
- `api_key`
- `credentials`
- `.pem`
- `.p12`
- `id_rsa`
- large model artifacts such as `.gguf`, `.onnx`, `.safetensors`, and `.bin`

It does **not** scan file contents, validate licenses, inspect model behavior, or perform a security/compliance audit.

---

## 🧱 Project structure

```text
.
├── app.py                  # Gradio UI
├── quickstart_core.py       # Parsing, metadata, snippet generation, export logic
├── assets/
│   └── Example.png
├── tests/
│   └── test_quickstart_core.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧪 Development checks

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m compileall .
python -m pytest -q
```

---

## 📌 Notes

- Generated snippets are **best-effort** and depend on Hub metadata.
- Some repositories need extra files, custom code, hardware, or manual setup.
- Large models may require GPU/VRAM or selective downloads.
- Gated/private repositories require proper access permissions.

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE).

**Author:** Tarek Masryo
