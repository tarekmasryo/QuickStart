
# QuickStart is a **Gradio** app that turns any Hugging Face **URL** or **Repo ID** into reliable, copy-ready **first-run artifacts** (best-effort).

[![UI](https://img.shields.io/badge/UI-Gradio-FF7A18)](https://www.gradio.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-orange)
[![HF Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/tarekmasryo/QuickStart)

**Live demo:** https://huggingface.co/spaces/tarekmasryo/QuickStart

---

## Preview
![QuickStart UI](assets/example.png)

---

## What you get

Given a repo (**Model / Dataset / Space**), QuickStart generates:

- **Run snippet** *(best-effort)*
- **Download recipes**
  - Python: `snapshot_download()`
  - CLI: `huggingface-cli download`
- **Files view** + lightweight **risk hints** *(filename-based only)*
- **Exportable zip** with a minimal runnable scaffold

---

## Supported inputs

**Repo ID**
```text
<owner>/<repo>
```

**URLs**
```text
https://huggingface.co/<owner>/<repo>
https://huggingface.co/datasets/<owner>/<repo>
https://huggingface.co/spaces/<owner>/<repo>
```

Also accepted:
```text
datasets/<owner>/<repo>
spaces/<owner>/<repo>
```

---

## Run locally

```bash
git clone https://github.com/tarekmasryo/quickstart.git
cd quickstart

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

python app.py
```

---

## Authentication (private / gated repos)

`HF_TOKEN` is **optional**. You only need it if the target repo is **private** or **gated**.

**Option A — Environment variable**

Windows (PowerShell):
```bash
setx HF_TOKEN "YOUR_TOKEN"
```
Restart the terminal.

macOS/Linux:
```bash
export HF_TOKEN="YOUR_TOKEN"
```

**Option B — CLI login**
```bash
huggingface-cli login
```

**On Hugging Face Spaces**
- Space **Settings → Secrets**
- Add: `HF_TOKEN` = your token

---

## Export output (contract)

The exported zip is a minimal runnable scaffold (best-effort generated):
```text
run.py
download.py
requirements.txt
.env.example
README.md
```

---

## Risk hints (important)

Risk hints are **filename-based only**:
- ✅ Flags names like: `.env`, `token`, `api_key`, `credentials`, private keys
- ✅ Highlights common ML artifacts by extension (e.g., `.safetensors`, `.bin`, `.onnx`, `.gguf`)
- ❌ Does **not** scan file contents
- ❌ Not a security/compliance audit

Use it as a **signal**, not a verdict.

---

## Known limitations

- Snippets are **best-effort** and depend on Hub metadata.
- Files view is limited and may be incomplete for some repos.
- No content scanning (by design).

---

## License

Apache-2.0

**Author:** Tarek Masryo
