import os
import re
import html
import textwrap
import tempfile
import zipfile
import inspect
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, List

import gradio as gr
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


RE_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def norm_type(x: str) -> str:
    x = (x or "model").strip().lower()
    return x if x in {"model", "dataset", "space"} else "model"


def norm_id(x: str) -> str:
    return (x or "").strip().strip("/")


def is_valid_repo_id(repo_id: str) -> bool:
    return bool(RE_REPO_ID.match(norm_id(repo_id)))


def human_bytes(n: Optional[int]) -> str:
    if not isinstance(n, int) or n <= 0:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f} {units[i]}"


def safe_str(x: Any, max_chars: int = 500) -> str:
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s


def call_with_optional_kwargs(fn, *args, **kwargs):
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return fn(*args, **safe_kwargs)
    except Exception:
        return fn(*args)


def parse_hf_input(user_input: str) -> Tuple[str, str]:
    s = (user_input or "").strip()
    if not s:
        return "model", ""

    if "huggingface.co" in s:
        m = re.search(r"huggingface\.co/(datasets|spaces)/([^?#]+)", s)
        if m:
            rt = "dataset" if m.group(1) == "datasets" else "space"
            path = m.group(2).strip("/")
            path = re.split(r"/(tree|blob|resolve|raw)/", path)[0].strip("/")
            return rt, path

        m2 = re.search(r"huggingface\.co/([^?#]+)", s)
        if m2:
            path = m2.group(1).strip("/")
            path = re.split(r"/(tree|blob|resolve|raw)/", path)[0].strip("/")
            return "model", path

    if s.startswith("datasets/"):
        return "dataset", s.replace("datasets/", "", 1).strip("/")
    if s.startswith("spaces/"):
        return "space", s.replace("spaces/", "", 1).strip("/")

    return "model", s.strip("/")


def hf_url(repo_type: str, repo_id: str) -> str:
    rt = norm_type(repo_type)
    rid = norm_id(repo_id)
    if rt == "dataset":
        return f"https://huggingface.co/datasets/{rid}"
    if rt == "space":
        return f"https://huggingface.co/spaces/{rid}"
    return f"https://huggingface.co/{rid}"


def safe_hf_error(e: HfHubHTTPError) -> str:
    status = getattr(getattr(e, "response", None), "status_code", "N/A")
    msg = getattr(e, "server_message", None) or str(e)
    return f"Hugging Face Error: {status} - {safe_str(msg, 500)}"


def extract_file_entries(info_obj) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    siblings = getattr(info_obj, "siblings", None) or []
    for s in siblings:
        name = getattr(s, "rfilename", None) or getattr(s, "path", None) or None
        if not name:
            continue
        size = getattr(s, "size", None)
        if size is None:
            lfs = getattr(s, "lfs", None)
            size = getattr(lfs, "size", None) if lfs is not None else None
        out.append({"path": str(name), "size": int(size) if isinstance(size, int) else None})
    return out


def files_risk_report(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    paths = [f.get("path", "") for f in files if f.get("path")]
    total_known = sum(int(f["size"]) for f in files if isinstance(f.get("size"), int))

    has_gguf = any(p.lower().endswith(".gguf") for p in paths)
    has_onnx = any(p.lower().endswith(".onnx") for p in paths)
    has_safetensors = any(p.lower().endswith(".safetensors") for p in paths)
    has_bin = any(p.lower().endswith(".bin") for p in paths)

    suspicious_names = []
    suspicious_patterns = [
        r"\.env$",
        r"secrets?",
        r"token",
        r"api[_-]?key",
        r"credentials?",
        r"id_rsa",
        r"\.pem$",
        r"\.p12$",
        r"\.kdbx$",
    ]
    for p in paths:
        pl = p.lower()
        if any(re.search(rx, pl) for rx in suspicious_patterns):
            suspicious_names.append(p)

    return {
        "files_count": len(paths),
        "total_size_known": total_known if total_known > 0 else None,
        "has_gguf": has_gguf,
        "has_onnx": has_onnx,
        "has_safetensors": has_safetensors,
        "has_bin": has_bin,
        "suspicious_names": suspicious_names[:30],
    }


def warnings_from_meta(meta: Dict[str, Any]) -> List[str]:
    w: List[str] = []
    if (meta.get("Gated") == "Yes") or (meta.get("Private") == "Yes"):
        w.append("Repo may require HF_TOKEN (private/gated).")

    ts = meta.get("_risk", {}).get("total_size_known")
    if isinstance(ts, int) and ts > 8 * 1024**3:
        w.append("Large repo size detected (>8GB). Downloads may be slow; consider selective download.")

    if meta.get("_has_gguf"):
        w.append("GGUF detected. Prefer llama-cpp-python / llama.cpp flow for local CPU/GPU inference.")

    pipeline = meta.get("Pipeline", "N/A")
    if pipeline == "text-generation":
        w.append("text-generation models often need GPU for good speed; device_map='auto' helps but not magic.")

    return w


def status_card(meta_public: Dict[str, Any], warnings: List[str], rt: str, rid: str) -> str:
    url = hf_url(rt, rid)

    last_mod = meta_public.get("Last Modified", "N/A")
    last_mod = str(last_mod).split()[0] if last_mod and last_mod != "N/A" else "N/A"

    pills = []
    if rt == "space":
        sdk = meta_public.get("SDK", "N/A")
        if sdk and sdk != "N/A":
            pills.append(f"<span class='pill'>SDK: {esc(sdk)}</span>")

    license_ = meta_public.get("License", "N/A")
    if license_ and license_ != "N/A":
        pills.append(f"<span class='pill'>{esc(license_)}</span>")

    pipeline = meta_public.get("Pipeline", "N/A")
    if pipeline and pipeline != "N/A":
        pills.append(f"<span class='pill'>{esc(pipeline)}</span>")

    size_s = meta_public.get("Total Size", "N/A")
    if size_s and size_s != "N/A":
        pills.append(f"<span class='pill'>{esc(size_s)}</span>")

    gated = meta_public.get("Gated", "N/A")
    if gated == "Yes":
        pills.append("<span class='pill warn'>Gated</span>")

    warn_html = ""
    if warnings:
        items = "".join([f"<li>{esc(x)}</li>" for x in warnings])
        warn_html = f"""
        <div class="warnbox">
          <div class="warn_title">Warnings</div>
          <ul class="warn_list">{items}</ul>
        </div>
        """

    pills_html = "".join(pills) if pills else ""

    likes = meta_public.get("Likes", 0)
    downloads = meta_public.get("Downloads", 0)
    author = meta_public.get("Author", "N/A")

    return f"""
    <div class="card ok">
      <div class="head">
        <div class="title">{esc(meta_public.get("Repo ID", rid))}</div>
        <a class="link" href="{esc(url)}" target="_blank">Open</a>
      </div>

      <div class="pills">{pills_html}</div>

      <div class="stats">
        <div class="stat accent">
          <div class="k">Likes</div>
          <div class="v">{esc(likes)}</div>
        </div>
        <div class="stat">
          <div class="k">Downloads</div>
          <div class="v">{esc(downloads)}</div>
        </div>
        <div class="stat">
          <div class="k">Last modified</div>
          <div class="v">{esc(last_mod)}</div>
        </div>
        <div class="stat">
          <div class="k">Author</div>
          <div class="v">{esc(author)}</div>
        </div>
      </div>

      {warn_html}
    </div>
    """


def status_err_card(msg: str) -> str:
    return f"""
    <div class="card err">
      <div class="title">Failed</div>
      <div class="msg">{esc(msg)}</div>
      <div class="hint">
        If this is a private/gated repo, provide a token locally or enable a server token for trusted use.
      </div>
    </div>
    """


def render_risk_html(risk: Dict[str, Any]) -> str:
    suspicious = risk.get("suspicious_names") or []
    suspicious_html = ""
    if suspicious:
        items = "".join([f"<li><code>{esc(x)}</code></li>" for x in suspicious[:20]])
        suspicious_html = f"""
        <div class="riskbox">
          <div class="risk_title">Potentially Sensitive Filenames</div>
          <ul class="risk_list">{items}</ul>
          <div class="risk_note">Filename-based only (no content scanning).</div>
        </div>
        """

    feats = []
    if risk.get("has_gguf"):
        feats.append("GGUF")
    if risk.get("has_onnx"):
        feats.append("ONNX")
    if risk.get("has_safetensors"):
        feats.append("safetensors")
    if risk.get("has_bin"):
        feats.append(".bin")
    feats_s = ", ".join(feats) if feats else "N/A"

    size_s = human_bytes(risk.get("total_size_known")) if risk.get("total_size_known") else "N/A"

    return f"""
    <div class="card">
      <div class="title">Files and risk</div>
      <div class="mini_stats">
        <span>Files <b>{esc(risk.get("files_count", 0))}</b></span>
        <span>Size <b>{esc(size_s)}</b></span>
        <span>Artifacts <b>{esc(feats_s)}</b></span>
      </div>
      {suspicious_html}
    </div>
    """


def to_files_table(files: List[Dict[str, Any]], limit: int = 250) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for f in (files or [])[:limit]:
        rows.append([f.get("path", ""), human_bytes(f.get("size")) if isinstance(f.get("size"), int) else "N/A"])
    return rows


def filter_files(files: List[Dict[str, Any]], q: str, limit: int = 250) -> List[List[Any]]:
    q = (q or "").strip().lower()
    if not q:
        return to_files_table(files, limit=limit)

    out: List[List[Any]] = []
    for f in files or []:
        p = (f.get("path") or "")
        if q in p.lower():
            out.append([p, human_bytes(f.get("size")) if isinstance(f.get("size"), int) else "N/A"])
        if len(out) >= limit:
            break
    return out


def first_file_with_ext(files: List[Dict[str, Any]], ext: str) -> Optional[str]:
    ext = (ext or "").lower()
    for f in files or []:
        p = (f.get("path") or "")
        if p.lower().endswith(ext):
            return p
    return None


def compute_requirements(rt: str, meta: Dict[str, Any]) -> List[str]:
    rt = norm_type(rt)
    pipeline_tag = (meta or {}).get("_pipeline_tag", "N/A")
    has_gguf = bool((meta or {}).get("_has_gguf", False))

    if rt == "dataset":
        return ["datasets", "huggingface_hub"]

    if rt == "space":
        return ["gradio", "huggingface_hub", "requests"]

    if has_gguf:
        return ["huggingface_hub", "llama-cpp-python"]

    if pipeline_tag == "text-generation":
        return ["transformers", "huggingface_hub", "torch", "accelerate"]

    if pipeline_tag in {"image-classification", "image-to-text", "image-segmentation", "object-detection"}:
        return ["transformers", "huggingface_hub", "torch", "pillow", "requests"]

    return ["transformers", "huggingface_hub", "torch"]


def generate_install(rt: str, meta: Dict[str, Any]) -> str:
    return "pip install -U " + " ".join(compute_requirements(rt, meta))


def generate_quickstart(rt: str, rid: str, meta: Dict[str, Any]) -> str:
    rt = norm_type(rt)
    rid = norm_id(rid)

    pipeline_tag = (meta or {}).get("_pipeline_tag", "N/A")
    sdk = (meta or {}).get("_sdk", "N/A")
    has_gguf = bool((meta or {}).get("_has_gguf", False))
    files = (meta or {}).get("_files", []) or []

    if rt == "dataset":
        return textwrap.dedent(f"""        from datasets import load_dataset

        ds = load_dataset("{rid}")
        print(ds)
        """).strip()

    if rt == "space":
        repo_dir = rid.split("/")[-1]
        if sdk == "streamlit":
            return textwrap.dedent(f"""            import os
            import subprocess

            subprocess.check_call(["git", "clone", "{hf_url("space", rid)}"])
            os.chdir("{repo_dir}")
            subprocess.check_call(["python", "-m", "pip", "install", "-r", "requirements.txt"])
            subprocess.check_call(["streamlit", "run", "app.py"])
            """).strip()
        return textwrap.dedent(f"""        import os
        import subprocess

        subprocess.check_call(["git", "clone", "{hf_url("space", rid)}"])
        os.chdir("{repo_dir}")
        subprocess.check_call(["python", "-m", "pip", "install", "-r", "requirements.txt"])
        subprocess.check_call(["python", "app.py"])
        """).strip()

    if has_gguf:
        gguf_name = first_file_with_ext(files, ".gguf") or "MODEL.gguf"
        return textwrap.dedent(f"""        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        gguf_path = hf_hub_download(repo_id="{rid}", filename="{gguf_name}")
        llm = Llama(model_path=gguf_path, n_ctx=4096)

        out = llm("Q: Hello!\nA:", max_tokens=128)
        print(out["choices"][0]["text"])
        """).strip()

    if pipeline_tag == "text-generation":
        return textwrap.dedent(f"""        from transformers import pipeline

        pipe = pipeline(
            "text-generation",
            model="{rid}",
            device_map="auto",
        )
        out = pipe("Hello, Hugging Face!", max_new_tokens=64)
        print(out[0]["generated_text"])
        """).strip()

    if pipeline_tag == "text-classification":
        return textwrap.dedent(f"""        from transformers import pipeline

        clf = pipeline("text-classification", model="{rid}")
        print(clf("I love this project."))
        """).strip()

    if pipeline_tag == "image-classification":
        return textwrap.dedent(f"""        from transformers import pipeline
        from PIL import Image
        import requests
        from io import BytesIO

        img = Image.open(BytesIO(requests.get(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        ).content))
        pipe = pipeline("image-classification", model="{rid}")
        print(pipe(img))
        """).strip()

    return textwrap.dedent(f"""    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained("{rid}")
    model = AutoModel.from_pretrained("{rid}")
    print(type(model))
    """).strip()


def generate_snapshot_download(rt: str, rid: str) -> str:
    rt = norm_type(rt)
    rid = norm_id(rid)
    local_dir = rid.split("/")[-1]

    return textwrap.dedent(f"""    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="{rid}",
        repo_type="{rt}",
        local_dir="./{local_dir}",
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {{path}}")
    """).strip()


def generate_cli_download(rt: str, rid: str) -> str:
    rt = norm_type(rt)
    rid = norm_id(rid)
    return f'huggingface-cli download {rid} --repo-type {rt} --local-dir "./downloaded_repo" --local-dir-use-symlinks False'


def generate_badge(rt: str, rid: str) -> str:
    rt = norm_type(rt)
    rid = norm_id(rid)
    url = hf_url(rt, rid)
    encoded = rid.replace("/", "%2F")
    return f"[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-{encoded}-blue)]({url})"


def token_allowed_for_repo(repo_id: str) -> bool:
    owners = os.getenv("TOKEN_ALLOWED_OWNERS", "").strip()
    if not owners:
        return True
    allowed = {x.strip().lower() for x in owners.split(",") if x.strip()}
    owner = (norm_id(repo_id).split("/")[0] if "/" in norm_id(repo_id) else "").lower()
    return bool(owner) and owner in allowed


def get_effective_token(repo_id: str) -> Optional[str]:
    if os.getenv("ALLOW_SERVER_TOKEN", "").strip() != "1":
        return None
    t = (os.getenv("HF_TOKEN") or "").strip()
    if not t:
        return None
    return t if token_allowed_for_repo(repo_id) else None


def fetch_repo_info(repo_type: str, repo_id: str, token: Optional[str]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    api = HfApi()
    rt = norm_type(repo_type)
    rid = norm_id(repo_id)
    token = (token or "").strip() or None

    if not rid:
        return False, None, "Empty Repo ID."
    if not is_valid_repo_id(rid):
        return False, None, "Invalid Repo ID. Expected: owner/name"

    try:
        if rt == "dataset":
            info = call_with_optional_kwargs(api.dataset_info, rid, token=token, files_metadata=True)
        elif rt == "space":
            info = call_with_optional_kwargs(api.space_info, rid, token=token, files_metadata=True)
        else:
            info = call_with_optional_kwargs(api.model_info, rid, token=token, files_metadata=True)

        card = getattr(info, "cardData", None) or {}
        license_ = card.get("license") or getattr(info, "license", None) or "N/A"

        gated = getattr(info, "gated", None)
        private = getattr(info, "private", None)

        pipeline = getattr(info, "pipeline_tag", None) or "N/A"
        sdk = getattr(info, "sdk", None) or "N/A"

        files = extract_file_entries(info)

        if not files:
            try:
                names = api.list_repo_files(repo_id=rid, repo_type=rt, token=token)
                files = [{"path": n, "size": None} for n in (names or [])]
            except Exception:
                files = []

        risk = files_risk_report(files)
        total_size_str = human_bytes(risk.get("total_size_known")) if risk.get("total_size_known") else "N/A"

        preview: Dict[str, Any] = {
            "Repo ID": getattr(info, "id", rid),
            "Type": rt,
            "Author": getattr(info, "author", None) or getattr(info, "owner", None) or "N/A",
            "Likes": getattr(info, "likes", 0) or 0,
            "Downloads": getattr(info, "downloads", 0) or 0,
            "Last Modified": safe_str(getattr(info, "lastModified", "N/A"), 200),
            "License": str(license_) if license_ else "N/A",
            "Pipeline": str(pipeline) if pipeline else "N/A",
            "Gated": "Yes" if gated is True else ("No" if gated is False else "N/A"),
            "Private": "Yes" if private is True else ("No" if private is False else "N/A"),
            "Total Size": total_size_str,
            "Files Count": risk.get("files_count", 0),
        }

        if rt == "space":
            preview["SDK"] = sdk or "N/A"
            hw = getattr(info, "hardware", None)
            if hw:
                preview["Hardware"] = safe_str(hw, 200)

        preview["_pipeline_tag"] = pipeline or "N/A"
        preview["_sdk"] = sdk or "N/A"
        preview["_files"] = files
        preview["_risk"] = risk
        preview["_has_gguf"] = bool(risk.get("has_gguf"))

        return True, preview, None

    except HfHubHTTPError as e:
        return False, None, safe_hf_error(e)
    except Exception as e:
        return False, None, f"Unexpected Error: {safe_str(e, 500)}"


@lru_cache(maxsize=512)
def cached_public(repo_type: str, repo_id: str):
    return fetch_repo_info(repo_type, repo_id, token=None)


def build_quickstart_zip(state: Dict[str, Any]) -> Tuple[Optional[str], str]:
    if not isinstance(state, dict) or not state.get("Repo ID"):
        return None, "No repo loaded yet."

    rt = norm_type(state.get("Type", "model"))
    rid = norm_id(state.get("Repo ID", "")) or norm_id(state.get("_rid", ""))

    install = generate_install(rt, state)
    quickstart = generate_quickstart(rt, rid, state)
    snap = generate_snapshot_download(rt, rid)

    readme = textwrap.dedent(f"""    # QuickStart — {rid}

    ## Setup
    ```bash
    python -m venv .venv
    pip install -U pip
    pip install -r requirements.txt
    ```

    ## Run
    ```bash
    python run.py
    ```

    ## Download (optional)
    ```bash
    python download.py
    ```
    """).strip()

    requirements = compute_requirements(rt, state)
    env_example = "HF_TOKEN=\n"

    run_py = textwrap.dedent(f"""    import os

    def main():
        print("Install (reference):")
        print("{install}")

    {textwrap.indent(quickstart, "    ")}

    if __name__ == "__main__":
        main()
    """)

    download_py = textwrap.dedent(f"""    {snap}
    """)

    tmpdir = tempfile.mkdtemp(prefix="quickstart_")
    zip_path = os.path.join(tmpdir, f"{rid.replace('/', '__')}_quickstart.zip")

    proj_dir = os.path.join(tmpdir, "project")
    os.makedirs(proj_dir, exist_ok=True)

    def write_file(path: str, content: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    write_file(os.path.join(proj_dir, "README.md"), readme + "\n")
    write_file(os.path.join(proj_dir, "requirements.txt"), "\n".join(requirements) + "\n")
    write_file(os.path.join(proj_dir, ".env.example"), env_example)
    write_file(os.path.join(proj_dir, "run.py"), run_py + "\n")
    write_file(os.path.join(proj_dir, "download.py"), download_py + "\n")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname in ["README.md", "requirements.txt", ".env.example", "run.py", "download.py"]:
            z.write(os.path.join(proj_dir, fname), arcname=fname)

    return zip_path, "Zip built. Download it, unzip it, then run: python run.py"


def process(user_input: str, type_override: str):
    auto_type, rid = parse_hf_input(user_input)
    rt = auto_type if (type_override or "auto") == "auto" else norm_type(type_override)
    rid = norm_id(rid)

    token = get_effective_token(rid)

    if token:
        ok, meta, err = fetch_repo_info(rt, rid, token=token)
    else:
        ok, meta, err = cached_public(rt, rid)

    if not ok or not meta:
        empty_rows: List[List[Any]] = []
        return (
            status_err_card(err or "Unknown error"),
            "",
            "",
            "",
            "",
            "",
            empty_rows,
            "",
            {},
            {},
        )

    meta_public = {k: v for k, v in meta.items() if not str(k).startswith("_")}

    install = generate_install(rt, meta)
    quickstart = generate_quickstart(rt, rid, meta)
    snap = generate_snapshot_download(rt, rid)
    cli = generate_cli_download(rt, rid)
    badge = generate_badge(rt, rid)

    files = meta.get("_files", []) or []
    risk = meta.get("_risk", {}) or {}
    warnings = warnings_from_meta(meta)

    status = status_card(meta_public, warnings, rt, rid)
    files_rows = to_files_table(files, limit=250)
    risk_html = render_risk_html(risk)

    state = dict(meta)
    state["_rid"] = rid
    state["_rt"] = rt

    return (
        status,
        install,
        quickstart,
        snap,
        cli,
        badge,
        files_rows,
        risk_html,
        meta_public,
        state,
    )


def do_filter_files(state: Dict[str, Any], q: str):
    files = (state or {}).get("_files", []) or []
    return filter_files(files, q, limit=250)


def build_ui():
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        radius_size=gr.themes.sizes.radius_lg,
    )

    css = """
    .gradio-container { max-width: 1120px !important; margin: auto; }

    .hero{
      padding: 18px 18px;
      border-radius: 18px;
      border: 1px solid rgba(148,163,184,.25);
      background:
        radial-gradient(1200px 300px at 30% 0%, rgba(249,115,22,.18), transparent 60%),
        radial-gradient(1000px 260px at 70% 20%, rgba(99,102,241,.14), transparent 55%),
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0));
      box-shadow: 0 12px 40px rgba(0,0,0,.10);
      margin-bottom: 14px;
    }
    h1{
      text-align:center;
      margin: 0 0 6px 0;
      color: var(--body-text-color);
      font-weight: 850;
      letter-spacing: -0.03em;
    }
    .sub{
      text-align:center;
      color: var(--body-text-color-subdued);
      margin: 0;
      line-height: 1.45;
    }

    .card{
      padding: 14px 16px;
      border-radius: 16px;
      background: var(--block-background-fill);
      border: 1px solid var(--block-border-color);
      color: var(--body-text-color);
      box-shadow: 0 10px 30px rgba(0,0,0,.06);
    }
    .ok{ border-left: 6px solid rgba(16,185,129,.95); }
    .err{ border-left: 6px solid rgba(239,68,68,.95); }

    .head{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
    .title{ font-weight: 850; font-size: 1.02rem; }
    .link{
      color: #ffffff !important;
      text-decoration: none !important;
      font-weight: 900;
      padding: 6px 12px;
      border-radius: 10px;
      border: none !important;
      background: linear-gradient(135deg, rgba(249,115,22,1), rgba(245,158,11,1)) !important;
      box-shadow: 0 10px 26px rgba(249,115,22,.18);
    }
    .link:hover{
      filter: brightness(1.05);
      transform: translateY(-0.5px);
    }

    .pills{ margin-top: 10px; display:flex; gap: 10px; flex-wrap:wrap; }
    .pill{
      display:inline-flex; align-items:center;
      padding: 4px 10px; border-radius: 999px; font-size: .82rem;
      border: 1px solid rgba(148,163,184,.28);
      background: rgba(255,255,255,0.04);
      color: var(--body-text-color);
    }
    .pill.warn{
      border-color: rgba(245,158,11,.35);
      background: rgba(245,158,11,.10);
    }

    .stats{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 10px;
    }
    .stat{
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,.22);
      background:
        radial-gradient(600px 140px at 0% 0%, rgba(255,255,255,.05), transparent 55%),
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0));
    }
    .stat .k{
      font-size: .76rem;
      color: var(--body-text-color-subdued);
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .stat .v{
      margin-top: 6px;
      font-weight: 900;
      font-size: 1.06rem;
      color: var(--body-text-color);
      font-variant-numeric: tabular-nums;
    }
    .stat.accent{
      border-color: rgba(249,115,22,.30);
      background:
        radial-gradient(700px 160px at 10% 0%, rgba(249,115,22,.20), transparent 60%),
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0));
    }
    @media (max-width: 820px){
      .stats{ grid-template-columns: repeat(2, minmax(150px, 1fr)); }
    }
    @media (max-width: 460px){
      .stats{ grid-template-columns: 1fr; }
    }

    .warnbox{
      margin-top: 12px;
      padding: 12px 12px;
      border-radius: 14px;
      border: 1px solid rgba(245, 158, 11, .30);
      background: rgba(245, 158, 11, .08);
    }
    .warn_title{ font-weight: 850; margin-bottom: 6px; }
    .warn_list{ margin: 0; padding-left: 18px; color: var(--body-text-color); }

    .mini_stats{
      display:flex;
      gap: 14px;
      flex-wrap:wrap;
      margin-top: 10px;
      color: var(--body-text-color-subdued);
      font-size: .92rem;
    }

    .riskbox{
      margin-top: 12px;
      padding: 12px 12px;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,.20);
      background: rgba(255,255,255,0.03);
    }
    .risk_title{ font-weight: 850; margin-bottom: 6px; }
    .risk_list{ margin: 0; padding-left: 18px; color: var(--body-text-color); }
    .risk_note{ margin-top: 6px; color: var(--body-text-color-subdued); font-size: .9rem; }

    button.primary, .gr-button-primary, .primary > button {
      border: none !important;
      background: linear-gradient(135deg, rgba(249,115,22,1), rgba(245,158,11,1)) !important;
      color: white !important;
      font-weight: 850 !important;
      box-shadow: 0 10px 26px rgba(249,115,22,.18);
    }
    button.primary:hover, .gr-button-primary:hover, .primary > button:hover {
      filter: brightness(1.05);
      transform: translateY(-0.5px);
    }
    """

    df_sig = inspect.signature(gr.Dataframe)
    df_count_kw = {"column_count": (2, "fixed")} if "column_count" in df_sig.parameters else {"col_count": (2, "fixed")}

    with gr.Blocks(title="QuickStart") as demo:
        gr.Markdown(
            "<div class='hero'>"
            "<h1>QuickStart</h1>"
            "<p class='sub'>Paste a Hugging Face URL or Repo ID to generate run/download snippets and export a ready zip.</p>"
            "</div>"
        )

        state = gr.State({})

        with gr.Row(variant="panel"):
            with gr.Column(scale=7):
                inp = gr.Textbox(
                    label="HF URL or Repo ID",
                    placeholder="google/gemma-2-9b-it or https://huggingface.co/datasets/squad",
                    autofocus=True,
                )
            with gr.Column(scale=2):
                t = gr.Dropdown(["auto", "model", "dataset", "space"], value="auto", label="Type")
            with gr.Column(scale=2):
                btn = gr.Button("Generate", variant="primary")

        out_status = gr.HTML(label="Summary")

        with gr.Tabs():
            with gr.TabItem("QuickStart"):
                out_py = gr.Code(language="python", label="Python QuickStart", interactive=False)
                copy_py = gr.Button("Copy")
                out_install = gr.Code(language="shell", label="Install", interactive=False)
                copy_install = gr.Button("Copy")

            with gr.TabItem("Download"):
                out_snap = gr.Code(language="python", label="snapshot_download()", interactive=False)
                copy_snap = gr.Button("Copy")
                out_cli = gr.Code(language="shell", label="huggingface-cli download", interactive=False)
                copy_cli = gr.Button("Copy")

            with gr.TabItem("Files"):
                file_q = gr.Textbox(label="Filter", placeholder="e.g. .gguf or config.json")
                files_table = gr.Dataframe(
                    headers=["path", "size"],
                    datatype=["str", "str"],
                    label="Files (first 250)",
                    interactive=False,
                    row_count=10,
                    **df_count_kw,
                )
                risk_html = gr.HTML(label="Risk")

            with gr.TabItem("Export"):
                gr.Markdown("Exports a zip: `run.py`, `download.py`, `requirements.txt`, `.env.example`, `README.md`.")
                zip_btn = gr.Button("Build Zip", variant="primary")
                zip_file = gr.File(label="Zip file")
                zip_msg = gr.Markdown()

            with gr.TabItem("Badge"):
                out_badge = gr.Code(language="markdown", label="Markdown", interactive=False)
                copy_badge = gr.Button("Copy")

        with gr.Accordion("Details", open=False):
            out_meta = gr.JSON(label="Metadata")

        outputs = [
            out_status,
            out_install,
            out_py,
            out_snap,
            out_cli,
            out_badge,
            files_table,
            risk_html,
            out_meta,
            state,
        ]

        btn.click(process, inputs=[inp, t], outputs=outputs)
        inp.submit(process, inputs=[inp, t], outputs=outputs)

        file_q.change(do_filter_files, inputs=[state, file_q], outputs=[files_table])
        zip_btn.click(build_quickstart_zip, inputs=[state], outputs=[zip_file, zip_msg])

        js_copy = "(t)=>{ if(!t){return [];} navigator.clipboard.writeText(String(t)); return []; }"
        copy_install.click(None, inputs=[out_install], outputs=[], js=js_copy)
        copy_py.click(None, inputs=[out_py], outputs=[], js=js_copy)
        copy_snap.click(None, inputs=[out_snap], outputs=[], js=js_copy)
        copy_cli.click(None, inputs=[out_cli], outputs=[], js=js_copy)
        copy_badge.click(None, inputs=[out_badge], outputs=[], js=js_copy)

    return demo, theme, css


if __name__ == "__main__":
    app, theme, css = build_ui()
    app.launch(theme=theme, css=css)
