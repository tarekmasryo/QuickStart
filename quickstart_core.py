"""Core helpers for the QuickStart Hugging Face repo assistant."""

from __future__ import annotations

import html
import inspect
import os
import re
import tempfile
import textwrap
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

VALID_REPO_TYPES = {"model", "dataset", "space"}
RE_REPO_ID = re.compile(
    r"^(?!.*(?:--|\.\.))[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$"
)
SENSITIVE_FILENAME_PATTERNS = [
    r"(^|/)\.env$",
    r"secrets?",
    r"token",
    r"api[_-]?key",
    r"credentials?",
    r"id_rsa",
    r"\.pem$",
    r"\.p12$",
    r"\.kdbx$",
]


def esc(value: Any) -> str:
    """HTML-escape values before injecting them into custom Gradio HTML."""
    return html.escape("" if value is None else str(value), quote=True)


def norm_type(value: str | None) -> str:
    repo_type = (value or "model").strip().lower()
    return repo_type if repo_type in VALID_REPO_TYPES else "model"


def norm_id(value: str | None) -> str:
    return (value or "").strip().strip("/")


def is_valid_repo_id(repo_id: str) -> bool:
    repo_id = (repo_id or "").strip()
    if not RE_REPO_ID.match(repo_id):
        return False
    return all(not part.startswith(("-", ".")) and not part.endswith(("-", ".")) for part in repo_id.split("/"))


def human_bytes(num_bytes: int | None) -> str:
    if not isinstance(num_bytes, int) or num_bytes <= 0:
        return "N/A"

    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    return f"{value:.2f} {units[unit_index]}"


def safe_str(value: Any, max_chars: int = 500) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def parse_hf_input(user_input: str) -> tuple[str, str]:
    """Parse a Hugging Face URL, typed repo path, or plain owner/repo ID."""
    value = (user_input or "").strip()
    if not value:
        return "model", ""

    if "huggingface.co" in value:
        scoped_match = re.search(r"huggingface\.co/(datasets|spaces)/([^?#]+)", value)
        if scoped_match:
            repo_type = "dataset" if scoped_match.group(1) == "datasets" else "space"
            repo_id = _strip_hf_file_path(scoped_match.group(2))
            return repo_type, repo_id

        model_match = re.search(r"huggingface\.co/([^?#]+)", value)
        if model_match:
            repo_id = _strip_hf_file_path(model_match.group(1))
            return "model", repo_id

    if value.startswith("datasets/"):
        return "dataset", value.replace("datasets/", "", 1).strip("/")
    if value.startswith("spaces/"):
        return "space", value.replace("spaces/", "", 1).strip("/")

    return "model", value.strip("/")


def _strip_hf_file_path(path: str) -> str:
    path = (path or "").strip("/")
    path = re.split(r"/(tree|blob|resolve|raw)/", path)[0].strip("/")
    return path


def hf_url(repo_type: str, repo_id: str) -> str:
    repo_type = norm_type(repo_type)
    repo_id = norm_id(repo_id)
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}"
    if repo_type == "space":
        return f"https://huggingface.co/spaces/{repo_id}"
    return f"https://huggingface.co/{repo_id}"


def safe_hf_error(error: HfHubHTTPError) -> str:
    status = getattr(getattr(error, "response", None), "status_code", "N/A")
    message = getattr(error, "server_message", None) or str(error)
    return f"Hugging Face Error: {status} - {safe_str(message, 500)}"


def call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call SDK functions with only supported kwargs without swallowing API errors."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)

    allowed = set(signature.parameters)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    return fn(*args, **supported_kwargs)


def extract_file_entries(info_obj: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    siblings = getattr(info_obj, "siblings", None) or []
    for sibling in siblings:
        path = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
        if not path:
            continue

        size = getattr(sibling, "size", None)
        if size is None:
            lfs = getattr(sibling, "lfs", None)
            size = getattr(lfs, "size", None) if lfs is not None else None

        entries.append({"path": str(path), "size": int(size) if isinstance(size, int) else None})
    return entries


def files_risk_report(files: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [item.get("path", "") for item in files if item.get("path")]
    total_known = sum(int(item["size"]) for item in files if isinstance(item.get("size"), int))

    lower_paths = [path.lower() for path in paths]
    suspicious_names = [
        path
        for path in paths
        if any(re.search(pattern, path.lower()) for pattern in SENSITIVE_FILENAME_PATTERNS)
    ]

    return {
        "files_count": len(paths),
        "total_size_known": total_known if total_known > 0 else None,
        "has_gguf": any(path.endswith(".gguf") for path in lower_paths),
        "has_onnx": any(path.endswith(".onnx") for path in lower_paths),
        "has_safetensors": any(path.endswith(".safetensors") for path in lower_paths),
        "has_bin": any(path.endswith(".bin") for path in lower_paths),
        "suspicious_names": suspicious_names[:30],
    }


def warnings_from_meta(meta: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    risk = meta.get("_risk", {}) or {}

    if meta.get("Gated") == "Yes" or meta.get("Private") == "Yes":
        warnings.append("Repo may require HF_TOKEN because it is private or gated.")

    total_size = risk.get("total_size_known")
    if isinstance(total_size, int) and total_size > 8 * 1024**3:
        warnings.append("Large repo size detected (>8GB). Prefer selective download when possible.")

    if risk.get("has_gguf"):
        warnings.append("GGUF detected. Use a llama.cpp / llama-cpp-python flow instead of generic Transformers.")

    if risk.get("suspicious_names"):
        warnings.append("Potentially sensitive filenames detected. This is filename-based only; review before use.")

    if meta.get("Pipeline") == "text-generation":
        warnings.append("Text-generation models can be slow without adequate GPU/VRAM.")

    return warnings


def to_files_table(files: list[dict[str, Any]], limit: int = 250) -> list[list[Any]]:
    return [
        [item.get("path", ""), human_bytes(item.get("size")) if isinstance(item.get("size"), int) else "N/A"]
        for item in (files or [])[:limit]
    ]


def filter_files(files: list[dict[str, Any]], query: str, limit: int = 250) -> list[list[Any]]:
    query = (query or "").strip().lower()
    if not query:
        return to_files_table(files, limit=limit)

    rows: list[list[Any]] = []
    for item in files or []:
        path = item.get("path") or ""
        if query in path.lower():
            size = human_bytes(item.get("size")) if isinstance(item.get("size"), int) else "N/A"
            rows.append([path, size])
        if len(rows) >= limit:
            break
    return rows


def first_file_with_ext(files: list[dict[str, Any]], extension: str) -> str | None:
    extension = (extension or "").lower()
    for item in files or []:
        path = item.get("path") or ""
        if path.lower().endswith(extension):
            return path
    return None


def compute_requirements(repo_type: str, meta: dict[str, Any]) -> list[str]:
    repo_type = norm_type(repo_type)
    pipeline_tag = (meta or {}).get("_pipeline_tag", "N/A")
    sdk = (meta or {}).get("_sdk", "N/A")
    has_gguf = bool((meta or {}).get("_risk", {}).get("has_gguf") or (meta or {}).get("_has_gguf"))

    if repo_type == "dataset":
        return ["datasets", "huggingface_hub"]

    if repo_type == "space":
        if sdk == "streamlit":
            return ["streamlit", "huggingface_hub", "requests"]
        if sdk == "gradio":
            return ["gradio", "huggingface_hub", "requests"]
        return ["huggingface_hub", "requests"]

    if has_gguf:
        return ["huggingface_hub", "llama-cpp-python"]

    if pipeline_tag == "text-generation":
        return ["transformers", "huggingface_hub", "torch", "accelerate"]

    if pipeline_tag in {"image-classification", "image-to-text", "image-segmentation", "object-detection"}:
        return ["transformers", "huggingface_hub", "torch", "pillow", "requests"]

    return ["transformers", "huggingface_hub", "torch"]


def generate_install(repo_type: str, meta: dict[str, Any]) -> str:
    return "python -m pip install " + " ".join(compute_requirements(repo_type, meta))


def generate_quickstart(repo_type: str, repo_id: str, meta: dict[str, Any]) -> str:
    repo_type = norm_type(repo_type)
    repo_id = norm_id(repo_id)
    pipeline_tag = (meta or {}).get("_pipeline_tag", "N/A")
    sdk = (meta or {}).get("_sdk", "N/A")
    risk = (meta or {}).get("_risk", {}) or {}
    has_gguf = bool(risk.get("has_gguf") or (meta or {}).get("_has_gguf"))
    files = (meta or {}).get("_files", []) or []

    if repo_type == "dataset":
        return textwrap.dedent(
            f"""
            from datasets import load_dataset

            ds = load_dataset("{repo_id}")
            print(ds)
            """
        ).strip()

    if repo_type == "space":
        repo_dir = repo_id.split("/")[-1]
        if sdk == "streamlit":
            return textwrap.dedent(
                f"""
                import os
                import subprocess

                subprocess.check_call(["git", "clone", "{hf_url('space', repo_id)}"])
                os.chdir("{repo_dir}")
                subprocess.check_call(["python", "-m", "pip", "install", "-r", "requirements.txt"])
                subprocess.check_call(["streamlit", "run", "app.py"])
                """
            ).strip()

        return textwrap.dedent(
            f"""
            import os
            import subprocess

            subprocess.check_call(["git", "clone", "{hf_url('space', repo_id)}"])
            os.chdir("{repo_dir}")
            subprocess.check_call(["python", "-m", "pip", "install", "-r", "requirements.txt"])
            subprocess.check_call(["python", "app.py"])
            """
        ).strip()

    if has_gguf:
        gguf_name = first_file_with_ext(files, ".gguf") or "MODEL.gguf"
        return textwrap.dedent(
            f"""
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama

            gguf_path = hf_hub_download(repo_id="{repo_id}", filename="{gguf_name}")
            llm = Llama(model_path=gguf_path, n_ctx=4096)

            out = llm("Q: Hello!\\nA:", max_tokens=128)
            print(out["choices"][0]["text"])
            """
        ).strip()

    if pipeline_tag == "text-generation":
        return textwrap.dedent(
            f"""
            from transformers import pipeline

            pipe = pipeline(
                "text-generation",
                model="{repo_id}",
                device_map="auto",
            )
            out = pipe("Hello, Hugging Face!", max_new_tokens=64)
            print(out[0]["generated_text"])
            """
        ).strip()

    if pipeline_tag == "text-classification":
        return textwrap.dedent(
            f"""
            from transformers import pipeline

            clf = pipeline("text-classification", model="{repo_id}")
            print(clf("I love this project."))
            """
        ).strip()

    if pipeline_tag == "image-classification":
        return textwrap.dedent(
            f"""
            from io import BytesIO

            import requests
            from PIL import Image
            from transformers import pipeline

            image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            image = Image.open(BytesIO(requests.get(image_url, timeout=20).content))
            pipe = pipeline("image-classification", model="{repo_id}")
            print(pipe(image))
            """
        ).strip()

    return textwrap.dedent(
        f"""
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
        model = AutoModel.from_pretrained("{repo_id}")
        print(type(tokenizer))
        print(type(model))
        """
    ).strip()


def generate_snapshot_download(repo_type: str, repo_id: str) -> str:
    repo_type = norm_type(repo_type)
    repo_id = norm_id(repo_id)
    local_dir = repo_id.split("/")[-1]

    lines = [
        "from huggingface_hub import snapshot_download",
        "",
        "path = snapshot_download(",
        f'    repo_id="{repo_id}",',
    ]
    if repo_type != "model":
        lines.append(f'    repo_type="{repo_type}",')
    lines.extend(
        [
            f'    local_dir="./{local_dir}",',
            ")",
            'print(f"Downloaded to: {path}")',
        ]
    )
    return "\n".join(lines)


def generate_cli_download(repo_type: str, repo_id: str) -> str:
    repo_type = norm_type(repo_type)
    repo_id = norm_id(repo_id)
    return f'hf download {repo_id} --repo-type {repo_type} --local-dir "./downloaded_repo"'


def generate_badge(repo_type: str, repo_id: str) -> str:
    repo_type = norm_type(repo_type)
    repo_id = norm_id(repo_id)
    url = hf_url(repo_type, repo_id)
    encoded = repo_id.replace("/", "%2F")
    return f"[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-{encoded}-blue)]({url})"


def token_allowed_for_repo(repo_id: str) -> bool:
    owners = os.getenv("TOKEN_ALLOWED_OWNERS", "").strip()
    if not owners:
        return True

    allowed_owners = {owner.strip().lower() for owner in owners.split(",") if owner.strip()}
    owner = (norm_id(repo_id).split("/")[0] if "/" in norm_id(repo_id) else "").lower()
    return bool(owner) and owner in allowed_owners


def get_effective_token(repo_id: str) -> str | None:
    if os.getenv("ALLOW_SERVER_TOKEN", "").strip() != "1":
        return None

    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        return None

    return token if token_allowed_for_repo(repo_id) else None


def fetch_repo_info(
    repo_type: str, repo_id: str, token: str | None
) -> tuple[bool, dict[str, Any] | None, str | None]:
    api = HfApi()
    repo_type = norm_type(repo_type)
    repo_id = norm_id(repo_id)
    token = (token or "").strip() or None

    if not repo_id:
        return False, None, "Empty Repo ID."
    if not is_valid_repo_id(repo_id):
        return False, None, "Invalid Repo ID. Expected: owner/name"

    try:
        if repo_type == "dataset":
            info = call_with_supported_kwargs(api.dataset_info, repo_id, token=token, files_metadata=True)
        elif repo_type == "space":
            info = call_with_supported_kwargs(api.space_info, repo_id, token=token, files_metadata=True)
        else:
            info = call_with_supported_kwargs(api.model_info, repo_id, token=token, files_metadata=True)

        card = getattr(info, "cardData", None) or {}
        license_name = card.get("license") or getattr(info, "license", None) or "N/A"
        gated = getattr(info, "gated", None)
        private = getattr(info, "private", None)
        pipeline = getattr(info, "pipeline_tag", None) or "N/A"
        sdk = getattr(info, "sdk", None) or "N/A"
        files = extract_file_entries(info)

        if not files:
            try:
                names = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
                files = [{"path": name, "size": None} for name in (names or [])]
            except Exception:
                files = []

        risk = files_risk_report(files)
        total_size = human_bytes(risk.get("total_size_known")) if risk.get("total_size_known") else "N/A"

        preview: dict[str, Any] = {
            "Repo ID": getattr(info, "id", repo_id),
            "Type": repo_type,
            "Author": getattr(info, "author", None) or getattr(info, "owner", None) or "N/A",
            "Likes": getattr(info, "likes", 0) or 0,
            "Downloads": getattr(info, "downloads", 0) or 0,
            "Last Modified": safe_str(getattr(info, "lastModified", "N/A"), 200),
            "License": str(license_name) if license_name else "N/A",
            "Pipeline": str(pipeline) if pipeline else "N/A",
            "Gated": "Yes" if gated is True else ("No" if gated is False else "N/A"),
            "Private": "Yes" if private is True else ("No" if private is False else "N/A"),
            "Total Size": total_size,
            "Files Count": risk.get("files_count", 0),
        }

        if repo_type == "space":
            preview["SDK"] = sdk or "N/A"
            hardware = getattr(info, "hardware", None)
            if hardware:
                preview["Hardware"] = safe_str(hardware, 200)

        preview.update(
            {
                "_pipeline_tag": pipeline or "N/A",
                "_sdk": sdk or "N/A",
                "_files": files,
                "_risk": risk,
                "_has_gguf": bool(risk.get("has_gguf")),
                "_rid": repo_id,
                "_rt": repo_type,
            }
        )
        return True, preview, None

    except HfHubHTTPError as error:
        return False, None, safe_hf_error(error)
    except Exception as error:
        return False, None, f"Unexpected Error: {safe_str(error, 500)}"


@lru_cache(maxsize=512)
def cached_public(repo_type: str, repo_id: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    return fetch_repo_info(repo_type, repo_id, token=None)


def build_export_files(state: dict[str, Any]) -> dict[str, str]:
    if not isinstance(state, dict) or not state.get("Repo ID"):
        raise ValueError("No repo loaded yet.")

    repo_type = norm_type(state.get("Type", "model"))
    repo_id = norm_id(state.get("Repo ID", "")) or norm_id(state.get("_rid", ""))

    install = generate_install(repo_type, state)
    quickstart = generate_quickstart(repo_type, repo_id, state)
    snapshot = generate_snapshot_download(repo_type, repo_id)
    requirements = compute_requirements(repo_type, state)

    readme = textwrap.dedent(
        f"""
        # QuickStart — {repo_id}

        Minimal first-run scaffold generated for `{repo_id}`.

        ## Setup

        ```bash
        python -m venv .venv
        python -m pip install -r requirements.txt
        ```

        ## Run

        ```bash
        python run.py
        ```

        ## Download full snapshot

        ```bash
        python download.py
        ```

        ## Reference install

        ```bash
        {install}
        ```
        """
    ).strip()

    run_py = "\n".join(
        [
            "def main():",
            "    print(\"Install/reference command:\")",
            f"    print({install!r})",
            "",
            textwrap.indent(quickstart, "    "),
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ]
    )

    download_py = snapshot.strip()

    return {
        "README.md": readme + "\n",
        "requirements.txt": "\n".join(requirements) + "\n",
        ".env.example": "HF_TOKEN=\n",
        "run.py": run_py + "\n",
        "download.py": download_py + "\n",
    }


def build_quickstart_zip(state: dict[str, Any]) -> tuple[str | None, str]:
    try:
        files = build_export_files(state)
    except ValueError as error:
        return None, str(error)

    repo_id = norm_id(state.get("Repo ID", "")) or norm_id(state.get("_rid", "repo"))
    temp_dir = Path(tempfile.mkdtemp(prefix="quickstart_"))
    zip_path = temp_dir / f"{repo_id.replace('/', '__')}_quickstart.zip"

    project_dir = temp_dir / "project"
    project_dir.mkdir(parents=True, exist_ok=True)

    for name, content in files.items():
        path = project_dir / name
        path.write_text(content, encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(project_dir / name, arcname=name)

    return str(zip_path), "Zip built. Download it, unzip it, then run: python run.py"
