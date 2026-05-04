"""Gradio UI for QuickStart."""

from __future__ import annotations

import inspect
from typing import Any

import gradio as gr

from quickstart_core import (
    build_quickstart_zip,
    cached_public,
    esc,
    fetch_repo_info,
    filter_files,
    generate_badge,
    generate_cli_download,
    generate_install,
    generate_quickstart,
    generate_snapshot_download,
    get_effective_token,
    hf_url,
    human_bytes,
    norm_id,
    norm_type,
    parse_hf_input,
    to_files_table,
    warnings_from_meta,
)


def status_card(meta_public: dict[str, Any], warnings: list[str], repo_type: str, repo_id: str) -> str:
    url = hf_url(repo_type, repo_id)
    last_modified = meta_public.get("Last Modified", "N/A")
    last_modified = str(last_modified).split()[0] if last_modified and last_modified != "N/A" else "N/A"

    pills: list[str] = []
    if repo_type == "space":
        sdk = meta_public.get("SDK", "N/A")
        if sdk and sdk != "N/A":
            pills.append(f"<span class='pill'>SDK: {esc(sdk)}</span>")

    for key in ("License", "Pipeline", "Total Size"):
        value = meta_public.get(key, "N/A")
        if value and value != "N/A":
            pills.append(f"<span class='pill'>{esc(value)}</span>")

    if meta_public.get("Gated") == "Yes":
        pills.append("<span class='pill warn'>Gated</span>")

    warnings_html = ""
    if warnings:
        items = "".join(f"<li>{esc(item)}</li>" for item in warnings)
        warnings_html = f"""
        <div class="warnbox">
          <div class="warn_title">Warnings</div>
          <ul class="warn_list">{items}</ul>
        </div>
        """

    return f"""
    <div class="card ok">
      <div class="head">
        <div class="title">{esc(meta_public.get("Repo ID", repo_id))}</div>
        <a class="link" href="{esc(url)}" target="_blank">Open</a>
      </div>

      <div class="pills">{"".join(pills)}</div>

      <div class="stats">
        <div class="stat accent">
          <div class="k">Likes</div>
          <div class="v">{esc(meta_public.get("Likes", 0))}</div>
        </div>
        <div class="stat">
          <div class="k">Downloads</div>
          <div class="v">{esc(meta_public.get("Downloads", 0))}</div>
        </div>
        <div class="stat">
          <div class="k">Last modified</div>
          <div class="v">{esc(last_modified)}</div>
        </div>
        <div class="stat">
          <div class="k">Author</div>
          <div class="v">{esc(meta_public.get("Author", "N/A"))}</div>
        </div>
      </div>

      {warnings_html}
    </div>
    """


def status_error_card(message: str) -> str:
    return f"""
    <div class="card err">
      <div class="title">Failed</div>
      <div class="msg">{esc(message)}</div>
      <div class="hint">
        If this is a private/gated repo, enable a scoped server token locally or in Space settings.
      </div>
    </div>
    """


def render_risk_html(risk: dict[str, Any]) -> str:
    suspicious = risk.get("suspicious_names") or []
    suspicious_html = ""
    if suspicious:
        items = "".join(f"<li><code>{esc(item)}</code></li>" for item in suspicious[:20])
        suspicious_html = f"""
        <div class="riskbox">
          <div class="risk_title">Potentially Sensitive Filenames</div>
          <ul class="risk_list">{items}</ul>
          <div class="risk_note">Filename-based only. File contents are not scanned.</div>
        </div>
        """

    artifacts = []
    if risk.get("has_gguf"):
        artifacts.append("GGUF")
    if risk.get("has_onnx"):
        artifacts.append("ONNX")
    if risk.get("has_safetensors"):
        artifacts.append("safetensors")
    if risk.get("has_bin"):
        artifacts.append(".bin")
    artifacts_text = ", ".join(artifacts) if artifacts else "N/A"
    size_text = human_bytes(risk.get("total_size_known")) if risk.get("total_size_known") else "N/A"

    return f"""
    <div class="card">
      <div class="title">Files and risk</div>
      <div class="mini_stats">
        <span>Files <b>{esc(risk.get("files_count", 0))}</b></span>
        <span>Size <b>{esc(size_text)}</b></span>
        <span>Artifacts <b>{esc(artifacts_text)}</b></span>
      </div>
      {suspicious_html}
    </div>
    """


def process(user_input: str, type_override: str):
    auto_type, repo_id = parse_hf_input(user_input)
    repo_type = auto_type if (type_override or "auto") == "auto" else norm_type(type_override)
    repo_id = norm_id(repo_id)
    token = get_effective_token(repo_id)

    if token:
        ok, meta, error = fetch_repo_info(repo_type, repo_id, token=token)
    else:
        ok, meta, error = cached_public(repo_type, repo_id)

    if not ok or not meta:
        return (
            status_error_card(error or "Unknown error"),
            "",
            "",
            "",
            "",
            "",
            [],
            "",
            {},
            {},
        )

    meta_public = {key: value for key, value in meta.items() if not str(key).startswith("_")}
    install = generate_install(repo_type, meta)
    quickstart = generate_quickstart(repo_type, repo_id, meta)
    snapshot = generate_snapshot_download(repo_type, repo_id)
    cli = generate_cli_download(repo_type, repo_id)
    badge = generate_badge(repo_type, repo_id)

    files = meta.get("_files", []) or []
    risk = meta.get("_risk", {}) or {}
    state = dict(meta)
    state["_rid"] = repo_id
    state["_rt"] = repo_type

    return (
        status_card(meta_public, warnings_from_meta(meta), repo_type, repo_id),
        install,
        quickstart,
        snapshot,
        cli,
        badge,
        to_files_table(files, limit=250),
        render_risk_html(risk),
        meta_public,
        state,
    )


def do_filter_files(state: dict[str, Any], query: str):
    files = (state or {}).get("_files", []) or []
    return filter_files(files, query, limit=250)


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
    .link:hover{ filter: brightness(1.05); transform: translateY(-0.5px); }

    .pills{ margin-top: 10px; display:flex; gap: 10px; flex-wrap:wrap; }
    .pill{
      display:inline-flex; align-items:center;
      padding: 4px 10px; border-radius: 999px; font-size: .82rem;
      border: 1px solid rgba(148,163,184,.28);
      background: rgba(255,255,255,0.04);
      color: var(--body-text-color);
    }
    .pill.warn{ border-color: rgba(245,158,11,.35); background: rgba(245,158,11,.10); }

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
    @media (max-width: 820px){ .stats{ grid-template-columns: repeat(2, minmax(150px, 1fr)); } }
    @media (max-width: 460px){ .stats{ grid-template-columns: 1fr; } }

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

    dataframe_signature = inspect.signature(gr.Dataframe)
    dataframe_count_kw = (
        {"column_count": (2, "fixed")}
        if "column_count" in dataframe_signature.parameters
        else {"col_count": (2, "fixed")}
    )

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
                user_input = gr.Textbox(
                    label="HF URL or Repo ID",
                    placeholder="google/gemma-2-9b-it or https://huggingface.co/datasets/squad",
                    autofocus=True,
                )
            with gr.Column(scale=2):
                repo_type = gr.Dropdown(["auto", "model", "dataset", "space"], value="auto", label="Type")
            with gr.Column(scale=2):
                generate_button = gr.Button("Generate", variant="primary")

        out_status = gr.HTML(label="Summary")

        with gr.Tabs():
            with gr.TabItem("QuickStart"):
                out_py = gr.Code(language="python", label="Python QuickStart", interactive=False)
                copy_py = gr.Button("Copy")
                out_install = gr.Code(language="shell", label="Install", interactive=False)
                copy_install = gr.Button("Copy")

            with gr.TabItem("Download"):
                out_snapshot = gr.Code(language="python", label="snapshot_download()", interactive=False)
                copy_snapshot = gr.Button("Copy")
                out_cli = gr.Code(language="shell", label="HF CLI download", interactive=False)
                copy_cli = gr.Button("Copy")

            with gr.TabItem("Files"):
                file_query = gr.Textbox(label="Filter", placeholder="e.g. .gguf or config.json")
                files_table = gr.Dataframe(
                    headers=["path", "size"],
                    datatype=["str", "str"],
                    label="Files (first 250)",
                    interactive=False,
                    row_count=10,
                    **dataframe_count_kw,
                )
                risk_html = gr.HTML(label="Risk")

            with gr.TabItem("Export"):
                gr.Markdown(
                    "Exports a zip: `run.py`, `download.py`, `requirements.txt`, `.env.example`, `README.md`."
                )
                zip_button = gr.Button("Build Zip", variant="primary")
                zip_file = gr.File(label="Zip file")
                zip_message = gr.Markdown()

            with gr.TabItem("Badge"):
                out_badge = gr.Code(language="markdown", label="Markdown", interactive=False)
                copy_badge = gr.Button("Copy")

        with gr.Accordion("Details", open=False):
            out_meta = gr.JSON(label="Metadata")

        outputs = [
            out_status,
            out_install,
            out_py,
            out_snapshot,
            out_cli,
            out_badge,
            files_table,
            risk_html,
            out_meta,
            state,
        ]

        generate_button.click(process, inputs=[user_input, repo_type], outputs=outputs)
        user_input.submit(process, inputs=[user_input, repo_type], outputs=outputs)
        file_query.change(do_filter_files, inputs=[state, file_query], outputs=[files_table])
        zip_button.click(build_quickstart_zip, inputs=[state], outputs=[zip_file, zip_message])

        js_copy = "(t)=>{ if(!t){return [];} navigator.clipboard.writeText(String(t)); return []; }"
        copy_install.click(None, inputs=[out_install], outputs=[], js=js_copy)
        copy_py.click(None, inputs=[out_py], outputs=[], js=js_copy)
        copy_snapshot.click(None, inputs=[out_snapshot], outputs=[], js=js_copy)
        copy_cli.click(None, inputs=[out_cli], outputs=[], js=js_copy)
        copy_badge.click(None, inputs=[out_badge], outputs=[], js=js_copy)

    return demo, theme, css


if __name__ == "__main__":
    app, app_theme, app_css = build_ui()
    app.launch(theme=app_theme, css=app_css)
