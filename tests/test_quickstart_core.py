import ast
import zipfile

import pytest

from quickstart_core import (
    build_export_files,
    cached_public,
    compute_requirements,
    generate_cli_download,
    generate_quickstart,
    generate_snapshot_download,
    get_effective_token,
    is_valid_repo_id,
    parse_hf_input,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("owner/repo", ("model", "owner/repo")),
        ("bert-base-uncased", ("model", "bert-base-uncased")),
        ("datasets/owner/repo", ("dataset", "owner/repo")),
        ("datasets/squad", ("dataset", "squad")),
        ("spaces/owner/repo", ("space", "owner/repo")),
        ("https://huggingface.co/owner/repo", ("model", "owner/repo")),
        ("https://hf.co/owner/repo", ("model", "owner/repo")),
        ("https://huggingface.co/datasets/owner/repo/blob/main/data.csv", ("dataset", "owner/repo")),
        ("https://huggingface.co/spaces/owner/repo/tree/main", ("space", "owner/repo")),
        ("https://huggingface.co/datasets/squad/viewer/plain_text/train", ("dataset", "squad")),
        ("https://huggingface.co/owner/repo/discussions/1", ("model", "owner/repo")),
    ],
)
def test_parse_hf_input(value, expected):
    assert parse_hf_input(value) == expected


@pytest.mark.parametrize(
    "repo_id",
    ["owner/repo", "org-name/model.name", "user_1/repo_2", "bert-base-uncased", "squad"],
)
def test_valid_repo_ids(repo_id):
    assert is_valid_repo_id(repo_id)


@pytest.mark.parametrize(
    "repo_id",
    ["", "/owner/repo", "owner/repo/extra", "owner/..repo", "owner/repo--bad", "owner/.repo"],
)
def test_invalid_repo_ids(repo_id):
    assert not is_valid_repo_id(repo_id)


def test_gguf_quickstart_is_valid_python():
    code = generate_quickstart(
        "model",
        "owner/repo",
        {"_risk": {"has_gguf": True}, "_files": [{"path": "models/model.gguf"}]},
    )
    ast.parse(code)
    assert '"Q: Hello!\\nA:"' in code


def test_snapshot_download_code_avoids_removed_symlink_argument():
    code = generate_snapshot_download("dataset", "owner/repo")
    ast.parse(code)
    assert "local_dir_use_symlinks" not in code
    assert 'repo_type="dataset"' in code


def test_cli_uses_modern_hf_command():
    command = generate_cli_download("model", "owner/repo")
    assert command.startswith("hf download owner/repo")
    assert "huggingface-cli" not in command


def test_requirements_are_type_aware():
    assert compute_requirements("dataset", {}) == ["datasets", "huggingface_hub"]
    assert "llama-cpp-python" in compute_requirements("model", {"_risk": {"has_gguf": True}})
    assert "accelerate" in compute_requirements("model", {"_pipeline_tag": "text-generation"})


def test_export_files_compile_for_gguf():
    files = build_export_files(
        {
            "Repo ID": "owner/repo",
            "Type": "model",
            "_risk": {"has_gguf": True},
            "_files": [{"path": "model.gguf"}],
        }
    )
    ast.parse(files["run.py"])
    ast.parse(files["download.py"])
    assert set(files) == {"README.md", "requirements.txt", ".env.example", "run.py", "download.py"}


def test_export_zip_contract(tmp_path, monkeypatch):
    # Keep tempfile output under pytest tmp path for this test.
    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix: str(tmp_path / prefix.rstrip("_")))
    from quickstart_core import build_quickstart_zip

    zip_path, message = build_quickstart_zip({"Repo ID": "owner/repo", "Type": "dataset"})
    assert zip_path is not None
    assert "Zip built" in message
    with zipfile.ZipFile(zip_path) as archive:
        assert set(archive.namelist()) == {
            "README.md",
            "requirements.txt",
            ".env.example",
            "run.py",
            "download.py",
        }


def test_server_token_requires_allowed_owner(monkeypatch):
    monkeypatch.setenv("ALLOW_SERVER_TOKEN", "1")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.delenv("TOKEN_ALLOWED_OWNERS", raising=False)

    assert get_effective_token("owner/repo") is None

    monkeypatch.setenv("TOKEN_ALLOWED_OWNERS", "owner")
    assert get_effective_token("owner/repo") == "secret-token"


def test_cached_public_only_caches_success(monkeypatch):
    import quickstart_core

    calls = {"count": 0}

    def fake_fetch(repo_type, repo_id, token):
        calls["count"] += 1
        if calls["count"] == 1:
            return False, None, "temporary error"
        return True, {"Repo ID": repo_id, "Type": repo_type}, None

    quickstart_core._PUBLIC_CACHE.clear()
    monkeypatch.setattr(quickstart_core, "fetch_repo_info", fake_fetch)

    first = cached_public("model", "owner/repo")
    second = cached_public("model", "owner/repo")
    third = cached_public("model", "owner/repo")

    assert first[0] is False
    assert second[0] is True
    assert third[0] is True
    assert calls["count"] == 2
