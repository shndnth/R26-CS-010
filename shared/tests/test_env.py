"""One project .env and the two keys."""

from __future__ import annotations

import pytest
from r26_common import env

KEY = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for name in (env.DATASET_KEY, env.HMAC_KEY, env.ENV_FILE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env, "repository_root", lambda: None)
    project = tmp_path / "project"
    (project / "a" / "b").mkdir(parents=True)
    monkeypatch.chdir(project / "a" / "b")
    return project


def test_env_file_is_found_in_a_parent_folder(clean):
    (clean / ".env").write_text(f"R26_DATASET_AES_KEY={KEY}\nR26_AUDIT_HMAC_KEY=signing\n")
    assert env.dataset_aes_key() == KEY.encode()
    assert env.audit_hmac_key() == b"signing"


def test_environment_wins_over_the_file(clean, monkeypatch):
    (clean / ".env").write_text("R26_AUDIT_HMAC_KEY=from-file\n")
    monkeypatch.setenv(env.HMAC_KEY, "from-environment")
    assert env.audit_hmac_key() == b"from-environment"


def test_explicit_env_file(clean, monkeypatch, tmp_path):
    elsewhere = tmp_path / "keys.env"
    elsewhere.write_text("R26_AUDIT_HMAC_KEY=explicit\n")
    monkeypatch.setenv(env.ENV_FILE_VARIABLE, str(elsewhere))
    assert env.audit_hmac_key() == b"explicit"


def test_repository_root_is_the_fallback(clean, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("R26_AUDIT_HMAC_KEY=from-repo\n")
    monkeypatch.setattr(env, "repository_root", lambda: repo)
    assert env.audit_hmac_key() == b"from-repo"


def test_missing_key_explains_what_to_do(clean):
    with pytest.raises(env.MissingCredentialError, match=r"\.env\.example"):
        env.audit_hmac_key()


def test_dataset_key_must_be_32_bytes(clean, monkeypatch):
    monkeypatch.setenv(env.DATASET_KEY, "short")
    with pytest.raises(env.MissingCredentialError, match="32 bytes"):
        env.dataset_aes_key()


def test_installed_package_finds_this_repository():
    root = env.repository_root()
    assert root is not None and (root / "shared" / "r26_common").is_dir()
