"""Encryption, key loading and access control used by the capture script."""

from __future__ import annotations

import pytest
import rbac_gateway
import security_engine
from conftest import TEST_KEY
from security_engine import SecurityEngine


@pytest.fixture
def no_key(monkeypatch, tmp_path):
    monkeypatch.delenv(security_engine.KEY_VARIABLE, raising=False)
    monkeypatch.setenv(security_engine.ENV_FILE_VARIABLE, str(tmp_path / "absent.env"))
    monkeypatch.chdir(tmp_path)


def test_encrypt_then_decrypt_round_trip_keeps_the_encrypted_file(tmp_path):
    engine = SecurityEngine(key=TEST_KEY.encode())
    original = tmp_path / "rgb_f000001.png"
    original.write_bytes(b"frame" * 100)
    digest = engine.generate_sha256(str(original))

    encrypted = engine.encrypt_data(str(original))
    assert not original.exists()

    assert engine.decrypt_data(encrypted)
    assert (tmp_path / "rgb_f000001.png.enc").exists()
    assert engine.generate_sha256(str(original)) == digest


def test_missing_key_is_an_error_not_a_random_key(no_key):
    with pytest.raises(RuntimeError, match=r"\.env"):
        SecurityEngine()


def test_key_must_be_32_bytes():
    with pytest.raises(ValueError, match="32 bytes"):
        SecurityEngine(key=b"short")


def test_key_is_read_from_the_project_env_file(no_key, monkeypatch, tmp_path):
    env_file = tmp_path / "project.env"
    env_file.write_text(f"R26_DATASET_AES_KEY={TEST_KEY}\n")
    monkeypatch.setenv(security_engine.ENV_FILE_VARIABLE, str(env_file))
    assert security_engine.dataset_key() == TEST_KEY.encode()


def test_env_file_in_a_parent_folder_is_found(no_key, monkeypatch, tmp_path):
    monkeypatch.delenv(security_engine.ENV_FILE_VARIABLE)
    (tmp_path / ".env").write_text(f"R26_DATASET_AES_KEY={TEST_KEY}\n")
    nested = tmp_path / "carla" / "PythonAPI" / "examples"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert security_engine.dataset_key() == TEST_KEY.encode()


def test_role_permissions():
    gateway = rbac_gateway.RBACGateway()
    assert gateway.verify_access("Simulation_Engineer", "create")
    assert not gateway.verify_access("ML_Researcher", "create")
