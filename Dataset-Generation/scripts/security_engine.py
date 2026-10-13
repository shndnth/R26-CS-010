import hashlib
import os

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

KEY_VARIABLE = "R26_DATASET_AES_KEY"
ENV_FILE_VARIABLE = "R26_ENV_FILE"


def load_project_env():
    """Load the project .env into the environment without overriding it."""
    explicit = os.environ.get(ENV_FILE_VARIABLE)
    candidates = [explicit] if explicit else []
    for start in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        folder = start
        while True:
            candidates.append(os.path.join(folder, ".env"))
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if line and not line.startswith("#") and "=" in line:
                        name, _, value = line.partition("=")
                        os.environ.setdefault(name.strip(), value.strip().strip("'\""))
            return path
    return None


def dataset_key():
    """The 32-byte AES-256 dataset key from the environment or the project .env."""
    load_project_env()
    value = os.environ.get(KEY_VARIABLE)
    if not value:
        raise RuntimeError(
            f"{KEY_VARIABLE} is not set. Copy .env.example at the repository root to .env "
            "and fill it in. Never commit .env."
        )
    key = value.encode()
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be 32 bytes, got {len(key)}")
    return key


class SecurityEngine:
    def __init__(self, key=None):
        # No random fallback: a key nobody holds would make the dataset unrecoverable.
        if key is None:
            key = dataset_key()
        if len(key) != 32:
            raise ValueError(f"AES-256 key must be 32 bytes, got {len(key)}")
        self.key = key

    def generate_sha256(self, file_path):
        """Creates a digital fingerprint for integrity verification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def encrypt_data(self, file_path):
        """Encrypts one file with AES-256-CBC and removes the plaintext."""
        cipher = AES.new(self.key, AES.MODE_CBC)
        with open(file_path, "rb") as f:
            data = f.read()

        encrypted_bytes = cipher.encrypt(pad(data, AES.block_size))

        enc_file_path = file_path + ".enc"
        with open(enc_file_path, "wb") as f:
            f.write(cipher.iv + encrypted_bytes)

        # The plaintext is removed only after the encrypted copy is fully written.
        os.remove(file_path)
        return enc_file_path

    def decrypt_data(self, enc_file_path, remove_encrypted=False):
        """Restores the original file next to the .enc file."""
        try:
            with open(enc_file_path, "rb") as f:
                iv = f.read(16)
                encrypted_data = f.read()

            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            original_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

            original_path = enc_file_path[:-4] if enc_file_path.endswith(".enc") else enc_file_path + ".dec"
            with open(original_path, "wb") as f:
                f.write(original_data)

            if remove_encrypted:
                os.remove(enc_file_path)
            return True
        except (OSError, ValueError) as e:
            print(f"Error during decryption of {enc_file_path}: {e}")
            return False
