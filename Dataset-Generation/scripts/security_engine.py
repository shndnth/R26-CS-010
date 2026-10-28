"""AES-256 encryption and SHA-256 hashing for the CARLA capture.

Two file formats are supported:

    cbc (default)  [16-byte IV][AES-256-CBC ciphertext, PKCS7 padded]
    gcm            [16-byte nonce][16-byte tag][AES-256-GCM ciphertext]

CBC is the format of the delivered dataset and the one Utility Evaluation
decrypts. GCM is authenticated: a changed byte makes decryption fail instead of
returning corrupted data. Select it with ``--cipher gcm`` in capture_town.py.
"""

import hashlib
import os

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

KEY_VARIABLE = "R26_DATASET_AES_KEY"
ENV_FILE_VARIABLE = "R26_ENV_FILE"

CIPHER_MODES = ("cbc", "gcm")
IV_BYTES = 16
NONCE_BYTES = 16
TAG_BYTES = 16


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
    def __init__(self, key=None, mode="cbc"):
        # No random fallback: a key nobody holds would make the dataset unrecoverable.
        if key is None:
            key = dataset_key()
        if len(key) != 32:
            raise ValueError(f"AES-256 key must be 32 bytes, got {len(key)}")
        if mode not in CIPHER_MODES:
            raise ValueError(f"cipher mode must be one of {CIPHER_MODES}, got {mode!r}")
        self.key = key
        self.mode = mode

    @property
    def label(self):
        return f"AES-256-{self.mode.upper()}"

    def generate_sha256(self, file_path):
        """Creates a digital fingerprint for integrity verification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    # In-memory operations, used by the file methods and by the security audit.

    def encrypt_bytes(self, data):
        """Encrypt bytes into the on-disk format of the selected mode."""
        if self.mode == "gcm":
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=os.urandom(NONCE_BYTES))
            ciphertext, tag = cipher.encrypt_and_digest(data)
            return cipher.nonce + tag + ciphertext
        cipher = AES.new(self.key, AES.MODE_CBC, iv=os.urandom(IV_BYTES))
        return cipher.iv + cipher.encrypt(pad(data, AES.block_size))

    def decrypt_bytes(self, blob):
        """Decrypt bytes in the selected format. Raises ValueError if they do not decrypt.

        In GCM mode a changed byte anywhere raises ValueError (tag check failed).
        """
        if self.mode == "gcm":
            nonce = blob[:NONCE_BYTES]
            tag = blob[NONCE_BYTES:NONCE_BYTES + TAG_BYTES]
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
            return cipher.decrypt_and_verify(blob[NONCE_BYTES + TAG_BYTES:], tag)
        cipher = AES.new(self.key, AES.MODE_CBC, blob[:IV_BYTES])
        return unpad(cipher.decrypt(blob[IV_BYTES:]), AES.block_size)

    def header_bytes(self):
        """Bytes before the ciphertext: the IV (CBC) or nonce and tag (GCM)."""
        return NONCE_BYTES + TAG_BYTES if self.mode == "gcm" else IV_BYTES

    # File operations used by the capture.

    def encrypt_data(self, file_path):
        """Encrypts one file and removes the plaintext."""
        with open(file_path, "rb") as f:
            data = f.read()

        enc_file_path = file_path + ".enc"
        with open(enc_file_path, "wb") as f:
            f.write(self.encrypt_bytes(data))

        # The plaintext is removed only after the encrypted copy is fully written.
        os.remove(file_path)
        return enc_file_path

    def decrypt_data(self, enc_file_path, remove_encrypted=False):
        """Restores the original file next to the .enc file."""
        try:
            with open(enc_file_path, "rb") as f:
                original_data = self.decrypt_bytes(f.read())

            original_path = enc_file_path[:-4] if enc_file_path.endswith(".enc") else enc_file_path + ".dec"
            with open(original_path, "wb") as f:
                f.write(original_data)

            if remove_encrypted:
                os.remove(enc_file_path)
            return True
        except (OSError, ValueError) as e:
            print(f"Error during decryption of {enc_file_path}: {e}")
            return False
