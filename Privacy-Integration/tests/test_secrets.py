"""No key material in source."""

from __future__ import annotations


class TestNoHardcodedSecrets:
    # Digests of the retired keys, so this test never contains them.
    RETIRED_KEY_DIGESTS = frozenset({
        "f12a12368f6e924e17fbc99aa778bb75173864e806fd4a71665ceedee59f43e5",
        "b26faeb8198563e29b160df0df0b97c18c573189450c65a68adc0ff79f3569c4",
    })

    def test_source_contains_no_literal_key_material(self):
        import hashlib
        import re

        from privacy_integration.paths import project_root

        literal = re.compile(r"""[\"']([^\"'\n]{16,64})[\"']""")
        offenders = []
        root = project_root()
        sources = [p for folder in ("privacy_integration", "scripts") for p in (root / folder).rglob("*.py")]
        for path in sources:
            for match in literal.finditer(path.read_text(encoding="utf-8")):
                if hashlib.sha256(match.group(1).encode()).hexdigest() in self.RETIRED_KEY_DIGESTS:
                    offenders.append(path.name)
        assert not offenders, f"hardcoded credentials found in: {offenders}"
