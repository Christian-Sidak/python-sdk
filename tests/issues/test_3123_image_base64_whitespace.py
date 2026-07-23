"""Tests for issue #3123: image content blocks with large base64 data.

``base64.encodebytes()`` inserts a ``\n`` after every 76 output characters and
appends a trailing ``\n``.  These embedded newlines survive a JSON round-trip as
literal ``\n`` characters inside the JSON string value.  Strict RFC 4648
decoders -- such as Go's ``encoding/base64`` used by Claude Code and claude.ai
-- then reject the value with "illegal base64 data at input byte N", even though
the underlying binary payload is valid.

The fix: ``ImageContent.data``, ``AudioContent.data``, and
``BlobResourceContents.blob`` now strip ASCII whitespace on assignment, so the
stored value is always compact, RFC 4648-compliant base64 regardless of how the
caller produced the encoded string.
"""

import base64

import pytest

from mcp.types import AudioContent, BlobResourceContents, ImageContent


@pytest.fixture()
def raw_bytes_small() -> bytes:
    """58 bytes: produces two lines with base64.encodebytes() (line break at char 76)."""
    return bytes(range(58))


@pytest.fixture()
def raw_bytes_large() -> bytes:
    """200 bytes: well past the 99-byte threshold reported in issue #3123."""
    return bytes(range(200))


# ---------------------------------------------------------------------------
# ImageContent
# ---------------------------------------------------------------------------


def test_image_content_b64encode_unchanged(raw_bytes_large: bytes) -> None:
    """Standard b64encode output (no whitespace) passes through unchanged."""
    b64 = base64.b64encode(raw_bytes_large).decode()
    ic = ImageContent(type="image", data=b64, mime_type="image/png")
    assert ic.data == b64


def test_image_content_encodebytes_newlines_stripped(raw_bytes_large: bytes) -> None:
    """base64.encodebytes() newlines are stripped so the stored value is compact."""
    b64_with_newlines = base64.encodebytes(raw_bytes_large).decode()
    assert "\n" in b64_with_newlines, "fixture should contain newlines"

    ic = ImageContent(type="image", data=b64_with_newlines, mime_type="image/png")

    assert "\n" not in ic.data
    assert "\r" not in ic.data
    assert ic.data == base64.b64encode(raw_bytes_large).decode()


def test_image_content_encodebytes_small_stripped(raw_bytes_small: bytes) -> None:
    """Trailing newline from encodebytes() on small data is also stripped."""
    b64_with_newlines = base64.encodebytes(raw_bytes_small).decode()
    ic = ImageContent(type="image", data=b64_with_newlines, mime_type="image/webp")
    assert "\n" not in ic.data
    assert ic.data == base64.b64encode(raw_bytes_small).decode()


def test_image_content_json_has_no_literal_newlines(raw_bytes_large: bytes) -> None:
    """Serialised JSON must not contain literal newlines in the data field.

    A literal newline inside a JSON string value would split the line-delimited
    JSON-RPC stream and corrupt the message framing on the stdio transport.
    """
    b64_with_newlines = base64.encodebytes(raw_bytes_large).decode()
    ic = ImageContent(type="image", data=b64_with_newlines, mime_type="image/png")
    json_str = ic.model_dump_json(by_alias=True, exclude_unset=True)
    assert "\n" not in json_str


def test_image_content_round_trips_correctly(raw_bytes_large: bytes) -> None:
    """Data stored after stripping decodes back to the original bytes."""
    b64_with_newlines = base64.encodebytes(raw_bytes_large).decode()
    ic = ImageContent(type="image", data=b64_with_newlines, mime_type="image/png")
    assert base64.b64decode(ic.data) == raw_bytes_large


# ---------------------------------------------------------------------------
# AudioContent
# ---------------------------------------------------------------------------


def test_audio_content_encodebytes_newlines_stripped(raw_bytes_large: bytes) -> None:
    """Same whitespace-stripping applies to AudioContent.data."""
    b64_with_newlines = base64.encodebytes(raw_bytes_large).decode()
    ac = AudioContent(type="audio", data=b64_with_newlines, mime_type="audio/wav")
    assert "\n" not in ac.data
    assert ac.data == base64.b64encode(raw_bytes_large).decode()


# ---------------------------------------------------------------------------
# BlobResourceContents
# ---------------------------------------------------------------------------


def test_blob_resource_contents_encodebytes_newlines_stripped(raw_bytes_large: bytes) -> None:
    """Same whitespace-stripping applies to BlobResourceContents.blob."""
    b64_with_newlines = base64.encodebytes(raw_bytes_large).decode()
    blob = BlobResourceContents(uri="test://resource", blob=b64_with_newlines)
    assert "\n" not in blob.blob
    assert blob.blob == base64.b64encode(raw_bytes_large).decode()


def test_blob_resource_contents_round_trips_correctly(raw_bytes_large: bytes) -> None:
    b64_with_newlines = base64.encodebytes(raw_bytes_large).decode()
    blob = BlobResourceContents(uri="test://resource", blob=b64_with_newlines)
    assert base64.b64decode(blob.blob) == raw_bytes_large
