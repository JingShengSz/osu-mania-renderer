from __future__ import annotations

import lzma
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .models import ReplayData, ReplayPress


def _read_uleb128(buf: memoryview, off: int) -> tuple[int, int]:
    n = 0
    shift = 0
    while True:
        if off >= len(buf):
            raise ValueError("eof uleb")
        b = buf[off]
        off += 1
        n |= (b & 0x7F) << shift
        if not (b & 0x80):
            return n, off
        shift += 7
        if shift > 35:
            raise ValueError("uleb overflow")


def _read_string(buf: memoryview, off: int) -> tuple[str | None, int]:
    """osr string: 0x00 = empty, 0x0b + ULEB128 length + UTF-8 (SerializationReader)."""
    if off >= len(buf):
        raise ValueError("eof string")
    marker = buf[off]
    off += 1
    if marker == 0:
        return None, off
    # BinaryWriter / osu SerializationReader: 0x0b then 7-bit/ULEB128 length
    n, off = _read_uleb128(buf, off)
    if n < 0 or off + n > len(buf):
        raise ValueError(f"eof string body n={n}")
    raw = bytes(buf[off : off + n])
    off += n
    return raw.decode("utf-8", "replace"), off


def _read_bytes(buf: memoryview, off: int) -> tuple[bytes | None, int]:
    if off + 4 > len(buf):
        raise ValueError("eof bytes len")
    (n,) = struct.unpack_from("<i", buf, off)
    off += 4
    if n < 0:
        return None, off
    if n == 0:
        return b"", off
    raw = bytes(buf[off : off + n])
    off += n
    return raw, off


@dataclass
class OsrHeader:
    ruleset_id: int = 0
    version: int = 0
    beatmap_md5: str = ""
    username: str = ""
    replay_bytes: bytes | None = field(default=None)


def parse_osr(path: str | Path) -> tuple[OsrHeader, ReplayData]:
    data = Path(path).read_bytes()
    buf = memoryview(data)
    off = 0

    head = OsrHeader()
    head.ruleset_id = buf[off]
    off += 1
    (head.version,) = struct.unpack_from("<i", buf, off)
    off += 4
    head.beatmap_md5, off = _read_string(buf, off)
    head.username, off = _read_string(buf, off)
    _, off = _read_string(buf, off)  # replay hash
    # judgement counts
    off += 2 * 6
    off += 4  # score
    off += 2  # max combo
    off += 1  # perfect
    off += 4  # mods
    _, off = _read_string(buf, off)  # life bar
    off += 8  # timestamp
    blob, off = _read_bytes(buf, off)
    head.replay_bytes = blob

    if head.ruleset_id != 3:
        raise ValueError(f"replay ruleset {head.ruleset_id} is not mania (3)")

    replay = ReplayData()
    if not blob:
        return head, replay

    # 5-byte LZMA props + 8-byte uncompressed size + payload
    raw = lzma.decompress(blob, format=lzma.FORMAT_ALONE)
    text = raw.decode("utf-8", "replace")
    t = 0
    prev_mask = 0
    frames = text.split(",")
    for fr in frames:
        fr = fr.strip()
        if not fr:
            continue
        parts = fr.split("|")
        if len(parts) < 2:
            continue
        try:
            w = int(float(parts[0]))
            x = int(float(parts[1]))
        except ValueError:
            continue
        if w == -12345:
            continue
        # drop stable sentinel
        y = int(float(parts[2])) if len(parts) > 2 else 0
        if x == 256 and y == -500:
            continue
        t += w
        mask = x & 0xFFFFF
        for bit in range(20):
            prev = (prev_mask >> bit) & 1
            cur = (mask >> bit) & 1
            if prev == 0 and cur == 1:
                replay.presses.append(ReplayPress(t, bit))
            elif prev == 1 and cur == 0:
                replay.releases.append(ReplayPress(t, bit))
        prev_mask = mask

    replay.presses.sort(key=lambda p: p.time_ms)
    replay.releases.sort(key=lambda p: p.time_ms)
    return head, replay
