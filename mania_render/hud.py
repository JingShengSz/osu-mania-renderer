from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from .models import FrameEvents, HitObject, OsuBeatmap, ReplayData


@dataclass
class HudState:
    combo: int = 0
    max_combo: int = 0
    accuracy: float = 100.0
    clicks_window: list[int] = field(default_factory=list)
    bpm: float = 0.0
    time_s: float = 0.0
    length_s: float = 0.0
    error_ms: float | None = None


def build_fc_sim(bm: OsuBeatmap) -> FrameEvents:
    """Perfect play: every Note/head/tail is a CPS click (ADR-0001)."""
    ev = FrameEvents()
    combo = 0
    pairs: list[tuple[int, int]] = []
    for h in bm.hit_objects:
        ev.clicks.append(h.start_time)  # press = note / LN head only（LN 尾不计 CPS）
        if h.is_hold:
            ev.key_held.setdefault(h.column, []).append((h.start_time, h.end_time))
        else:
            ev.key_held.setdefault(h.column, []).append((h.start_time, h.start_time))
    # Combo trail. An LN contributes 2 (head + tail), a note 1 — but they must be
    # NUMBERED in the order the judgements actually happen, not head-then-tail-immediately.
    #
    # Two bugs lived here. The list was not sorted, and `hud_state_at` stops at the first
    # sample later than the frame, so an LN's far-future tail in the middle froze the
    # counter for the whole hold. And the tail was numbered right after its own head, so
    # notes landing during the hold were numbered after it and the count went *down* when
    # the tail finally arrived. Collect milestones, sort, then number.
    marks: list[int] = []
    for h in bm.hit_objects:
        marks.append(h.start_time)
        if h.is_hold:
            marks.append(h.end_time)
    marks.sort()
    ev.combo_samples = [(t, i + 1) for i, t in enumerate(marks)]
    ev.clicks.sort()
    # FC: every object judged at t=0 offset
    ev.hit_events = []
    ev.hit_pairs = []
    for h in sorted(bm.hit_objects, key=lambda x: x.start_time):
        ev.hit_events.append((h.start_time, 0.0))
        ev.hit_pairs.append((h.start_time, 0.0))
    ev.hit_offsets = [0.0] * len(ev.hit_events)
    return ev


def build_from_replay(bm: OsuBeatmap, replay: ReplayData) -> FrameEvents:
    """CPS from real presses only; combo via perfect-object match against presses."""
    ev = FrameEvents()
    ev.clicks = [p.time_ms for p in replay.presses]

    # held intervals per column from press/release pairs
    for col in range(20):
        downs = [p.time_ms for p in replay.presses if p.column == col]
        ups = [p.time_ms for p in replay.releases if p.column == col]
        holds: list[tuple[int, int]] = []
        di = 0
        ui = 0
        cur: int | None = None
        while di < len(downs) or ui < len(ups):
            nxt_d = downs[di] if di < len(downs) else None
            nxt_u = ups[ui] if ui < len(ups) else None
            if cur is None:
                if nxt_d is None:
                    break
                cur = nxt_d
                di += 1
            else:
                if nxt_u is None or (nxt_d is not None and nxt_d < nxt_u):
                    # next press without release
                    holds.append((cur, cur))
                    cur = nxt_d
                    di += 1
                else:
                    holds.append((cur, nxt_u))
                    cur = None
                    ui += 1
        if cur is not None:
            holds.append((cur, cur))
        if holds:
            ev.key_held[col] = holds

    # combo / hit offsets: nearest press to object start per column
    presses_by_col: dict[int, list[int]] = {}
    for p in replay.presses:
        presses_by_col.setdefault(p.column, []).append(p.time_ms)
    for col in presses_by_col:
        presses_by_col[col].sort()

    combo = 0
    samples: list[tuple[int, int]] = []
    offsets: list[float] = []
    used: set[tuple[int, int]] = set()
    for h in sorted(bm.hit_objects, key=lambda x: x.start_time):
        arr = presses_by_col.get(h.column, [])
        i = bisect_right(arr, h.start_time)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(arr)]
        best = None
        best_d = 1 << 30
        for j in candidates:
            if (h.column, j) in used:
                continue
            d = abs(arr[j] - h.start_time)
            if d < best_d:
                best_d = d
                best = j
        if best is not None and best_d <= 200:
            used.add((h.column, best))
            off = float(arr[best] - h.start_time)
            offsets.append(off)
            ev.hit_events.append((int(arr[best]), off))
            ev.hit_pairs.append((h.start_time, off))
            combo += 1
            samples.append((arr[best], combo))
            if h.is_hold:
                combo += 1
                samples.append((h.end_time, combo))
        else:
            combo = 0
            samples.append((h.start_time, 0))
    ev.combo_samples = samples
    ev.hit_offsets = offsets
    ev.hit_events.sort(key=lambda x: x[0])
    ev.hit_pairs.sort(key=lambda x: x[0])
    return ev


def _bpm_at(bm: OsuBeatmap, t_ms: float) -> float:
    red = [tp for tp in bm.timings if tp.is_red_line and tp.beat_length > 0]
    if not red:
        return 0.0
    red.sort(key=lambda x: x.time)
    chosen = red[0]
    for tp in red:
        if tp.time <= t_ms + 1e-6:
            chosen = tp
        else:
            break
    return float(chosen.bpm)


def hud_state_at(ev: FrameEvents, bm: OsuBeatmap, now_ms: float, length_s: float) -> HudState:
    """SongProgress follows lazer: elapsed song time / full beatmap duration.

    `length_s` is ignored for that readout (kept for API compat); clip range is
    not the denominator — mixing absolute t with clip length is the 35s/30s bug.
    """
    st = HudState()
    song_len = max(0.0, float(getattr(bm, "duration_s", 0.0) or 0.0))
    if song_len <= 0 and bm.hit_objects:
        song_len = max(h.end_time for h in bm.hit_objects) / 1000.0
    st.length_s = song_len
    # absolute position in the song (lead-in frames clamp to 0)
    st.time_s = max(0.0, now_ms / 1000.0)
    if song_len > 0:
        st.time_s = min(st.time_s, song_len)

    combo = 0
    for t, c in ev.combo_samples:
        if t <= now_ms:
            combo = c
        else:
            break
    st.combo = combo
    st.max_combo = combo  # running display; max tracked below
    running = 0
    best = 0
    for t, c in ev.combo_samples:
        if t > now_ms:
            break
        if c == 0:
            running = 0
        else:
            running = c
        if running > best:
            best = running
    st.max_combo = best

    # live BPM from the active red timing point (must change when the map does)
    st.bpm = _bpm_at(bm, now_ms)

    # clicks in last 1000ms (presses / FC clicks)
    win = [t for t in ev.clicks if now_ms - 1000 < t <= now_ms]
    st.clicks_window = win

    # accuracy: osu!mania weighted — (MAX+300)*300 + 200*200 + 100*100 + 50*50 / (total*300)
    counts = judge_counts(bm, ev, now_ms=now_ms)
    total = counts["max"] + counts["300"] + counts["200"] + counts["100"] + counts["50"] + counts["miss"]
    if total > 0:
        weighted = (counts["max"] + counts["300"]) * 300 + counts["200"] * 200 + counts["100"] * 100 + counts["50"] * 50
        st.accuracy = 100.0 * weighted / (300 * total)
    else:
        st.accuracy = 100.0

    if ev.hit_offsets:
        # nearest offset to now for bar meter trail — show last few
        st.error_ms = None
        recent = [o for t, o in zip(
            [h for h in range(len(ev.hit_offsets))],  # placeholder
            ev.hit_offsets,
        )]
        st.error_ms = ev.hit_offsets[-1] if ev.hit_offsets else None

    return st


def recent_errors(ev: FrameEvents, bm: OsuBeatmap, now_ms: float, count: int = 24) -> list[tuple[int, float]]:
    """(time, offset) samples near now for the bar meter."""
    out: list[tuple[int, float]] = []
    for t, off in ev.hit_events:
        if t <= now_ms <= t + 3000:
            out.append((t, off))
    return out[-count:]


def mania_hit_windows(od: float, is_convert: bool = False, classic: bool = False) -> dict[str, float]:
    """ManiaHitWindows half-widths (ms) from OD."""
    import math

    def drange(quality: float, hi: float, mid: float, lo: float) -> float:
        if quality > 5:
            return mid + (lo - mid) * (quality - 5) / 5.0
        if quality < 5:
            return mid + (hi - mid) * (5 - quality) / 5.0
        return mid

    def f05(x: float) -> float:
        return math.floor(x) + 0.5

    if is_convert or classic:
        inv = max(0.0, min(10.0, 10.0 - od))
        return {
            "perfect": f05(16.0),
            "great": f05(34.0 + 3.0 * inv),
            "good": f05(67.0 + 3.0 * inv),
            "ok": f05(97.0 + 3.0 * inv),
            "meh": f05(121.0 + 3.0 * inv),
            "miss": f05(158.0 + 3.0 * inv),
        }
    return {
        "perfect": f05(drange(od, 22.4, 19.4, 13.9)),
        "great": f05(drange(od, 64.0, 49.0, 34.0)),
        "good": f05(drange(od, 97.0, 82.0, 67.0)),
        "ok": f05(drange(od, 127.0, 112.0, 97.0)),
        "meh": f05(drange(od, 151.0, 136.0, 121.0)),
        "miss": f05(drange(od, 188.0, 173.0, 158.0)),
    }


def result_for_offset(offset: float, windows: dict[str, float]) -> str:
    a = abs(offset)
    for name in ("perfect", "great", "good", "ok", "meh"):
        if a <= windows[name]:
            return name
    return "miss"


def judge_counts(bm: OsuBeatmap, ev: FrameEvents, od: float | None = None, now_ms: float | None = None) -> dict[str, int]:
    """Mania judgement tallies (stable names).

    MAX/彩=Perfect, 300/黄=Great, 200=Good, 100=Ok, 50=Meh, 0=Miss.
    now_ms: only count events at or before this time (live panel).
    """
    windows = mania_hit_windows(od if od is not None else (bm.od or 8.0))
    counts = {"max": 0, "300": 0, "200": 0, "100": 0, "50": 0, "miss": 0}
    key = {"perfect": "max", "great": "300", "good": "200", "ok": "100", "meh": "50", "miss": "miss"}
    matched_starts: set[int] = set()
    # pair is (object_start, offset) — one entry per matched object
    for obj_start, off in ev.hit_pairs:
        if now_ms is not None and obj_start > now_ms:
            continue
        matched_starts.add(int(obj_start))
        counts[key[result_for_offset(float(off), windows)]] += 1
    # objects due and never matched to a press → miss
    for h in bm.hit_objects:
        if now_ms is not None and h.start_time > now_ms:
            continue
        if int(h.start_time) not in matched_starts:
            counts["miss"] += 1
    return counts


def unstable_rate(ev: FrameEvents, od: float | None = None, default_od: float = 8.0, now_ms: float | None = None) -> float | None:
    """HitEventExtensions.CalculateUnstableRate — UR = 10 * sqrt(M2/n).

    AffectsUnstableRate: only successful hits (result.IsHit()); misses excluded.
    Welford's online algorithm for population variance. now_ms = live cutoff.
    """
    windows = mania_hit_windows(od if od is not None else default_od)
    n = 0
    mean = 0.0
    m2 = 0.0
    for t, off in ev.hit_events:
        if now_ms is not None and t > now_ms:
            continue
        x = float(off)
        if result_for_offset(x, windows) == "miss":
            continue  # !IsHit()
        n += 1
        next_mean = mean + (x - mean) / n
        m2 += (x - mean) * (x - next_mean)
        mean = next_mean
    if n == 0:
        return None
    return 10.0 * (m2 / n) ** 0.5


def ratio_label(counts: dict[str, int]) -> str:
    """Ratio = MAX/300, 1 decimal. 300==0 → ∞ (n/0, n=MAX)."""
    mx, g = counts.get("max", 0), counts.get("300", 0)
    if g == 0:
        return "∞"
    return f"{mx / g:.1f}"
