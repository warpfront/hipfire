#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""CPU-only proof that a PM module's `.text` code-end tail is the ONLY change.

Compares an OLD and a NEW checked-in PM module (`*.hxaco` clang offload bundles
or plain ELF code objects such as `*.co`) and proves that NEW is OLD plus

  * alignment padding of the final `.text` up to a 128-byte instruction-cache
    line, then
  * exactly 96 little-endian dwords `0xbf9f0000` (`s_code_end`, 3 cache lines),

and nothing else except the ELF size/offset/address fields that growing `.text`
mathematically forces.  Anything else is a failure; a descriptor or kernel that
moved is reported as a genuine relocation, never silently allowed.

What is proven per ELF (stdlib only, no GPU, no toolchain):

  1. every byte of OLD `[0, .text offset + .text size)` is byte-identical in NEW
     (so all old `.text` bytes are a prefix of NEW `.text`, and every kernel code
     symbol, every 64-byte `.kd` descriptor, `.rodata`, `.note`, the dynamic
     symbol/hash/string tables and all inter-section gaps are unchanged), except
     the ELF header `e_shoff` and the program header table, which are compared
     field by field instead;
  2. the tail NEW `.text[old size:]` is `(-old size) % 128` alignment bytes of
     0xbf9f0000 followed by 96 x 0xbf9f0000;
  3. every kernel `FUNC` symbol and `.kd` `OBJECT` symbol has the same value and
     size in OLD and NEW, and its bytes are compared individually and reported;
  4. every section after `.text` (`.dynamic`, `.comment`, `.symtab`, `.shstrtab`,
     `.strtab`, ...) has a byte-identical payload at a uniform file-offset shift,
     the only exception being `.symtab` `st_value` fields of symbols that live in
     a shifted section (`_DYNAMIC`), which must equal OLD + that section's
     address shift;
  5. section headers match field by field: only `.text` `sh_size`, the offsets and
     addresses of the sections after `.text` (uniform shifts), and the NOBITS
     `.relro_padding` size may differ; link/info/flags/align/entsize never;
  6. program headers match field by field: the `.text` LOAD segment grows with
     `.text`, the read-write LOAD / DYNAMIC / GNU_RELRO segments follow the
     shifted `.dynamic` (and the RELRO padding), everything else is identical;
  7. layout rules the OLD file satisfies are re-checked on NEW: the lld
     segment-start placement of the first section after `.text`, the page-end
     rule of `.relro_padding`, `p_offset % p_align == p_vaddr % p_align` for
     every LOAD, `e_shoff` 8-byte aligned and the section header table at the
     file end;
  8. for `.hxaco` bundles the container is compared too: same entry ids and
     offsets, only the ELF entries' size fields differ, host entries stay empty,
     gaps between entries are zero.

Per file it reports old/new `.text` bytes, bytes of room to the 4 KiB page end
before/after, the fill actually found, and the kernel/descriptor verification.

Usage (exit 0 = every file proven, 1 = any file failed or unmatched, 2 = usage):

  # old/new roots: every `*.hxaco` / `*.co` / `*.hsaco` whose stem has a `pm`
  # token (`mq4_rows_x4_pm_gfx1151`, `gemm_hfq4g256_rows_pm`) under both roots,
  # matched by relative path; NEW defaults to <repo>/kernels.
  scripts/check-pm-code-end.py --old-root /path/to/old/kernels
  scripts/check-pm-code-end.py --old-root OLD --new-root NEW [--include REGEX]

  # explicit file pairs (repeatable)
  scripts/check-pm-code-end.py --pair OLD.hxaco NEW.hxaco [--pair ...]

Options: `--include REGEX` (stem filter for roots; default matches a `pm` token),
`--json PATH` (machine-readable report), `-q` (print failures and the summary only),
`--self-test` (run the built-in accept/reject tests on synthetic code objects).
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import unittest
from pathlib import Path

PAGE = 0x1000
CACHE_LINE = 128
CODE_END_DWORDS = 96
S_CODE_END = 0xBF9F0000
_S_NOP = 0xBF800000
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
DEFAULT_INCLUDE = r"(?:^|[_-])pm(?:[_.-]|$)"
SUFFIXES = (".hxaco", ".co", ".hsaco")

SHT_NULL, SHT_SYMTAB, SHT_NOBITS, SHT_DYNSYM = 0, 2, 8, 11
SHF_ALLOC, SHF_EXEC = 0x2, 0x4
PT_LOAD = 1
PT_GNU_RELRO = 0x6474E552
SHN_LORESERVE = 0xFF00
STT_OBJECT, STT_FUNC = 1, 2
KD_SIZE = 64

EHDR_SIZE = 64
EHDR_SHOFF = (0x28, 0x30)


class FormatError(Exception):
    pass


def align(x: int, a: int) -> int:
    return x if a <= 1 else (x + a - 1) // a * a


def align_skew(x: int, a: int, skew: int) -> int:
    """Smallest y >= x with y % a == skew % a."""
    base = x - x % a + skew % a
    return base if base >= x else base + a


def segment_start(dot: int, section_align: int) -> int:
    """lld: address of a section that starts a new PT_LOAD."""
    return align(align(dot, PAGE) + dot % PAGE, section_align)


def room_to_page_end(addr_end: int) -> int:
    return (-addr_end) % PAGE


# ---------------------------------------------------------------------------
# ELF parsing


class Section:
    __slots__ = ("idx", "name", "name_idx", "type", "flags", "addr", "offset", "size",
                 "link", "info", "addralign", "entsize")

    def __init__(self, idx, fields):
        (self.name_idx, self.type, self.flags, self.addr, self.offset, self.size,
         self.link, self.info, self.addralign, self.entsize) = fields
        self.idx = idx
        self.name = ""

    def fields(self):
        return (self.name_idx, self.type, self.flags, self.addr, self.offset, self.size,
                self.link, self.info, self.addralign, self.entsize)

    @property
    def nobits(self):
        return self.type == SHT_NOBITS

    @property
    def alloc(self):
        return bool(self.flags & SHF_ALLOC)


class Elf:
    def __init__(self, data: bytes, label: str):
        self.data = data
        self.label = label
        if len(data) < EHDR_SIZE or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
            raise FormatError(f"{label}: not an ELF64 little-endian object")
        (self.e_type, self.e_machine, _, self.e_entry, self.e_phoff, self.e_shoff, self.e_flags,
         self.e_ehsize, self.e_phentsize, self.e_phnum, self.e_shentsize, self.e_shnum,
         self.e_shstrndx) = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
        if self.e_shentsize != 64 or self.e_phentsize != 56:
            raise FormatError(f"{label}: unexpected e_shentsize/e_phentsize")
        if self.e_phoff + self.e_phnum * 56 > len(data) or self.e_shoff + self.e_shnum * 64 > len(data):
            raise FormatError(f"{label}: header tables run past the end of the file")
        self.phdrs = [struct.unpack_from("<IIQQQQQQ", data, self.e_phoff + 56 * i) for i in range(self.e_phnum)]
        self.sections = [Section(i, struct.unpack_from("<IIQQQQIIQQ", data, self.e_shoff + 64 * i))
                         for i in range(self.e_shnum)]
        if self.e_shnum and self.e_shstrndx < self.e_shnum:
            strs = self.payload(self.sections[self.e_shstrndx])
            for s in self.sections:
                s.name = cstr(strs, s.name_idx)
        for s in self.sections:
            if not s.nobits and s.type != SHT_NULL and s.offset + s.size > len(data):
                raise FormatError(f"{label}: section {s.name or s.idx} runs past the end of the file")

    def payload(self, s: Section) -> bytes:
        return b"" if s.nobits else self.data[s.offset:s.offset + s.size]

    def symbols(self, s: Section):
        strs = self.payload(self.sections[s.link])
        out = []
        raw = self.payload(s)
        for i in range(len(raw) // 24):
            name, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", raw, 24 * i)
            out.append((cstr(strs, name), info, other, shndx, value, size))
        return out

    def section_at(self, off: int) -> str:
        for s in self.sections:
            if s.type != SHT_NULL and not s.nobits and s.offset <= off < s.offset + max(s.size, 1):
                return s.name
        if off < EHDR_SIZE:
            return "ELF header"
        if self.e_phoff <= off < self.e_phoff + self.e_phnum * 56:
            return "program headers"
        if self.e_shoff <= off < self.e_shoff + self.e_shnum * 64:
            return "section headers"
        return "<gap>"


def cstr(buf: bytes, off: int) -> str:
    if off >= len(buf):
        return ""
    end = buf.find(b"\0", off)
    return buf[off:end if end >= 0 else len(buf)].decode("ascii", "replace")


# ---------------------------------------------------------------------------
# Containers (clang offload bundle or bare ELF)


class Entry:
    def __init__(self, ident, offset, size, size_pos):
        self.ident, self.offset, self.size, self.size_pos = ident, offset, size, size_pos


class Module:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        self.header_end = 0
        if self.data[:len(BUNDLE_MAGIC)] == BUNDLE_MAGIC:
            self.kind = "bundle"
            self.entries = self._parse_bundle()
        elif self.data[:4] == b"\x7fELF":
            self.kind = "elf"
            self.entries = [Entry("elf", 0, len(self.data), None)]
        else:
            raise FormatError(f"{path}: neither a clang offload bundle nor an ELF")

    def _parse_bundle(self):
        d = self.data
        if len(d) < 32:
            raise FormatError(f"{self.path}: truncated offload bundle header")
        (n,) = struct.unpack_from("<Q", d, 24)
        pos, entries = 32, []
        for _ in range(n):
            if pos + 24 > len(d):
                raise FormatError(f"{self.path}: truncated offload bundle entry table")
            off, size, idlen = struct.unpack_from("<QQQ", d, pos)
            ident = d[pos + 24:pos + 24 + idlen].decode("ascii", "replace")
            entries.append(Entry(ident, off, size, pos + 8))
            pos += 24 + idlen
            if pos > len(d):
                raise FormatError(f"{self.path}: truncated offload bundle entry id")
        for e in entries:
            if e.size and e.offset + e.size > len(d):
                raise FormatError(f"{self.path}: bundle entry {e.ident!r} runs past the end of the file")
        self.header_end = pos
        return entries

    def payload(self, e: Entry) -> bytes:
        return self.data[e.offset:e.offset + e.size]


# ---------------------------------------------------------------------------
# Comparison helpers


def hexs(v: int) -> str:
    return f"0x{v:x}"


def diff_runs(a: bytes, b: bytes, limit: int = 4):
    """Runs of differing bytes between equal-length a and b: [(start, end)]."""
    runs, i, n = [], 0, len(a)
    while i < n and len(runs) < limit:
        if a[i] == b[i]:
            i += 1
            continue
        j = i
        while j < n and a[j] != b[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def masked(data: bytes, base: int, masks) -> bytes:
    """Copy of data with every (abs_start, abs_end) mask zeroed; base = abs offset of data[0]."""
    out = bytearray(data)
    for s, e in masks:
        s, e = max(s - base, 0), min(e - base, len(out))
        if s < e:
            out[s:e] = bytes(e - s)
    return bytes(out)


def compare_region(errors, what, old_bytes, new_bytes, old_base, new_base, old_elf, new_elf,
                   old_masks=(), new_masks=()):
    if len(old_bytes) != len(new_bytes):
        errors.append(f"{what}: length {len(old_bytes)} -> {len(new_bytes)} (must be equal)")
        return
    a = masked(old_bytes, old_base, old_masks)
    b = masked(new_bytes, new_base, new_masks)
    if a == b:
        return
    runs = diff_runs(a, b, 4)
    total = sum(1 for x, y in zip(a, b) if x != y)
    parts = [f"{hexs(old_base + s)}..{hexs(old_base + e)} old({old_elf.section_at(old_base + s)}) -> "
             f"{hexs(new_base + s)} new({new_elf.section_at(new_base + s)}), {e - s} B" for s, e in runs]
    errors.append(f"{what}: {total} byte(s) differ (genuine change, not a derived field): " + "; ".join(parts))


def all_zero(b: bytes) -> bool:
    return not any(b)


# ---------------------------------------------------------------------------
# ELF comparison


def compare_elf(old: Elf, new: Elf) -> dict:
    errors: list[str] = []
    notes: list[str] = []
    rep: dict = {"errors": errors, "notes": notes}

    def text_of(elf: Elf, which: str):
        cands = [s for s in elf.sections if s.name == ".text" and s.flags & SHF_EXEC]
        if len(cands) != 1:
            raise FormatError(f"{which}: expected exactly one executable .text, found {len(cands)}")
        return cands[0]

    ot, nt = text_of(old, "old"), text_of(new, "new")
    rep.update(text_old=ot.size, text_new=nt.size, text_addr=ot.addr)

    # ---- ELF header (e_shoff is derived; checked via the file layout below) ----
    oh, nh = old.data[:EHDR_SIZE], new.data[:EHDR_SIZE]
    hm = bytearray(oh), bytearray(nh)
    for h in hm:
        h[EHDR_SHOFF[0]:EHDR_SHOFF[1]] = bytes(8)
    if hm[0] != hm[1]:
        errors.append("ELF header differs outside e_shoff at bytes " +
                      ", ".join(f"{hexs(s)}..{hexs(e)}" for s, e in diff_runs(bytes(hm[0]), bytes(hm[1]))))
    if new.e_shoff % 8:
        errors.append(f"new e_shoff {hexs(new.e_shoff)} is not 8-byte aligned")

    # ---- section table shape ----
    if old.e_shnum != new.e_shnum:
        errors.append(f"section count {old.e_shnum} -> {new.e_shnum}")
        return rep
    text_end_old = ot.offset + ot.size
    text_end_new = nt.offset + nt.size
    if ot.offset != nt.offset or ot.addr != nt.addr:
        errors.append(f".text start moved: offset {hexs(ot.offset)} -> {hexs(nt.offset)}, "
                      f"addr {hexs(ot.addr)} -> {hexs(nt.addr)} (genuine relocation of kernels/descriptors)")
    if ot.fields()[:3] != nt.fields()[:3] or ot.fields()[6:] != nt.fields()[6:]:
        errors.append(".text header fields other than size/addr/offset changed")

    # classify sections: pre (before .text), post (after), text
    cls: dict[int, str] = {}
    for s in old.sections:
        if s.type == SHT_NULL:
            cls[s.idx] = "null"
        elif s is ot:
            cls[s.idx] = "text"
        elif s.nobits:
            cls[s.idx] = "post" if s.alloc and s.addr >= ot.addr + ot.size else "pre"
        elif s.offset >= text_end_old:
            cls[s.idx] = "post"
        elif s.offset + s.size <= ot.offset:
            cls[s.idx] = "pre"
        else:
            cls[s.idx] = "overlap"
            errors.append(f"old section {s.name} overlaps .text")
    if any(v == "overlap" for v in cls.values()):
        return rep

    post_file = [s for s in old.sections if cls[s.idx] == "post" and not s.nobits]
    if not post_file:
        errors.append("no file-resident section after .text: layout not supported by this checker")
        return rep
    first_old = min(post_file, key=lambda s: s.offset)
    first_new = new.sections[first_old.idx]
    d_file = first_new.offset - first_old.offset
    post_alloc = [s for s in old.sections if cls[s.idx] == "post" and s.alloc]
    d_addr = (new.sections[post_alloc[0].idx].addr - post_alloc[0].addr) if post_alloc else 0
    rep.update(shift_file=d_file, shift_addr=d_addr)
    addr_delta: dict[int, int] = {}
    relro_old = relro_new = 0

    for s in old.sections:
        n = new.sections[s.idx]
        c = cls[s.idx]
        nm = s.name or f"#{s.idx}"
        if (s.name_idx, s.type, s.flags, s.link, s.info, s.addralign, s.entsize) != \
                (n.name_idx, n.type, n.flags, n.link, n.info, n.addralign, n.entsize):
            errors.append(f"section {nm}: name/type/flags/link/info/align/entsize changed")
        addr_delta[s.idx] = 0
        if c == "null":
            if s.fields() != n.fields():
                errors.append("null section header changed")
        elif c == "pre":
            if (s.addr, s.offset, s.size) != (n.addr, n.offset, n.size):
                errors.append(f"section {nm} (before .text): addr/offset/size moved "
                              f"{hexs(s.addr)}/{hexs(s.offset)}/{hexs(s.size)} -> "
                              f"{hexs(n.addr)}/{hexs(n.offset)}/{hexs(n.size)}")
        elif c == "post":
            if n.offset - s.offset != d_file:
                errors.append(f"section {nm}: file offset shift {hexs(n.offset - s.offset)} != uniform "
                              f"shift {hexs(d_file)}")
            if s.alloc:
                addr_delta[s.idx] = n.addr - s.addr
                if n.addr - s.addr != d_addr:
                    errors.append(f"section {nm}: address shift {hexs(n.addr - s.addr)} != uniform "
                                  f"shift {hexs(d_addr)}")
            elif s.addr != n.addr:
                errors.append(f"section {nm}: non-alloc sh_addr changed")
            if s.nobits:
                relro_old += s.size
                relro_new += n.size
                if (n.addr + n.size) % PAGE != 0 and (s.addr + s.size) % PAGE == 0:
                    errors.append(f"NOBITS section {nm}: no longer ends on a page boundary "
                                  f"(addr {hexs(n.addr)} + size {hexs(n.size)})")
                if n.size >= PAGE:
                    errors.append(f"NOBITS section {nm}: padding {n.size} B >= one page")
            elif s.size != n.size:
                errors.append(f"section {nm}: size {s.size} -> {n.size} (only .text may grow)")
    if first_old.alloc:
        want_old = segment_start(ot.addr + ot.size, first_old.addralign)
        if first_old.addr == want_old:
            want_new = segment_start(ot.addr + nt.size, first_old.addralign)
            if first_new.addr != want_new:
                errors.append(f"{first_old.name}: new address {hexs(first_new.addr)} is not the lld "
                              f"segment start {hexs(want_new)} after the grown .text")
            want_off = align_skew(text_end_new, PAGE, first_new.addr)
            if first_new.offset != want_off:
                errors.append(f"{first_old.name}: new file offset {hexs(first_new.offset)} is not the "
                              f"smallest page-congruent offset {hexs(want_off)}")
        else:
            notes.append(f"old {first_old.name} placement does not follow the lld segment-start rule; "
                         "not re-derived")
    if first_new.offset < text_end_new:
        errors.append(f"new {first_new.name} (offset {hexs(first_new.offset)}) overlaps grown .text "
                      f"(ends {hexs(text_end_new)})")

    # ---- program headers ----
    if old.e_phnum != new.e_phnum:
        errors.append(f"program header count {old.e_phnum} -> {new.e_phnum}")
    else:
        for i, (po, pn) in enumerate(zip(old.phdrs, new.phdrs)):
            ptype, pflags, poff, pva, ppa, pfsz, pmsz, palign = po
            exp = list(po)
            if poff == ot.offset and pva == ot.addr and pfsz == ot.size:
                exp[5] = exp[6] = nt.size + (pmsz - pfsz)
            elif poff == first_old.offset and (not first_old.alloc or pva == first_old.addr):
                exp[2] = first_new.offset
                exp[3] = pva + d_addr
                exp[4] = ppa + d_addr
                if pmsz - pfsz == relro_old and ptype in (PT_LOAD, PT_GNU_RELRO):
                    exp[6] = pfsz + relro_new
            if tuple(exp) != pn:
                names = ("type", "flags", "offset", "vaddr", "paddr", "filesz", "memsz", "align")
                bad = [f"{nm}: {hexs(o)} -> {hexs(v)} (expected {hexs(e)})"
                       for nm, o, v, e in zip(names, po, pn, exp) if v != e]
                errors.append(f"program header {i} (type {hexs(ptype)}): " + ", ".join(bad))
        for i, pn in enumerate(new.phdrs):
            if pn[0] == PT_LOAD and pn[7] > 1 and pn[2] % pn[7] != pn[3] % pn[7]:
                errors.append(f"new LOAD {i}: p_offset {hexs(pn[2])} and p_vaddr {hexs(pn[3])} not congruent "
                              f"modulo p_align {hexs(pn[7])}")
            if pn[0] == PT_LOAD and pn[5] > pn[6]:
                errors.append(f"new LOAD {i}: filesz > memsz")
        if not any(pn[0] == PT_LOAD and pn[1] & 1 and pn[3] <= nt.addr and nt.addr + nt.size <= pn[3] + pn[6]
                   for pn in new.phdrs):
            errors.append("no executable LOAD segment covers the whole new .text")

    # ---- file layout: head, text tail, gap, post region, section headers ----
    for which, elf in (("old", old), ("new", new)):
        if elf.e_shoff < max(s.offset + s.size for s in elf.sections if not s.nobits):
            errors.append(f"{which}: section header table is not after all section payloads")
        if len(elf.data) != elf.e_shoff + elf.e_shnum * 64:
            errors.append(f"{which}: file is {len(elf.data)} B but headers end at "
                          f"{elf.e_shoff + elf.e_shnum * 64}")
    head = text_end_old
    ph_mask_o = [EHDR_SHOFF, (old.e_phoff, old.e_phoff + old.e_phnum * 56)]
    ph_mask_n = [EHDR_SHOFF, (new.e_phoff, new.e_phoff + new.e_phnum * 56)]
    compare_region(errors, "bytes [0, old .text end)", old.data[:head], new.data[:head], 0, 0, old, new,
                   ph_mask_o, ph_mask_n)

    # ---- .text tail ----
    old_text, new_text = old.payload(ot), new.payload(nt)
    tail = new_text[len(old_text):]
    align_len = (-ot.size) % CACHE_LINE
    want_len = align_len + 4 * CODE_END_DWORDS
    rep.update(align_bytes=align_len, tail_bytes=len(tail))
    if nt.size < ot.size or new_text[:len(old_text)] != old_text:
        errors.append("old .text is not a byte-identical prefix of new .text")
    if len(tail) != want_len or len(tail) % 4:
        errors.append(f".text grew by {len(tail)} B, expected {align_len} B alignment + "
                      f"{CODE_END_DWORDS} dwords = {want_len} B")
    if nt.size % CACHE_LINE or (nt.addr + nt.size) % CACHE_LINE:
        errors.append(f"new .text end (size {nt.size}, end addr {hexs(nt.addr + nt.size)}) is not "
                      f"{CACHE_LINE}-byte aligned")
    words = struct.unpack(f"<{len(tail) // 4}I", tail[:len(tail) // 4 * 4])
    pad_words, end_words = words[:align_len // 4], words[align_len // 4:]
    if align_len % 4:
        errors.append(f"old .text size {ot.size} is not a whole number of dwords")
    if any(w != S_CODE_END for w in pad_words):
        bad = sorted({hexs(w) for w in pad_words if w != S_CODE_END})
        errors.append(f"alignment padding is not all {hexs(S_CODE_END)}: found {', '.join(bad)}")
    if len(end_words) != CODE_END_DWORDS or any(w != S_CODE_END for w in end_words):
        errors.append(f"final {CODE_END_DWORDS} dwords are not all {hexs(S_CODE_END)}: "
                      f"{sum(1 for w in end_words if w == S_CODE_END)}/{len(end_words)} match")
    rep["fill"] = (f"{len(pad_words)} dword(s) {hexs(S_CODE_END)} alignment + "
                   f"{sum(1 for w in end_words if w == S_CODE_END)}/{CODE_END_DWORDS} dwords {hexs(S_CODE_END)}")

    # ---- room to page end ----
    end_old, end_new = ot.addr + ot.size, nt.addr + nt.size
    rep.update(room_old=room_to_page_end(end_old), room_new=room_to_page_end(end_new),
               end_old=end_old, end_new=end_new)
    if (end_new - 1) // PAGE != (end_old - 1) // PAGE:
        notes.append(f".text end crossed a page boundary ({hexs(end_old)} -> {hexs(end_new)})")

    # ---- gap between .text and the first following section ----
    gap_old = old.data[text_end_old:first_old.offset]
    gap_new = new.data[text_end_new:first_new.offset]
    if not all_zero(gap_old):
        errors.append("old bytes between .text end and the next section are not zero")
    if not all_zero(gap_new):
        errors.append("new bytes between .text end and the next section are not zero")

    # ---- symbols ----
    sym_masks_old: list[tuple[int, int]] = []
    sym_masks_new: list[tuple[int, int]] = []
    tables = {}
    for s in old.sections:
        if s.type in (SHT_SYMTAB, SHT_DYNSYM):
            n = new.sections[s.idx]
            so, sn = old.symbols(s), new.symbols(n)
            tables[s.name] = (so, sn)
            if len(so) != len(sn):
                errors.append(f"{s.name}: symbol count {len(so)} -> {len(sn)}")
                continue
            for i, (a, b) in enumerate(zip(so, sn)):
                name, info, other, shndx, value, size = a
                if (name, info, other, shndx, size) != (b[0], b[1], b[2], b[3], b[5]):
                    errors.append(f"{s.name}[{i}] {name!r}: name/info/other/shndx/size changed "
                                  f"(size {size} -> {b[5]}, shndx {shndx} -> {b[3]})")
                    continue
                delta = addr_delta.get(shndx, 0) if shndx < SHN_LORESERVE else 0
                if b[4] != value + delta:
                    errors.append(f"{s.name}[{i}] {name!r}: st_value {hexs(value)} -> {hexs(b[4])}, "
                                  f"expected {hexs(value + delta)} (genuine symbol relocation)")
                if delta and cls.get(s.idx) == "post":
                    sym_masks_old.append((s.offset + 24 * i + 8, s.offset + 24 * i + 16))
                    sym_masks_new.append((n.offset + 24 * i + 8, n.offset + 24 * i + 16))
                    rep.setdefault("relocated_symbols", []).append(f"{name}: {hexs(value)} -> {hexs(b[4])}")
    first_old_rel = first_old.offset
    compare_region(errors, "post-.text sections ([.dynamic..section headers))",
                   old.data[first_old_rel:old.e_shoff], new.data[first_new.offset:new.e_shoff],
                   first_old_rel, first_new.offset, old, new, sym_masks_old, sym_masks_new)
    if new.e_shoff - old.e_shoff != d_file:
        errors.append(f"e_shoff moved by {hexs(new.e_shoff - old.e_shoff)}, not by the section shift "
                      f"{hexs(d_file)}")
    dyn_old = next((s for s in old.sections if s.name == ".dynamic"), None)
    for name, (so, sn) in tables.items():
        for a, b in zip(so, sn):
            if a[0] == "_DYNAMIC" and dyn_old is not None and b[4] != new.sections[dyn_old.idx].addr:
                errors.append(f"{name}: _DYNAMIC st_value {hexs(b[4])} != new .dynamic address "
                              f"{hexs(new.sections[dyn_old.idx].addr)}")

    # ---- kernels and descriptors ----
    table = tables.get(".symtab") or tables.get(".dynsym")
    kern_ok = desc_ok = 0
    kernels: list[str] = []
    descs: list[str] = []
    kern_end = 0
    if table is None:
        errors.append("no symbol table: cannot verify kernel/descriptor bytes")
    else:
        so, sn = table
        rodata_like = {s.idx for s in old.sections if s.alloc and not s.flags & SHF_EXEC and not s.nobits}
        for a, b in zip(so, sn):
            name, info, other, shndx, value, size = a
            typ = info & 0xF
            if typ == STT_FUNC and shndx == ot.idx and size:
                kernels.append(name)
                lo = value - ot.addr
                kern_end = max(kern_end, lo + size)
                same = lo >= 0 and lo + size <= len(old_text) and b[4] == value and b[5] == size and \
                    new_text[lo:lo + size] == old_text[lo:lo + size]
                if same:
                    kern_ok += 1
                else:
                    errors.append(f"kernel {name}: code bytes/value/size not identical in new "
                                  f"(old {hexs(value)}+{size}, new {hexs(b[4])}+{b[5]})")
            elif typ == STT_OBJECT and shndx in rodata_like and (name.endswith(".kd") or size == KD_SIZE):
                descs.append(name)
                if size != KD_SIZE:
                    notes.append(f"descriptor {name} has size {size}, not {KD_SIZE}")
                so_, sn_ = old.sections[shndx], new.sections[shndx]
                lo_o, lo_n = value - so_.addr, b[4] - sn_.addr
                ob = old.payload(so_)[lo_o:lo_o + size]
                nb = new.payload(sn_)[lo_n:lo_n + size]
                if ob == nb and len(ob) == size and b[4] == value and b[5] == size:
                    desc_ok += 1
                else:
                    moved = [f"+{i:#x}" for i in range(min(len(ob), len(nb))) if ob[i] != nb[i]][:8]
                    errors.append(f"descriptor {name}: bytes differ ({', '.join(moved) or 'size/value'}); "
                                  f"genuine descriptor relocation, not allowed")
        missing_kd = [k for k in kernels if f"{k}.kd" not in descs]
        if missing_kd:
            notes.append("kernel(s) without a .kd descriptor symbol: " + ", ".join(missing_kd))
        if not kernels:
            errors.append("no kernel FUNC symbols found in .text")
        if kernels and kern_end != ot.size:
            notes.append(f"old .text has {ot.size - kern_end} B after the last kernel symbol end")
    rep.update(kernels=len(kernels), kernels_ok=kern_ok, descriptors=len(descs), descriptors_ok=desc_ok)
    return rep


def compare_modules(old: Module, new: Module) -> dict:
    errors: list[str] = []
    entries: list[dict] = []
    res = {"errors": errors, "entries": entries}
    if old.kind != new.kind:
        errors.append(f"container kind {old.kind} -> {new.kind}")
        return res
    if [e.ident for e in old.entries] != [e.ident for e in new.entries]:
        errors.append("bundle entry ids differ: "
                      f"{[e.ident for e in old.entries]} -> {[e.ident for e in new.entries]}")
        return res
    if old.kind == "bundle":
        if old.header_end != new.header_end:
            errors.append("bundle header length changed")
        else:
            size_masks = [(e.size_pos, e.size_pos + 8) for e in old.entries if e.size]
            a = masked(old.data[:old.header_end], 0, size_masks)
            b = masked(new.data[:new.header_end], 0, size_masks)
            if a != b:
                errors.append("bundle header differs outside ELF entry size fields at " +
                              ", ".join(f"{hexs(s)}..{hexs(e)}" for s, e in diff_runs(a, b)))
    spans_old, spans_new = [], []
    for eo, en in zip(old.entries, new.entries):
        if eo.offset != en.offset:
            errors.append(f"bundle entry {eo.ident}: offset {hexs(eo.offset)} -> {hexs(en.offset)}")
        bo, bn = old.payload(eo), new.payload(en)
        if not bo and not bn:
            continue
        spans_old.append((eo.offset, eo.offset + eo.size))
        spans_new.append((en.offset, en.offset + en.size))
        if bo[:4] != b"\x7fELF":
            if bo != bn:
                errors.append(f"bundle entry {eo.ident}: non-ELF payload differs")
            continue
        label = f"{eo.ident}"
        try:
            r = compare_elf(Elf(bo, f"old[{label}]"), Elf(bn, f"new[{label}]"))
        except FormatError as exc:
            r = {"errors": [str(exc)], "notes": []}
        r["entry"] = label
        entries.append(r)
    if old.kind == "bundle":
        for which, mod, spans in (("old", old, spans_old), ("new", new, spans_new)):
            pos = mod.header_end
            for s, e in sorted(spans):
                if s < pos:
                    errors.append(f"{which}: bundle entries overlap or start inside the header")
                elif not all_zero(mod.data[pos:s]):
                    errors.append(f"{which}: non-zero bytes between bundle entries at {hexs(pos)}..{hexs(s)}")
                pos = max(pos, e)
            if not all_zero(mod.data[pos:]):
                errors.append(f"{which}: non-zero trailing bytes after the last bundle entry")
    if not entries:
        errors.append("no ELF payload found to compare")
    return res


# ---------------------------------------------------------------------------
# Self-test (`--self-test`): synthetic code objects laid out like the native writer
# (crates/peacemaker-isa/src/native/elf.rs); the checker must accept the code-end tail
# and reject every other kind of change.

_SHSTRTAB = (b"\0.note\0.dynsym\0.gnu.hash\0.hash\0.dynstr\0.rodata\0.text\0.dynamic\0.relro_padding\0"
             b".comment\0.symtab\0.shstrtab\0.strtab\0")


def _shstr(name: bytes) -> int:
    return _SHSTRTAB.index(b"\0" + name + b"\0") + 1


def _code(nbytes: int, seed: int) -> bytes:
    words = [0xC0000000 | ((seed << 12) + i) for i in range(nbytes // 4 - 1)] + [0xBFB00000]
    return struct.pack(f"<{len(words)}I", *words)


def _desc(seed: int) -> bytes:
    return bytes((seed * 37 + i * 5) & 0xFF for i in range(KD_SIZE))


def _kernels(sizes=(300, 1000)):
    return [(f"k{i}", _code(n, i), _desc(i)) for i, n in enumerate(sizes)]


def _synth_elf(kernels, tail=True, tail_dwords=CODE_END_DWORDS, align_word=S_CODE_END) -> bytes:
    text = bytearray()
    entries = []
    for _name, code, _desc_bytes in kernels:
        while len(text) % 256:
            text += struct.pack("<I", _S_NOP)
        entries.append(len(text))
        text += code
    if tail:
        text += struct.pack("<I", align_word) * (((-len(text)) % CACHE_LINE) // 4)
        text += struct.pack("<I", S_CODE_END) * tail_dwords
    meta = bytes(range(16))
    note = struct.pack("<III", 7, len(meta), 32) + b"AMDGPU\0\0" + meta
    note += bytes(align(len(note), 4) - len(note))
    names = []
    for i, (sym, _c, _d) in enumerate(kernels):
        names += [(sym, False, i), (sym + ".kd", True, i)]
    dynstr = bytearray(b"\0")
    name_off = []
    for n, _kd, _i in names:
        name_off.append(len(dynstr))
        dynstr += n.encode() + b"\0"
    strtab = bytearray(dynstr)
    dyn_name = len(strtab)
    strtab += b"_DYNAMIC\0"
    nn = len(names)
    buckets = max(nn // 4, 1)
    bloom = 1 << (((nn * 12) // 64 + 1) - 1).bit_length()
    dynsym_size = (1 + nn) * 24
    gnu_size = 16 + 8 * bloom + 4 * buckets + 4 * nn
    hash_size = 4 * (2 + 2 * (1 + nn))
    rodata_size = KD_SIZE * len(kernels)
    note_off = align(EHDR_SIZE + 8 * 56, 4)
    dynsym_off = align(note_off + len(note), 8)
    gnu_off = align(dynsym_off + dynsym_size, 8)
    hash_off = align(gnu_off + gnu_size, 4)
    dynstr_off = hash_off + hash_size
    rodata_off = align(dynstr_off + len(dynstr), KD_SIZE)
    ro_end = rodata_off + rodata_size
    text_addr = segment_start(ro_end, 256)
    text_off = align_skew(ro_end, PAGE, text_addr)
    dyn_size = 7 * 16
    dyn_addr = segment_start(text_addr + len(text), 8)
    dyn_off = align_skew(text_off + len(text), PAGE, dyn_addr)
    relro_addr = dyn_addr + dyn_size
    relro_size = align(relro_addr, PAGE) - relro_addr
    comment = b"hipfire peacemaker native-emit selftest\0"
    comment_off = dyn_off + dyn_size
    symtab_off = align(comment_off + len(comment), 8)
    symtab_size = (2 + nn) * 24
    shstr_off = symtab_off + symtab_size
    strtab_off = shstr_off + len(_SHSTRTAB)
    shoff = align(strtab_off + len(strtab), 8)

    out = bytearray()

    def put(off: int, data: bytes) -> None:
        if len(out) < off + len(data):
            out.extend(bytes(off + len(data) - len(out)))
        out[off:off + len(data)] = data

    def kd_addr(i: int) -> int:
        return rodata_off + i * KD_SIZE

    def sym(i: int) -> bytes:
        _n, kd, k = names[i]
        return struct.pack("<IBBHQQ", name_off[i], 0x10 | (1 if kd else 2), 3, 6 if kd else 7,
                           kd_addr(k) if kd else text_addr + entries[k],
                           KD_SIZE if kd else len(kernels[k][1]))

    put(0, bytes([0x7F, 0x45, 0x4C, 0x46, 2, 1, 1, 0x40, 4]) + bytes(7) +
        struct.pack("<HHIQQQIHHHHHH", 3, 224, 1, 0, EHDR_SIZE, shoff, 0x4E, EHDR_SIZE, 56, 8, 64, 14, 12))
    phdrs = [
        (6, 4, EHDR_SIZE, EHDR_SIZE, 8 * 56, 8 * 56, 8),
        (1, 4, 0, 0, ro_end, ro_end, PAGE),
        (1, 5, text_off, text_addr, len(text), len(text), PAGE),
        (1, 6, dyn_off, dyn_addr, dyn_size, dyn_size + relro_size, PAGE),
        (2, 6, dyn_off, dyn_addr, dyn_size, dyn_size, 8),
        (PT_GNU_RELRO, 4, dyn_off, dyn_addr, dyn_size, dyn_size + relro_size, 1),
        (0x6474E551, 6, 0, 0, 0, 0, 0),
        (4, 4, note_off, note_off, len(note), len(note), 4),
    ]
    for i, (kind, flags, off, addr, filesz, memsz, palign) in enumerate(phdrs):
        put(EHDR_SIZE + 56 * i, struct.pack("<IIQQQQQQ", kind, flags, off, addr, addr, filesz, memsz, palign))
    put(note_off, note)
    put(dynsym_off, bytes(24) + b"".join(sym(i) for i in range(nn)))
    put(gnu_off, bytes([0xA5]) * gnu_size)
    put(hash_off, bytes([0x5A]) * hash_size)
    put(dynstr_off, bytes(dynstr))
    rodata = bytearray()
    for i, (_n, _c, desc) in enumerate(kernels):
        kd = bytearray(desc)
        kd[0x10:0x18] = struct.pack("<q", text_addr + entries[i] - kd_addr(i))
        rodata += kd
    put(rodata_off, bytes(rodata))
    put(text_off, bytes(text))
    dynamic = b"".join(struct.pack("<QQ", t, v) for t, v in
                       [(6, dynsym_off), (11, 24), (5, dynstr_off), (10, len(dynstr)),
                        (0x6FFFFEF5, gnu_off), (4, hash_off), (0, 0)])
    put(dyn_off, dynamic)
    put(comment_off, comment)
    put(symtab_off, bytes(24) + struct.pack("<IBBHQQ", dyn_name, 0, 2, 8, dyn_addr, 0) +
        b"".join(sym(i) for i in range(nn)))
    put(shstr_off, _SHSTRTAB)
    put(strtab_off, bytes(strtab))
    al, wr, ex, mg, st = 2, 1, 4, 0x10, 0x20
    sections = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (_shstr(b".note"), 7, al, note_off, note_off, len(note), 0, 0, 4, 0),
        (_shstr(b".dynsym"), 11, al, dynsym_off, dynsym_off, dynsym_size, 5, 1, 8, 24),
        (_shstr(b".gnu.hash"), 0x6FFFFFF6, al, gnu_off, gnu_off, gnu_size, 2, 0, 8, 0),
        (_shstr(b".hash"), 5, al, hash_off, hash_off, hash_size, 2, 0, 4, 4),
        (_shstr(b".dynstr"), 3, al, dynstr_off, dynstr_off, len(dynstr), 0, 0, 1, 0),
        (_shstr(b".rodata"), 1, al, rodata_off, rodata_off, rodata_size, 0, 0, KD_SIZE, 0),
        (_shstr(b".text"), 1, al | ex, text_addr, text_off, len(text), 0, 0, 256, 0),
        (_shstr(b".dynamic"), 6, al | wr, dyn_addr, dyn_off, dyn_size, 5, 0, 8, 16),
        (_shstr(b".relro_padding"), 8, al | wr, relro_addr, comment_off, relro_size, 0, 0, 1, 0),
        (_shstr(b".comment"), 1, mg | st, 0, comment_off, len(comment), 0, 0, 1, 1),
        (_shstr(b".symtab"), 2, 0, 0, symtab_off, symtab_size, 13, 2, 8, 24),
        (_shstr(b".shstrtab"), 3, 0, 0, shstr_off, len(_SHSTRTAB), 0, 0, 1, 0),
        (_shstr(b".strtab"), 3, 0, 0, strtab_off, len(strtab), 0, 0, 1, 0),
    ]
    put(shoff, b"".join(struct.pack("<IIQQQQIIQQ", *s) for s in sections))
    return bytes(out)


def _synth_bundle(elf: bytes) -> bytes:
    ids = [b"host-x86_64-unknown-linux-gnu", b"hipv4-amdgcn-amd-amdhsa--gfx1151"]
    off = 0x1000
    hdr = BUNDLE_MAGIC + struct.pack("<Q", 2)
    hdr += struct.pack("<QQQ", off, 0, len(ids[0])) + ids[0]
    hdr += struct.pack("<QQQ", off, len(elf), len(ids[1])) + ids[1]
    return hdr + bytes(off - len(hdr)) + elf


class SelfTest(unittest.TestCase):
    def setUp(self):
        self.old = _synth_elf(_kernels(), tail=False)
        self.new = _synth_elf(_kernels())

    @staticmethod
    def errors(old: bytes, new: bytes) -> list[str]:
        try:
            return compare_elf(Elf(old, "old"), Elf(new, "new"))["errors"]
        except FormatError as exc:
            return [str(exc)]

    @staticmethod
    def section(data: bytes, name: str) -> Section:
        return next(s for s in Elf(data, "x").sections if s.name == name)

    @staticmethod
    def flip(data: bytes, off: int, xor: int = 0x01) -> bytes:
        b = bytearray(data)
        b[off] ^= xor
        return bytes(b)

    @staticmethod
    def poke(data: bytes, off: int, raw: bytes) -> bytes:
        b = bytearray(data)
        b[off:off + len(raw)] = raw
        return bytes(b)

    def test_accepts_code_end_tail(self):
        r = compare_elf(Elf(self.old, "old"), Elf(self.new, "new"))
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["text_new"] - r["text_old"], r["align_bytes"] + CODE_END_DWORDS * 4)
        self.assertEqual((r["kernels"], r["kernels_ok"], r["descriptors"], r["descriptors_ok"]), (2, 2, 2, 2))
        self.assertEqual(r["text_new"] % CACHE_LINE, 0)
        self.assertIn("_DYNAMIC", " ".join(r.get("relocated_symbols", [])))

    def test_accepts_across_page_boundaries(self):
        for n in (3000, 3500, 3700, 3800, 3900, 3968, 3980, 4000, 4096, 5000):
            ks = [("k", _code(n, 1), _desc(1))]
            self.assertEqual(self.errors(_synth_elf(ks, tail=False), _synth_elf(ks)), [], n)

    def test_rejects_changed_code_byte(self):
        text = self.section(self.new, ".text")
        errs = self.errors(self.old, self.flip(self.new, text.offset + 8))
        self.assertTrue(any("kernel k0" in e for e in errs), errs)
        self.assertTrue(self.errors(self.old, _synth_elf(_kernels((304, 1000)))))

    def test_rejects_changed_descriptor_byte(self):
        rodata = self.section(self.new, ".rodata")
        for rel in (0x20, 0x10):  # payload byte, kernel_code_entry_byte_offset
            errs = self.errors(self.old, self.flip(self.new, rodata.offset + rel))
            self.assertTrue(any("descriptor k0.kd" in e for e in errs), (rel, errs))

    def test_rejects_unrelated_section_byte(self):
        for name in (".note", ".gnu.hash", ".hash", ".dynstr", ".dynamic", ".comment", ".symtab",
                     ".shstrtab", ".strtab"):
            s = self.section(self.new, name)
            self.assertTrue(self.errors(self.old, self.flip(self.new, s.offset + 1)), name)

    def test_rejects_unexpected_elf_fields(self):
        new = self.new
        e = Elf(new, "new")
        dyn = next(s for s in e.sections if s.name == ".dynamic")
        text = next(s for s in e.sections if s.name == ".text")
        cases = {
            "e_flags": self.flip(new, 0x30),
            "e_entry": self.poke(new, 0x18, struct.pack("<Q", 1)),
            "e_machine": self.flip(new, 0x12),
            "sh_flags(.dynamic)": self.flip(new, e.e_shoff + 64 * dyn.idx + 8, 0x01),
            "sh_addralign(.text)": self.flip(new, e.e_shoff + 64 * text.idx + 48, 0x01),
            "p_flags(LOAD rx)": self.flip(new, e.e_phoff + 56 * 2 + 4, 0x02),
            "p_align(LOAD rw)": self.flip(new, e.e_phoff + 56 * 3 + 48, 0x10),
            "p_memsz(LOAD rw)": self.flip(new, e.e_phoff + 56 * 3 + 40, 0x01),
        }
        for what, mutated in cases.items():
            self.assertTrue(self.errors(self.old, mutated), what)

    def test_rejects_malformed_final_fill(self):
        text = self.section(self.new, ".text")
        end = text.offset + text.size
        cases = {
            "last dword is s_nop": self.poke(self.new, end - 4, struct.pack("<I", _S_NOP)),
            "middle dword is zero": self.poke(self.new, end - 200, bytes(4)),
            "first code-end dword wrong": self.poke(self.new, end - 4 * CODE_END_DWORDS,
                                                    struct.pack("<I", 0xBF9F0001)),
            "alignment gap not code-end": _synth_elf(_kernels(), align_word=_S_NOP),
            "95 dwords": _synth_elf(_kernels(), tail_dwords=CODE_END_DWORDS - 1),
            "97 dwords": _synth_elf(_kernels(), tail_dwords=CODE_END_DWORDS + 1),
            "no tail": self.old,
        }
        for what, mutated in cases.items():
            self.assertTrue(self.errors(self.old, mutated), what)

    def test_bundle_and_root_mode(self):
        import contextlib
        import io
        import tempfile

        def run(argv):
            with contextlib.redirect_stdout(io.StringIO()):
                return main(argv)

        rodata = self.section(self.new, ".rodata")
        with tempfile.TemporaryDirectory() as tmp:
            old_root, new_root = Path(tmp, "old"), Path(tmp, "new")
            old_root.mkdir()
            new_root.mkdir()
            (old_root / "x_pm_gfx1151.hxaco").write_bytes(_synth_bundle(self.old))
            (new_root / "x_pm_gfx1151.hxaco").write_bytes(_synth_bundle(self.new))
            (old_root / "plain.hxaco").write_bytes(b"not a pm module")
            roots = ["--old-root", str(old_root), "--new-root", str(new_root)]
            self.assertEqual(run(roots), 0)
            self.assertEqual(run(["--pair", str(old_root / "x_pm_gfx1151.hxaco"),
                                  str(new_root / "x_pm_gfx1151.hxaco")]), 0)
            bad = self.flip(self.new, rodata.offset + 0x20)
            (new_root / "x_pm_gfx1151.hxaco").write_bytes(_synth_bundle(bad))
            self.assertEqual(run(roots), 1)
            junk = bytearray(_synth_bundle(self.new))
            junk[0x800] = 1  # non-zero byte in the gap between bundle header and ELF entry
            (new_root / "x_pm_gfx1151.hxaco").write_bytes(bytes(junk))
            self.assertEqual(run(roots), 1)
            (new_root / "x_pm_gfx1151.hxaco").unlink()
            self.assertEqual(run(roots), 1)  # OLD artifact without a NEW one is reported


def run_self_test() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest))
    return 0 if result.wasSuccessful() else 1


# ---------------------------------------------------------------------------
# Driver


def select(root: Path, include: re.Pattern) -> dict[str, Path]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in SUFFIXES and include.search(p.stem):
            out[p.relative_to(root).as_posix()] = p
    return out


def fmt_entry(e: dict) -> list[str]:
    lines: list[str] = []
    if "fill" in e:
        added = e["text_new"] - e["text_old"]
        lines += [
            f"    [{e['entry']}] .text old={e['text_old']} B new={e['text_new']} B ({added:+d} B: "
            f"{e['align_bytes']} B align + {CODE_END_DWORDS * 4} B code-end)",
            f"        room to page end: old={e['room_old']} B (end {hexs(e['end_old'])}) "
            f"new={e['room_new']} B (end {hexs(e['end_new'])})",
            f"        fill: {e['fill']}",
            f"        kernels {e['kernels_ok']}/{e['kernels']} code identical, descriptors "
            f"{e['descriptors_ok']}/{e['descriptors']} identical; shifts after .text: "
            f"file {e['shift_file']:+#x}, addr {e['shift_addr']:+#x}",
        ]
        for r in e.get("relocated_symbols", []):
            lines.append(f"        derived symbol move: {r}")
    else:
        lines.append(f"    [{e['entry']}] not compared")
    lines += [f"        note: {n}" for n in e["notes"]]
    lines += [f"        FAIL: {x}" for x in e["errors"]]
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prove PM module .text code-end tail is the only change.")
    ap.add_argument("--old-root", type=Path, help="directory of OLD modules (pre-fix artifacts)")
    ap.add_argument("--new-root", type=Path, help="directory of NEW modules (default: <repo>/kernels)")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("OLD", "NEW"), type=Path,
                    help="explicit OLD/NEW file pair (repeatable)")
    ap.add_argument("--include", default=DEFAULT_INCLUDE,
                    help=f"regex on the file stem for root mode (default {DEFAULT_INCLUDE!r})")
    ap.add_argument("--json", type=Path, help="write the full report as JSON")
    ap.add_argument("-q", "--quiet", action="store_true", help="print failures and the summary only")
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in rejection/acceptance tests on synthetic code objects and exit")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()

    pairs: list[tuple[str, Path, Path]] = []
    unmatched: list[str] = []
    if args.pair:
        pairs += [(str(n), o, n) for o, n in args.pair]
    if args.old_root is not None or not args.pair:
        if args.old_root is None:
            ap.error("give --old-root (and optionally --new-root) or one or more --pair OLD NEW")
        new_root = args.new_root or Path(__file__).resolve().parent.parent / "kernels"
        for r in (args.old_root, new_root):
            if not r.is_dir():
                ap.error(f"not a directory: {r}")
        inc = re.compile(args.include)
        o, n = select(args.old_root, inc), select(new_root, inc)
        unmatched += [f"{k}: present in new root only (no OLD artifact)" for k in sorted(n.keys() - o.keys())]
        unmatched += [f"{k}: present in old root only (no NEW artifact)" for k in sorted(o.keys() - n.keys())]
        pairs += [(k, o[k], n[k]) for k in sorted(o.keys() & n.keys())]
        if not o and not n:
            unmatched.append(f"no PM modules matched {args.include!r} under {args.old_root} or {new_root}")
    if not pairs and not unmatched:
        ap.error("nothing to compare")

    report, failed = [], 0
    for label, op, np_ in pairs:
        try:
            res = compare_modules(Module(op), Module(np_))
        except (FormatError, OSError) as exc:
            res = {"errors": [str(exc)], "entries": []}
        bad = bool(res["errors"]) or any(e["errors"] for e in res["entries"])
        failed += bad
        res.update(file=label, old=str(op), new=str(np_), ok=not bad)
        report.append(res)
        if bad or not args.quiet:
            print(f"{'FAIL' if bad else 'OK  '} {label}")
            for x in res["errors"]:
                print(f"    FAIL: {x}")
            for e in res["entries"]:
                print("\n".join(fmt_entry(e)))
    for u in unmatched:
        print(f"FAIL {u}")
    total = len(pairs)
    print(f"\n{total - failed}/{total} module(s) proven: only .text tail padding + {CODE_END_DWORDS} x "
          f"{hexs(S_CODE_END)} added" + (f"; {failed} failed" if failed else "") +
          (f"; {len(unmatched)} unmatched" if unmatched else ""))
    if args.json:
        args.json.write_text(json.dumps({"modules": report, "unmatched": unmatched}, indent=2, default=str) + "\n")
    return 1 if failed or unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
