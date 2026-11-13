"""Wire protocol between the runner (client) and a submission's policy server (spec §4.5).

Websocket (port 8000) + msgpack. numpy arrays travel as {"__nd__": 1, "dtype", "shape", "data": bytes}
(openpi-style, so existing msgpack-numpy code is easy to adapt). Every message is a dict with a "type".

  server -> client on connect : {"type": "metadata", "version": 1, "name": str, "action_horizon": int|None, ...}
  client -> server            : {"type": "reset", "task": str, "instruction": str, "seed": int}
  server -> client            : {"type": "reset_ok"}
  client -> server            : {"type": "act", "obs": {head_rgb, head_depth, ..., arm_q, hand_q, instruction, t}}
  server -> client            : {"type": "action", "action": {arm_q_target (14,)|(k,14), hand_q_target, done?}}
  server -> client on failure : {"type": "error", "message": str}
"""
from __future__ import annotations

import msgpack
import numpy as np

VERSION = 1
PORT = 8000


def _enc(o):
    if isinstance(o, np.ndarray):
        return {"__nd__": 1, "dtype": o.dtype.str, "shape": list(o.shape), "data": np.ascontiguousarray(o).tobytes()}
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"cannot serialise {type(o)}")


def _dec(d):
    if "__nd__" in d:
        return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()
    return d


def pack(msg) -> bytes:
    return msgpack.packb(msg, default=_enc, use_bin_type=True)


def unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_dec, raw=False)


def check_metadata(meta):
    if not isinstance(meta, dict) or meta.get("type") != "metadata":
        raise ConnectionError(f"expected a metadata handshake, got {str(meta)[:100]}")
    if meta.get("version") != VERSION:
        raise ConnectionError(f"protocol version mismatch: server {meta.get('version')} client {VERSION}")
    return meta
