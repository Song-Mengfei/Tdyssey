"""Saved-state RGB rendering and CPU-only R3A training data interfaces.

State/geometry/letter labels are renderer and audit data only. They are never
returned as model features. No physics stepping, model, Torch or training lives
in this module.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

LETTERS=tuple('TLOCXB')
PROTOCOL_SHA256='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class RecordedStateRenderer:
    """One reusable headless original renderer per seen letter; never step it."""
    def __init__(self,source_root,*,maximum_frame_cache_bytes=32*1024**2):
        if type(maximum_frame_cache_bytes)is not int or maximum_frame_cache_bytes<0:
            raise ValueError('Nonnegative bounded RAM cache required')
        self.source_root=Path(source_root).resolve();self.maximum_frame_cache_bytes=maximum_frame_cache_bytes
        self.cache=OrderedDict();self.cache_bytes=0;self.environments={}
        self.stats=dict(environment_constructions=0,rendered_frames=0,cache_hits=0,physics_steps=0,physics_controls=0)

    def _environment(self,letter,first_state):
        if letter not in LETTERS:raise ValueError('Only the six declared seen letters may be rendered')
        if letter not in self.environments:
            from .native_rgb_env import make_env
            from .envs.official import official_shape
            env=make_env(self.source_root,letter=letter,render_size=96)
            # The sole setup uses this saved pose, not a fresh random scene.
            # Velocities/forces do not affect RGB and are set immediately below.
            cog=np.asarray(official_shape(letter).centroid,dtype=np.float64)
            angle=float(first_state[4]);c,s=math.cos(angle),math.sin(angle)
            center=np.asarray(first_state[2:4])+np.array(((c,-s),(s,c)))@cog
            env.reset_scene(center=center,angle=angle,pusher=first_state[:2])
            self.environments[letter]=env;self.stats['environment_constructions']+=1
        return self.environments[letter]

    def render(self,letter,state,*,cache_key=None):
        state=np.asarray(state)
        if state.dtype!=np.float64 or state.shape!=(15,)or not np.isfinite(state).all():
            raise ValueError('Exact saved finite float64 state15 required; no pose reduction')
        key=None if cache_key is None else(letter,str(cache_key),hashlib.sha256(state.tobytes()).hexdigest())
        if key is not None and key in self.cache:
            self.cache.move_to_end(key);self.stats['cache_hits']+=1
            return self.cache[key].copy()
        env=self._environment(letter,state);env.set_render_state(state)
        before=env.state().copy();rgb=env.render_clean()
        if not np.array_equal(env.state(),before):raise RuntimeError('Recorded-state RGB rendering changed physics state')
        if rgb.dtype!=np.uint8 or rgb.shape!=(3,96,96):raise ValueError('Original96 CHW uint8 RGB layout changed')
        self.stats['rendered_frames']+=1
        if key is not None and rgb.nbytes<=self.maximum_frame_cache_bytes:
            while self.cache and self.cache_bytes+rgb.nbytes>self.maximum_frame_cache_bytes:
                _,removed=self.cache.popitem(last=False);self.cache_bytes-=removed.nbytes
            self.cache[key]=rgb.copy();self.cache_bytes+=rgb.nbytes
        return rgb.copy()

    def close(self):
        self.cache.clear();self.cache_bytes=0
        for env in self.environments.values():env.close()
        self.environments.clear()


def assert_model_inputs(value,*,batch=False):
    """Only RGB and normalized actions cross the network boundary."""
    if set(value)!={'image','action'}:raise ValueError('Model inputs may contain only image/action')
    image,action=np.asarray(value['image']),np.asarray(value['action'])
    shape=(None,5,3,96,96)if batch else(5,3,96,96)
    good_image=image.shape[1:]==shape[1:]if batch else image.shape==shape
    good_action=action.shape==(len(image),4,2)if batch else action.shape==(4,2)
    if (not good_image or not good_action or image.dtype!=np.float32 or action.dtype!=np.float32
            or not np.isfinite(image).all()or not np.isfinite(action).all()
            or image.min()<0 or image.max()>1 or action.min()<-1 or action.max()>1):
        raise ValueError('Expected float32 fiveRGB[0,1] and fouractions[-1,1]')
    return value
