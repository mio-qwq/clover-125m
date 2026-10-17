"""Training display and durable logs; no model, data or optimizer access."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from tqdm import tqdm


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '--:--:--'
    seconds = max(0, int(math.ceil(seconds)))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    value = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'{days}d {value}' if days else value


def text_bar(percent, width=20):
    filled = min(width, max(0, int(percent * width / 100)))
    return '[' + '#' * filled + '-' * (width - filled) + ']'


class TrainingProgress:
    """Absolute completed counters; ETA uses only this session's new tokens.

    Elapsed wall time includes data loading and observed validation/save overhead.
    ETA estimates the remaining training tokens, excluding final validation/export.
    Display updates never consume RNG, move the sampler or mutate model state.
    """
    def __init__(self, total_tokens, total_updates, *, initial_tokens=0,
                 initial_update=0, stream=None, clock=time.perf_counter, enabled=None):
        if not (total_tokens > 0 and total_updates > 0
                and 0 <= initial_tokens <= total_tokens
                and 0 <= initial_update <= total_updates):
            raise ValueError('Invalid training progress counters')
        self.total_tokens = total_tokens
        self.total_updates = total_updates
        self.tokens = self.initial_tokens = initial_tokens
        self.update = initial_update
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock
        self.started = clock()
        self.phase = 'Train'
        self.loss = None
        self.closed = False
        if enabled is None:
            enabled = bool(getattr(self.stream, 'isatty', lambda: False)())
        self.bar = tqdm(total=total_tokens, initial=initial_tokens, file=self.stream,
                        disable=not enabled, ascii=True, dynamic_ncols=True,
                        mininterval=0.5, desc=self.phase,
                        bar_format='{desc}: {percentage:6.2f}%|{bar:24}| {postfix}')
        self._render(force=True)

    def snapshot(self):
        elapsed = max(0., self.clock() - self.started)
        newly_done = self.tokens - self.initial_tokens
        speed = newly_done / elapsed if elapsed > 0 else 0.
        remaining = self.total_tokens - self.tokens
        eta = remaining / speed if speed > 0 else None
        if remaining == 0:
            eta = 0.
        return {'progress_percent': 100. * self.tokens / self.total_tokens,
                'elapsed_seconds': elapsed, 'tokens_per_second': speed,
                'eta_seconds': eta, 'phase': self.phase}

    def _render(self, force=False):
        values = self.snapshot()
        loss = '--' if self.loss is None else f'{self.loss:.4f}'
        postfix = (f'step {self.update:,}/{self.total_updates:,} | loss {loss} | '
                   f'{values["tokens_per_second"]:,.0f} tok/s | '
                   f'ETA(train)~ {duration(values["eta_seconds"])}')
        self.bar.set_postfix_str(postfix, refresh=False)
        if force:
            self.bar.refresh()
        return values

    def advance(self, update, tokens, loss=None):
        if self.closed or not (self.update <= update <= self.total_updates
                               and self.tokens <= tokens <= self.total_tokens):
            raise ValueError('Progress must use monotonic completed counters within budget')
        increment = tokens - self.tokens
        self.update, self.tokens, self.loss = update, tokens, loss
        values = self._render()
        self.bar.update(increment)
        return values

    def set_phase(self, phase):
        self.phase = phase
        self.bar.set_description_str(phase, refresh=False)
        self._render(force=True)

    def write(self, line):
        if self.bar.disable:
            print(line, file=self.stream, flush=True)
        else:
            tqdm.write(line, file=self.stream)
            self.stream.flush()

    def close(self):
        if not self.closed:
            self._render(force=True)
            self.bar.close()
            self.closed = True


def readable_event(event):
    """Single-line ASCII status, safe for Windows terminals and redirected logs."""
    stamp = datetime.fromisoformat(event['at_utc']).astimezone().strftime('%H:%M:%S')
    kind = event['kind']
    prefix = f'[{stamp}] {kind.upper()}'
    if kind == 'train':
        return (f'{prefix} {text_bar(event["progress_percent"])} '
                f'{event["progress_percent"]:.2f}% | '
                f'step {event["update"]:,}/{event["total_updates"]:,} | '
                f'loss(update) {event["loss"]:.4f} | lr {event["learning_rate"]:.2e} | '
                f'{event["tokens_per_second"]:,.0f} tok/s | '
                f'elapsed {duration(event["elapsed_seconds"])} | '
                f'ETA(train)~ {duration(event["eta_seconds"])} | '
                f'VRAM peak {event["peak_allocated_bytes"] / 2**30:.2f} GiB')
    if kind in ('mini_validation', 'full_validation', 'final_validation'):
        sources = ' '.join(f'{name}={row["loss"]:.4f}' for name, row in event['sources'].items()
                           if row['loss'] is not None)
        return (f'{prefix} step {event["update"]:,} | loss {event["loss"]:.4f} | '
                f'PPL {event["perplexity"]:.2f} | tokens {event["tokens"]:,} | {sources}')
    if kind == 'checkpoint':
        return (f'{prefix} saved step {event["update"]:,} | '
                f'{event["checkpoint"]} | took {duration(event["duration_seconds"])}')
    if kind == 'stopped':
        saved = f'{event["last_saved_update"]:,}' if event['last_saved_update'] >= 0 else 'none'
        return (f'{prefix} {event["error"]}: {event["message"]} | '
                f'last saved step {saved}')
    if kind == 'completed':
        return (f'{prefix} 100.00% | step {event["update"]:,}/{event["total_updates"]:,} | '
                f'tokens {event["tokens"]:,} | elapsed {duration(event["elapsed_seconds"])} | '
                f'export {event["export"]}')
    return f'{prefix} {event.get("message", "")} | phase {event.get("phase", "--")}'


def write_event(path, value, progress=None):
    event = {'at_utc': datetime.now(timezone.utc).isoformat(), **value}
    # Console messages stay on one physical line, including multiline exceptions.
    line = readable_event(event).replace('\r', '\\r').replace('\n', '\\n')
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    with Path(path).with_name('train.log').open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')
    if progress is None:
        print(line, file=sys.stderr, flush=True)
    else:
        progress.write(line)
    return event
