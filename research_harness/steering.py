from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional, TextIO

from .store import ArtifactStore


@dataclass
class SteeringHandle:
    stop_event: threading.Event
    thread: threading.Thread

    def stop(self) -> None:
        self.stop_event.set()


def start_cli_steering(
    store: ArtifactStore,
    *,
    input_stream: Optional[TextIO] = None,
    output_func: Callable[[str], None] = print,
) -> Optional[SteeringHandle]:
    stream = input_stream or sys.stdin
    if not getattr(stream, "isatty", lambda: False)():
        return None
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_steering_loop,
        args=(store, stream, output_func, stop_event),
        daemon=True,
        name="autore-user-steering",
    )
    thread.start()
    output_func("Steering: type /article <url | title | note> or /steer <note>; /help for commands.")
    return SteeringHandle(stop_event=stop_event, thread=thread)


def _steering_loop(
    store: ArtifactStore,
    stream: TextIO,
    output_func: Callable[[str], None],
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            line = stream.readline()
        except Exception:
            return
        if not line:
            return
        command = line.strip()
        if not command:
            continue
        lowered = command.lower()
        if lowered in {"/help", "/?"}:
            output_func(
                "Steering commands: /article <url | title | why it matters>, "
                "/steer <instruction or observation>, /note <context>. "
                "New input is applied at the next round boundary."
            )
            continue
        if lowered in {"/quit", "/exit"}:
            output_func("Steering listener stopped for this run.")
            return
        if lowered.startswith(("/article", "/steer", "/note")):
            store.append_user_steering(command)
            output_func("Steering captured; it will be ingested before the next proposal round.")
            continue
        output_func("Use /steer, /article, or /note so the run can distinguish steering from shell input.")
