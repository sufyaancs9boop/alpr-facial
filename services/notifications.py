"""
Async notification hub — replaces NestJS RxJS Subjects.
Consumers hold an asyncio.Queue; broadcast pushes to all.
"""
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class _EventBus:
    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers = [s for s in self._subscribers if s is not q]

    def emit(self, payload: Any):
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)


class NotificationService:
    def __init__(self):
        self.events = _EventBus()   # detection events
        self.alerts = _EventBus()   # watchlist alerts
        self.face_events = _EventBus()

    def emit_detection(self, event_data: dict):
        self.events.emit({"type": "detection", "data": event_data})

    def emit_alert(self, alert_data: dict):
        self.alerts.emit({"type": "alert", "data": alert_data})

    def emit_face_event(self, face_data: dict):
        self.face_events.emit({"type": "face", "data": face_data})


# Singleton used across the app
notifications = NotificationService()
