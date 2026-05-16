"""Structured event generation for HomeMindAI perception."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
import threading


LOGGER = logging.getLogger(__name__)


class EventType(str, Enum):
    """Supported perception event types."""

    KNOWN_PERSON_DETECTED = "KNOWN_PERSON_DETECTED"
    UNKNOWN_PERSON_DETECTED = "UNKNOWN_PERSON_DETECTED"
    PERSON_TRACKED = "PERSON_TRACKED"


@dataclass
class PerceptionEvent:
    """Event payload emitted by the perception stack."""

    event_type: EventType
    timestamp: str
    camera_id: str
    identity: str
    confidence: float
    snapshot_path: str | None
    tracking_id: str | None


class EventEngine:
    """In-memory event collector with structured logging."""

    def __init__(self, max_events: int = 500) -> None:
        self._events: deque[PerceptionEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def emit(
        self,
        event_type: EventType,
        camera_id: str,
        identity: str,
        confidence: float,
        snapshot_path: str | None,
        tracking_id: str | None,
    ) -> PerceptionEvent:
        """Create, store, and log a perception event."""

        event = PerceptionEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            camera_id=camera_id,
            identity=identity,
            confidence=confidence,
            snapshot_path=snapshot_path,
            tracking_id=tracking_id,
        )
        with self._lock:
            self._events.appendleft(event)

        LOGGER.info("Perception event emitted", extra=asdict(event))
        return event

    def list_recent(self, limit: int = 25) -> list[dict[str, str | float | None]]:
        """Return recent events as JSON-ready dictionaries."""

        with self._lock:
            events = list(self._events)[:limit]
        return [
            {
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "camera_id": event.camera_id,
                "identity": event.identity,
                "confidence": event.confidence,
                "snapshot_path": event.snapshot_path,
                "tracking_id": event.tracking_id,
            }
            for event in events
        ]