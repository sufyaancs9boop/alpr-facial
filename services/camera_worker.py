"""
Camera worker service — spawns an asyncio task per active camera.
Replaces NestJS CameraWorkerService.
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CameraWorkerService:
    def __init__(self, alpr_service_factory):
        self._alpr_factory = alpr_service_factory
        self._tasks: dict[str, asyncio.Task] = {}
        self._stop_flags: dict[str, bool] = {}

    def is_streaming(self, camera_id: str) -> bool:
        task = self._tasks.get(camera_id)
        return task is not None and not task.done()

    async def start_worker(self, camera):
        if self.is_streaming(camera.id):
            return
        self._stop_flags[camera.id] = False
        task = asyncio.create_task(
            self._run_camera(camera),
            name=f"camera-{camera.id}",
        )
        self._tasks[camera.id] = task
        task.add_done_callback(lambda t: self._on_task_done(camera.id, t))
        logger.info("Camera worker started: %s (%s)", camera.name, camera.url)

    async def stop_worker(self, camera_id: str):
        self._stop_flags[camera_id] = True
        task = self._tasks.pop(camera_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("Camera worker stopped: %s", camera_id)

    async def stop_all(self):
        for camera_id in list(self._tasks.keys()):
            await self.stop_worker(camera_id)

    async def _run_camera(self, camera):
        alpr = self._alpr_factory()
        url = camera.testVideoPath or camera.url
        frame_step = camera.frameStep or 5

        def should_continue():
            return not self._stop_flags.get(camera.id, True)

        # Loop video file or stream indefinitely until stopped
        while should_continue():
            try:
                async for _ in alpr.detect_live_stream(
                    url,
                    frame_step=frame_step,
                    camera_id=camera.id,
                    camera_name=camera.name,
                    should_continue=should_continue,
                ):
                    if not should_continue():
                        return
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Camera %s worker error: %s — retrying in 5s", camera.name, exc)
                await asyncio.sleep(5)
            if not should_continue():
                return
            await asyncio.sleep(0.2)

    def _on_task_done(self, camera_id: str, task: asyncio.Task):
        self._tasks.pop(camera_id, None)
        if not task.cancelled() and task.exception():
            logger.error("Camera worker %s crashed: %s", camera_id, task.exception())
