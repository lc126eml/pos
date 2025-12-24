import os
import time
import atexit
from pathlib import Path
from filelock import FileLock

class PriorityFileLock:
    """
    Lower prio number = higher priority (prio=0 beats prio=10).
    acquire() blocks until:
      1) our ticket is at the head of the queue, then
      2) we acquire the underlying FileLock.
    """
    def __init__(self, lock_path: str, prio: int = 8, poll_s: float = 0.02):
        self.lock_path = Path(lock_path)
        self.prio = prio
        self.poll_s = poll_s

        self.queue_dir = self.lock_path.with_suffix(self.lock_path.suffix + ".queue")
        self.queue_dir.mkdir(exist_ok=True)

        self._file_lock = FileLock(str(self.lock_path))
        self._ticket = None
        self.is_locked = False

    def acquire(self, timeout: float | None = None):
        """
        timeout=None means wait forever.
        timeout in seconds applies to the whole waiting process (queue + lock).
        """
        if self.is_locked:
            return True

        start = time.monotonic()

        pid = os.getpid()
        ticket_name = f"{self.prio:03d}.{time.time_ns()}.{pid}"
        ticket = self.queue_dir / ticket_name

        # atomic create
        fd = os.open(ticket, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        self._ticket = ticket

        # Ensure best-effort cleanup on normal exit
        atexit.register(self.release)

        try:
            # Wait for our ticket to be the smallest
            print("Waiting for our ticket...")
            while True:
                if timeout is not None and (time.monotonic() - start) > timeout:
                    self.release()
                    return False

                tickets = sorted(p.name for p in self.queue_dir.iterdir() if p.is_file())
                if tickets and tickets[0] == ticket_name:
                    break

                time.sleep(self.poll_s)

            print("Acquiring the lock...")
            # Now acquire the underlying file lock
            remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - start))
            got = self._file_lock.acquire(timeout=remaining)
            if not got:
                self.release()
                return False

            self.is_locked = True
            return True

        except Exception:
            self.release()
            raise

    def release(self):
        # Release underlying lock if held
        if self.is_locked:
            try:
                self._file_lock.release()
            finally:
                self.is_locked = False

        # Remove our ticket if it exists
        if self._ticket is not None:
            try:
                self._ticket.unlink()
            except FileNotFoundError:
                pass
            finally:
                self._ticket = None
