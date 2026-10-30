"""Process locks released by the OS, including after an interrupted worker."""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    stream.seek(0)
    if path.stat().st_size == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    acquired = False
    try:
        try:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        acquired = True
        yield
    except OSError as exc:
        if not acquired:
            raise ValueError("worker_already_running") from exc
        raise
    finally:
        if acquired:
            stream.seek(0)
            try:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)
            except ImportError:
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()
