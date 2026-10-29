"""Verified TOS transfers and a process-shared, bounded artifact cache."""

import contextlib
import hashlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def owner_id():
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class TosStore:
    def __init__(self, config, *, endpoint=None, scratch=None, timeout=300):
        self.config = str(Path(config).resolve())
        self.endpoint = endpoint
        self.scratch = Path(scratch or "/tmp/vace-tos") / str(os.getpid())
        self.timeout = timeout

    def _run(self, arguments):
        environment = dict(os.environ)
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            environment.pop(key, None)
        command = ["tosutil", *arguments, f"-conf={self.config}"]
        if self.endpoint:
            command.append(f"-e={self.endpoint}")
        result = subprocess.run(
            command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=self.timeout,
        )
        if result.returncode:
            raise RuntimeError(f"TOS command failed ({result.returncode}): {result.stdout[-3000:]}")
        return result.stdout

    def stat(self, uri):
        output = self._run(["stat", uri, "-sbh"])
        match = re.search(r"(?m)^Size:\s*\n\s*(\d+)\s*$", output)
        if not match:
            raise RuntimeError(f"TOS stat returned no byte size: {output[-2000:]}")
        return {"bytes": int(match.group(1)), "output": output}

    def put(self, path, uri):
        path = Path(path)
        sha = file_sha256(path)
        self._run([
            "cp", str(path), uri, "-bt=fns", "-vchecksum", "-sc=STANDARD", "-acl=private",
            "-p=4", f"-meta=sha256:{sha}",
            f"-cpd={self.scratch / 'checkpoint'}", f"-o={self.scratch / 'reports'}",
        ])
        if self.stat(uri)["bytes"] != path.stat().st_size:
            raise RuntimeError(f"Remote object size differs after upload: {uri}")
        return sha

    def get(self, uri, path, *, sha256=None, expected_bytes=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._run([
            "cp", uri, str(path), "-bt=fns", "-vchecksum", "-p=4",
            f"-tempFileDir={path.parent}",
            f"-cpd={self.scratch / 'checkpoint'}", f"-o={self.scratch / 'reports'}",
        ])
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise RuntimeError(f"Downloaded object size differs: {uri}")
        if sha256 and file_sha256(path) != sha256:
            raise RuntimeError(f"Downloaded object SHA256 differs: {uri}")


class ArtifactCache:
    """Reserve bytes before writes; only evict verified remote copies without pins."""

    def __init__(self, root, limit_bytes=50_000_000_000):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit_bytes = int(limit_bytes)
        # Keep the SQLite index and filesystem allocation overhead inside the cap.
        self.data_limit = self.limit_bytes - min(256 * 1024**2, self.limit_bytes // 10)
        self.database = self.root / "index.sqlite"
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            previous = db.execute("SELECT value FROM settings WHERE key='limit_bytes'").fetchone()
            if previous and int(previous[0]) != self.limit_bytes:
                raise ValueError("Cache limit differs from its existing shared index")
            db.execute("INSERT OR IGNORE INTO settings VALUES ('limit_bytes', ?)", (str(self.limit_bytes),))
            db.execute("""CREATE TABLE IF NOT EXISTS artifacts (
                uri TEXT PRIMARY KEY, run_id TEXT, ordinal INTEGER, metadata TEXT,
                state TEXT NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL DEFAULT 0,
                reserved INTEGER NOT NULL DEFAULT 0, sha256 TEXT,
                committed INTEGER NOT NULL DEFAULT 0, owner TEXT,
                touched REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT, retry_at REAL NOT NULL DEFAULT 0
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS artifact_queue ON artifacts(run_id, state, ordinal)")
            db.execute("CREATE TABLE IF NOT EXISTS pins (uri TEXT, token TEXT PRIMARY KEY, expires REAL)")

    @contextlib.contextmanager
    def transaction(self):
        db = sqlite3.connect(self.database, timeout=60, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=60000")
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _path(self, uri):
        key = hashlib.sha256(uri.encode()).hexdigest()
        return self.root / "objects" / key[:2] / f"{key}.pt"

    def register(self, run_id, rows, prefix):
        with self.transaction() as db:
            for row in rows:
                uri = prefix.rstrip("/") + "/" + row["target_key"]
                db.execute("""INSERT OR IGNORE INTO artifacts
                    (uri,run_id,ordinal,metadata,state,path,touched) VALUES (?,?,?,?,?,?,?)""",
                    (uri, run_id, row["ordinal"], json.dumps(row, sort_keys=True),
                     "planned", str(self._path(uri)), time.time()),
                )

    def _make_space(self, db, needed):
        db.execute("DELETE FROM pins WHERE expires < ?", (time.time(),))
        used = db.execute("SELECT COALESCE(SUM(size + reserved),0) FROM artifacts").fetchone()[0]
        while used + needed > self.data_limit:
            row = db.execute("""SELECT * FROM artifacts a WHERE state='cached'
                AND committed=1 AND NOT EXISTS (SELECT 1 FROM pins p WHERE p.uri=a.uri)
                ORDER BY touched LIMIT 1""").fetchone()
            if row is None:
                return False
            Path(row["path"]).unlink(missing_ok=True)
            db.execute("UPDATE artifacts SET state='evicted', size=0 WHERE uri=?", (row["uri"],))
            used -= row["size"]
        return True

    def claim_extract(self, run_id, owner, reservation=64 * 1024**2):
        with self.transaction() as db:
            row = db.execute("""SELECT * FROM artifacts WHERE run_id=? AND state='planned'
                AND retry_at<=? ORDER BY ordinal LIMIT 1""", (run_id, time.time())).fetchone()
            if row is None or not self._make_space(db, reservation):
                return None
            db.execute("""UPDATE artifacts SET state='extracting',owner=?,reserved=?,touched=?
                WHERE uri=?""", (owner, reservation, time.time(), row["uri"]))
            return dict(row)

    def ready(self, uri, path, sha256, owner):
        size = Path(path).stat().st_size
        with self.transaction() as db:
            row = db.execute("SELECT * FROM artifacts WHERE uri=?", (uri,)).fetchone()
            if row["owner"] != owner or row["state"] != "extracting":
                raise RuntimeError("Extraction lease was lost")
            if size > row["reserved"]:
                raise RuntimeError("Artifact exceeded its reserved cache capacity")
            metadata = json.loads(row["metadata"])
            metadata["target_bytes"] = size
            db.execute("""UPDATE artifacts SET state='ready',size=?,reserved=0,sha256=?,
                owner=NULL,touched=?,error=NULL,metadata=? WHERE uri=?""",
                (size, sha256, time.time(), json.dumps(metadata, sort_keys=True), uri))

    def claim_upload(self, run_id, owner):
        with self.transaction() as db:
            row = db.execute("""SELECT * FROM artifacts WHERE run_id=? AND state='ready'
                AND retry_at<=? ORDER BY ordinal LIMIT 1""", (run_id, time.time())).fetchone()
            if row is None:
                return None
            db.execute("UPDATE artifacts SET state='uploading', owner=?, touched=? WHERE uri=?",
                       (owner, time.time(), row["uri"]))
            return dict(row)

    def uploaded(self, uri, owner, sha256):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM artifacts WHERE uri=?", (uri,)).fetchone()
            if row["owner"] != owner or row["state"] != "uploading" or row["sha256"] != sha256:
                raise RuntimeError("Upload lease/checksum mismatch")
            db.execute("""UPDATE artifacts SET state='cached',committed=1,owner=NULL,
                touched=?,attempts=0,error=NULL,retry_at=0 WHERE uri=?""", (time.time(), uri))

    def failed(self, uri, owner, error, *, upload=False):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM artifacts WHERE uri=? AND owner=?", (uri, owner)).fetchone()
            if row is None:
                return
            attempts = row["attempts"] + 1
            if upload:
                state, size = "ready", row["size"]
            else:
                Path(row["path"]).unlink(missing_ok=True)
                Path(row["path"] + ".partial").unlink(missing_ok=True)
                state, size = ("failed" if attempts >= 3 else "planned"), 0
            db.execute("""UPDATE artifacts SET state=?,size=?,reserved=0,owner=NULL,
                attempts=?,error=?,retry_at=?,touched=? WHERE uri=?""",
                (state, size, attempts, str(error)[-3000:], time.time() + min(300, 2**attempts),
                 time.time(), uri))

    def recover_owner(self, owner):
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM artifacts WHERE owner=?", (owner,)).fetchall()
            for row in rows:
                if row["state"] == "uploading":
                    db.execute("UPDATE artifacts SET state='ready',owner=NULL WHERE uri=?", (row["uri"],))
                elif row["state"] == "extracting":
                    Path(row["path"]).unlink(missing_ok=True)
                    Path(row["path"] + ".partial").unlink(missing_ok=True)
                    db.execute("""UPDATE artifacts SET state='planned',owner=NULL,size=0,reserved=0
                        WHERE uri=?""", (row["uri"],))

    def summary(self, run_id=None):
        with self.transaction() as db:
            where, args = (" WHERE run_id=?", (run_id,)) if run_id else ("", ())
            states = dict(db.execute("SELECT state,COUNT(*) FROM artifacts" + where + " GROUP BY state", args))
            uploaded = db.execute("SELECT COUNT(*) FROM artifacts" + where +
                                  (" AND" if where else " WHERE") + " committed=1", args).fetchone()[0]
            used = db.execute("SELECT COALESCE(SUM(size+reserved),0) FROM artifacts").fetchone()[0]
            total = db.execute("SELECT COUNT(*) FROM artifacts" + where, args).fetchone()[0]
            return {"total": total, "uploaded": uploaded, "states": states,
                    "cache_accounted_bytes": used, "cache_limit_bytes": self.limit_bytes}

    def evict(self, uri):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM artifacts WHERE uri=?", (uri,)).fetchone()
            pinned = db.execute("SELECT 1 FROM pins WHERE uri=? AND expires>?",
                                (uri, time.time())).fetchone()
            if not row or row["state"] != "cached" or not row["committed"] or pinned:
                raise RuntimeError("Only verified, unpinned cached objects can be evicted")
            Path(row["path"]).unlink(missing_ok=True)
            db.execute("UPDATE artifacts SET state='evicted',size=0 WHERE uri=?", (uri,))

    def export(self, run_id):
        with self.transaction() as db:
            rows = db.execute("""SELECT * FROM artifacts WHERE run_id=? AND committed=1
                ORDER BY ordinal""", (run_id,)).fetchall()
        result = []
        for row in rows:
            metadata = json.loads(row["metadata"])
            metadata.update(target_uri=row["uri"], target_sha256=row["sha256"])
            # Eviction clears the resident size; the saved metadata carries object bytes.
            metadata["target_bytes"] = row["size"] or metadata.get("target_bytes", 0)
            result.append(metadata)
        return result

    @contextlib.contextmanager
    def local(self, uri, store, *, sha256=None, expected_bytes=None, wait_seconds=900):
        """Pin a local object while the caller loads it, downloading a miss once."""
        token = owner_id()
        deadline = time.monotonic() + wait_seconds
        downloading = False
        path = self._path(uri)
        while True:
            with self.transaction() as db:
                row = db.execute("SELECT * FROM artifacts WHERE uri=?", (uri,)).fetchone()
                if row and row["state"] == "cached" and Path(row["path"]).exists():
                    if sha256 and row["sha256"] != sha256:
                        raise RuntimeError("Manifest checksum differs from cached object")
                    path = Path(row["path"])
                    db.execute("INSERT INTO pins VALUES (?,?,?)", (uri, token, time.time() + wait_seconds))
                    db.execute("UPDATE artifacts SET touched=? WHERE uri=?", (time.time(), uri))
                    break
                if row is None or row["state"] == "evicted":
                    reservation = expected_bytes or 64 * 1024**2
                    if reservation > self.data_limit:
                        raise ValueError("Object is larger than cache capacity")
                    if self._make_space(db, reservation):
                        if row is None:
                            db.execute("""INSERT INTO artifacts (uri,state,path,reserved,owner,touched,committed)
                                VALUES (?,?,?,?,?,?,1)""", (uri, "downloading", str(path), reservation, token, time.time()))
                        else:
                            db.execute("""UPDATE artifacts SET state='downloading',reserved=?,owner=?,touched=?
                                WHERE uri=?""", (reservation, token, time.time(), uri))
                        downloading = True
                        break
            if time.monotonic() >= deadline:
                raise TimeoutError("Cache is full of pending/pinned files or this object is not uploaded yet")
            time.sleep(0.5)
        if downloading:
            temporary = path.with_suffix(".download")
            try:
                store.get(uri, temporary, sha256=sha256, expected_bytes=expected_bytes)
                size, actual_sha = temporary.stat().st_size, file_sha256(temporary)
                with self.transaction() as db:
                    row = db.execute("SELECT * FROM artifacts WHERE uri=?", (uri,)).fetchone()
                    if size > row["reserved"]:
                        raise RuntimeError("Download exceeded reserved cache capacity")
                    temporary.replace(path)
                    db.execute("""UPDATE artifacts SET state='cached',size=?,reserved=0,sha256=?,
                        owner=NULL,touched=? WHERE uri=?""", (size, actual_sha, time.time(), uri))
                    db.execute("INSERT INTO pins VALUES (?,?,?)", (uri, token, time.time() + wait_seconds))
            except BaseException:
                temporary.unlink(missing_ok=True)
                with self.transaction() as db:
                    db.execute("""UPDATE artifacts SET state='evicted',size=0,reserved=0,owner=NULL
                        WHERE uri=? AND owner=?""", (uri, token))
                raise
        try:
            yield path
        finally:
            with self.transaction() as db:
                db.execute("DELETE FROM pins WHERE token=?", (token,))
