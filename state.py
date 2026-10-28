"""每次 SQLite 操作有独立连接。异步调用者应使用 asyncio.to_thread。"""

from contextlib import contextmanager
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import hashlib
import json
import sqlite3
import time

from .models import IncomingReply, VideoItem, next_local_day, normalized


RETRY_OUTCOMES = {"read_error", "model_error", "model_timeout", "invalid_decision"}
MAX_FAILURES = 3  # 首次失败之后，最多再检查两次。
RETRY_INTERVAL = 600


class StateStore:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize(
        self, now: Optional[float] = None, timezone: str = "Asia/Shanghai", wait_hours: int = 48, wait_limit: int = 24
    ) -> None:
        now = time.time() if now is None else now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            seed_analyses = not db.execute("SELECT 1 FROM sqlite_master WHERE name='analyses'").fetchone()
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO meta VALUES ('paused', '1');
                INSERT OR IGNORE INTO meta VALUES ('pause_reason', '首次运行，请先登录并启动');
                INSERT OR IGNORE INTO meta(key,value)
                SELECT 'pause_kind', CASE
                    WHEN (SELECT value FROM meta WHERE key='paused')='0' THEN 'running'
                    WHEN (SELECT value FROM meta WHERE key='pause_reason') LIKE '首次运行%' THEN 'initial'
                    WHEN (SELECT value FROM meta WHERE key='pause_reason')='正在扫码登录' THEN 'login'
                    WHEN (SELECT value FROM meta WHERE key='pause_reason')='已在配置中禁用' THEN 'disabled'
                    WHEN (SELECT value FROM meta WHERE key='pause_reason')='管理员手动暂停' THEN 'manual'
                    ELSE 'error' END;
                CREATE TABLE IF NOT EXISTS browse_decisions (
                    id INTEGER PRIMARY KEY, uid TEXT NOT NULL, ts REAL NOT NULL,
                    should_browse INTEGER NOT NULL, reason TEXT NOT NULL, next_check_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS browse_decisions_account ON browse_decisions(uid,id DESC);
                CREATE TABLE IF NOT EXISTS visits (
                    uid TEXT NOT NULL, bvid TEXT NOT NULL, mode TEXT NOT NULL,
                    ts REAL NOT NULL, outcome TEXT NOT NULL, detail TEXT NOT NULL,
                    PRIMARY KEY (uid, bvid, mode)
                );
                CREATE INDEX IF NOT EXISTS visits_time ON visits(uid, ts);
                CREATE TABLE IF NOT EXISTS attempts (
                    uid TEXT NOT NULL, bvid TEXT NOT NULL, ts REAL NOT NULL,
                    text TEXT NOT NULL, text_hash TEXT NOT NULL,
                    status TEXT NOT NULL, rpid TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (uid, bvid), UNIQUE (uid, text_hash)
                );
                CREATE INDEX IF NOT EXISTS attempts_time ON attempts(uid, ts);
                CREATE TABLE IF NOT EXISTS reply_poll (uid TEXT PRIMARY KEY, last_checked REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS reply_inbox (
                    uid TEXT NOT NULL, source_rpid INTEGER NOT NULL, bvid TEXT NOT NULL,
                    root_rpid INTEGER NOT NULL, received_at REAL NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(uid,source_rpid)
                );
                CREATE TABLE IF NOT EXISTS reply_decisions (
                    uid TEXT NOT NULL, source_rpid INTEGER NOT NULL, mode TEXT NOT NULL,
                    ts REAL NOT NULL, outcome TEXT NOT NULL, text TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(uid,source_rpid,mode)
                );
                CREATE TABLE IF NOT EXISTS reply_attempts (
                    uid TEXT NOT NULL, source_rpid INTEGER NOT NULL, bvid TEXT NOT NULL,
                    root_rpid INTEGER NOT NULL, ts REAL NOT NULL, text TEXT NOT NULL,
                    text_hash TEXT NOT NULL, status TEXT NOT NULL, rpid TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(uid,source_rpid), UNIQUE(uid,text_hash)
                );
                CREATE INDEX IF NOT EXISTS reply_attempts_time ON reply_attempts(uid,ts);
                CREATE TABLE IF NOT EXISTS video_shares (
                    uid TEXT NOT NULL, bvid TEXT NOT NULL, group_id TEXT NOT NULL,
                    ts REAL NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '', message_id TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(uid,bvid,group_id)
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY, uid TEXT NOT NULL, kind TEXT NOT NULL,
                    item_id TEXT NOT NULL, mode TEXT NOT NULL, ts REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS analyses_account_time ON analyses(uid,kind,ts);
                CREATE TABLE IF NOT EXISTS read_recovery (
                    uid TEXT NOT NULL, stage TEXT NOT NULL, failures INTEGER NOT NULL,
                    retry_at REAL NOT NULL, reason TEXT NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(uid,stage)
                );
                CREATE TABLE IF NOT EXISTS evidence_cache (
                    uid TEXT NOT NULL, bvid TEXT NOT NULL, expires_at REAL NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(uid,bvid)
                );
                CREATE TABLE IF NOT EXISTS model_calls (
                    id INTEGER PRIMARY KEY, uid TEXT NOT NULL, stage TEXT NOT NULL, ts REAL NOT NULL,
                    elapsed REAL NOT NULL, outcome TEXT NOT NULL, usage TEXT NOT NULL DEFAULT '{}'
                );
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(browse_decisions)")}
            if "selected_bvids" not in columns:
                db.execute("ALTER TABLE browse_decisions ADD COLUMN selected_bvids TEXT NOT NULL DEFAULT '[]'")
            for table in ("visits", "reply_decisions"):
                columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
                for column, declaration in (
                    ("updated_at", "REAL NOT NULL DEFAULT 0"),
                    ("failures", "INTEGER NOT NULL DEFAULT 0"),
                    ("retry_at", "REAL NOT NULL DEFAULT 0"),
                ):
                    if column not in columns:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
                if "updated_at" not in columns:
                    db.execute(f"UPDATE {table} SET updated_at=ts")
                    # 旧记录只有首次时间；能从发送意图确认的后续时间才补入。
                    if table == "visits":
                        db.execute(
                            "UPDATE visits SET updated_at=max(ts,coalesce((SELECT a.ts FROM attempts a "
                            "WHERE a.uid=visits.uid AND a.bvid=visits.bvid AND visits.mode='publish'),ts))"
                        )
                    else:
                        db.execute(
                            "UPDATE reply_decisions SET updated_at=max(ts,coalesce((SELECT a.ts FROM reply_attempts a "
                            "WHERE a.uid=reply_decisions.uid AND a.source_rpid=reply_decisions.source_rpid "
                            "AND reply_decisions.mode='publish'),ts))"
                        )
                    db.execute(
                        f"UPDATE {table} SET failures=1 "
                        "WHERE outcome IN ('read_error','model_error','model_timeout','invalid_decision')"
                    )
                    retry_filter = "outcome IN ('model_error','model_timeout','invalid_decision')"
                    if table == "visits":
                        retry_filter += (
                            " OR (outcome='read_error' AND (detail LIKE '%-504%' "
                            "OR detail LIKE '%超时%' OR detail LIKE '%网络连接失败%'))"
                        )
                    db.execute(f"UPDATE {table} SET retry_at=updated_at+? WHERE {retry_filter}", (RETRY_INTERVAL,))
            columns = {row["name"] for row in db.execute("PRAGMA table_info(visits)")}
            if "expires_at" not in columns:
                db.execute("ALTER TABLE visits ADD COLUMN expires_at REAL NOT NULL DEFAULT 0")
            # 只迁移从未产生发送意图的旧额度阻塞记录。缺少元数据时，到期后等再次推荐重评。
            rows = db.execute(
                "SELECT v.* FROM visits v WHERE v.mode='publish' AND v.outcome='blocked' "
                "AND NOT EXISTS(SELECT 1 FROM attempts a WHERE a.uid=v.uid AND a.bvid=v.bvid)"
            ).fetchall()
            for row in rows:
                try:
                    detail = json.loads(row["detail"])
                except (ValueError, TypeError):
                    detail = {"reason": row["detail"]}
                if not isinstance(detail, dict):
                    continue
                if detail.get("block_reason", detail.get("reason")) != "daily_limit":
                    continue
                expiry = row["updated_at"] + wait_hours * 3600 if isinstance(detail.get("video"), dict) else 0
                db.execute(
                    "UPDATE visits SET outcome='quota_wait',retry_at=?,expires_at=? WHERE uid=? AND bvid=? AND mode='publish'",
                    (next_local_day(row["updated_at"], timezone), expiry, row["uid"], row["bvid"]),
                )
            rows = db.execute(
                "SELECT uid,bvid,ts,detail FROM visits WHERE mode='publish' "
                "AND outcome IN ('quota_wait','deferred') AND expires_at=0"
            ).fetchall()
            for row in rows:
                try:
                    detail = json.loads(row["detail"])
                except (ValueError, TypeError):
                    continue
                if isinstance(detail, dict) and isinstance(detail.get("video"), dict):
                    db.execute(
                        "UPDATE visits SET expires_at=? WHERE uid=? AND bvid=? AND mode='publish'",
                        (row["ts"] + wait_hours * 3600, row["uid"], row["bvid"]),
                    )
            for row in db.execute(
                "SELECT DISTINCT uid FROM visits WHERE mode='publish' AND outcome IN ('quota_wait','deferred')"
            ).fetchall():
                self._trim_waits(db, row["uid"], now, wait_limit)
            # 旧版容量淘汰也写成 expired；只有时间戳能明确证明未到期时才更正文案。
            db.execute("UPDATE visits SET outcome='evicted' WHERE mode='publish' AND outcome='expired' "
                       "AND updated_at>0 AND expires_at>updated_at")
            if seed_analyses:
                db.execute("INSERT INTO analyses(uid,kind,item_id,mode,ts) SELECT uid,'video',bvid,mode,ts FROM visits")
                db.execute(
                    "INSERT INTO analyses(uid,kind,item_id,mode,ts) "
                    "SELECT uid,'reply',cast(source_rpid AS TEXT),mode,ts FROM reply_decisions"
                )
            db.execute("CREATE INDEX IF NOT EXISTS visits_updated ON visits(uid,updated_at)")
            # 请求发出前已落盘；进程中断后无法知道平台是否接收，绝不重试该视频。
            db.execute("UPDATE attempts SET status='unknown' WHERE status='pending'")
            db.execute("UPDATE reply_attempts SET status='unknown' WHERE status='pending'")
            # 旧版把未发送的模型超时记录为终态失败；允许后续检查重新判断。
            db.execute(
                "UPDATE reply_decisions SET outcome='model_timeout' "
                "WHERE outcome='model_error' AND reason='模型调用失败：TimeoutError'"
            )
            self._recover_interrupted(db, now)
            db.execute(
                "UPDATE video_shares SET status='unknown',detail='进程中断，群分享结果待核对；不会自动重发' WHERE status='pending'"
            )
        self.path.chmod(0o600)

    def _recover_interrupted(self, db: sqlite3.Connection, now: float) -> None:
        # reading 已计入分析额度；中断计一次失败，不因反复重载重置重查上限。
        for table, attempts, key, outcome, field in (
            ("visits", "attempts", "bvid", "read_error", "detail"),
            ("reply_decisions", "reply_attempts", "source_rpid", "model_error", "reason"),
        ):
            rows = db.execute(
                f"SELECT d.* FROM {table} d WHERE d.outcome='reading' "
                f"AND NOT EXISTS(SELECT 1 FROM {attempts} a WHERE a.uid=d.uid AND a.{key}=d.{key})"
            ).fetchall()
            for row in rows:
                failures = row["failures"] + 1
                retry_at = now + RETRY_INTERVAL * failures if failures < MAX_FAILURES else 0
                reason = "上次分析因任务或进程中断未完成，尚未尝试发送"
                if table == "visits":
                    try:
                        detail = json.loads(row["detail"])
                    except (ValueError, TypeError):
                        detail = {}
                    if not isinstance(detail, dict):
                        detail = {}
                    detail["reason"] = reason
                    reason = json.dumps(detail, ensure_ascii=False)
                db.execute(
                    f"UPDATE {table} SET outcome=?,{field}=?,updated_at=?,failures=?,retry_at=? "
                    f"WHERE uid=? AND {key}=? AND mode=?",
                    (outcome, reason, now, failures, retry_at, row["uid"], row[key], row["mode"]),
                )

    def set_pause(self, paused: bool, reason: str, kind: str = "error") -> None:
        with self.connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                [
                    ("paused", "1" if paused else "0"),
                    ("pause_reason", reason[:500]),
                    ("pause_kind", kind if paused else "running"),
                ],
            )

    def reserve_share(self, uid: str, bvid: str, group_id: str, ts: float, title: str) -> bool:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute(
                "SELECT 1 FROM attempts WHERE uid=? AND bvid=? AND status='sent' AND rpid!=''", (uid, bvid)
            ).fetchone():
                return False
            cursor = db.execute(
                "INSERT OR IGNORE INTO video_shares(uid,bvid,group_id,ts,title,status) VALUES (?,?,?,?,?,'pending')",
                (uid, bvid, group_id, ts, title[:300]),
            )
            return cursor.rowcount == 1

    def finish_share(self, uid: str, bvid: str, group_id: str, status: str, detail: str, message_id: str = "") -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE video_shares SET status=?,detail=?,message_id=? WHERE uid=? AND bvid=? AND group_id=?",
                (status, detail[:500], message_id, uid, bvid, group_id),
            )

    def latest_share(self, uid: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM video_shares WHERE uid=? ORDER BY ts DESC LIMIT 1", (uid,)).fetchone()
            return dict(row) if row else None

    def share_context(self, uid: str, bvid: str) -> Dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT a.text,v.detail FROM attempts a LEFT JOIN visits v "
                "ON v.uid=a.uid AND v.bvid=a.bvid AND v.mode='publish' "
                "WHERE a.uid=? AND a.bvid=? AND a.status='sent' AND a.rpid!=''",
                (uid, bvid),
            ).fetchone()
            if row is None:
                raise ValueError("没有已提交成功的评论，不能生成群分享")
            detail = json.loads(row["detail"]) if row["detail"] else {}
            recent = db.execute(
                "SELECT detail FROM video_shares WHERE uid=? AND status='sent' ORDER BY ts DESC LIMIT 20", (uid,)
            ).fetchall()
            recent_texts = list(
                dict.fromkeys(
                    r["detail"].removeprefix("分享文案：") for r in recent if r["detail"].startswith("分享文案：")
                )
            )[:4]
            cached = db.execute(
                "SELECT payload FROM evidence_cache WHERE uid=? AND bvid=? AND expires_at>?", (uid, bvid, time.time())
            ).fetchone()
            return {
                "my_comment": row["text"],
                "reason": detail.get("reason", ""),
                "content_evidence": json.loads(cached["payload"]) if cached else detail.get("evidence_quote", ""),
                "recent_shares": recent_texts,
            }

    def control(self) -> Dict[str, str]:
        with self.connect() as db:
            return {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM meta")}

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def visit(
        self, uid: str, bvid: str, mode: str, ts: float, outcome: str, detail: str, retryable: Optional[bool] = None
    ) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT failures FROM visits WHERE uid=? AND bvid=? AND mode=?", (uid, bvid, mode)
            ).fetchone()
            failures = previous["failures"] if previous else 0
            if outcome in RETRY_OUTCOMES:
                failures += 1
            if retryable is None:
                retryable = outcome in RETRY_OUTCOMES
            retry_at = ts + RETRY_INTERVAL * max(1, failures) if retryable and failures < MAX_FAILURES else 0
            db.execute(
                "INSERT INTO visits(uid,bvid,mode,ts,outcome,detail,updated_at,failures,retry_at) VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(uid,bvid,mode) DO UPDATE SET outcome=excluded.outcome,detail=excluded.detail,"
                "updated_at=excluded.updated_at,failures=excluded.failures,retry_at=excluded.retry_at",
                (uid, bvid, mode, ts, outcome, detail, ts, failures, retry_at),
            )
            if outcome == "reading" or previous is None:
                db.execute(
                    "INSERT INTO analyses(uid,kind,item_id,mode,ts) VALUES (?,'video',?,?,?)", (uid, bvid, mode, ts)
                )

    def seen(self, uid: str, bvid: str, mode: str, include_deferred: bool = False, now: Optional[float] = None) -> bool:
        check_deadline = now is not None
        now = time.time() if now is None else now
        with self.connect() as db:
            if db.execute("SELECT 1 FROM attempts WHERE uid=? AND bvid=?", (uid, bvid)).fetchone():
                return True
            row = db.execute("SELECT * FROM visits WHERE uid=? AND bvid=? AND mode=?", (uid, bvid, mode)).fetchone()
            if row is None:
                return False
            if row["outcome"] == "quota_wait":
                return (include_deferred and row["expires_at"] > 0) or row["retry_at"] > now or (
                    row["expires_at"] > 0 and row["expires_at"] <= now
                )
            if row["outcome"] == "blocked" and row["detail"] == "cooldown":
                return include_deferred and row["retry_at"] > now
            if row["outcome"] == "deferred":
                return include_deferred or (
                    check_deadline and (row["retry_at"] > now or (row["expires_at"] > 0 and row["expires_at"] <= now))
                )
            return not (
                row["outcome"] in RETRY_OUTCOMES and row["failures"] < MAX_FAILURES and 0 < row["retry_at"] <= now
            )

    def retry_videos(
        self, uid: str, mode: str, now: float, eligible: Optional[Callable[[VideoItem], bool]] = None
    ) -> List[VideoItem]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT v.detail FROM visits v WHERE v.uid=? AND v.mode=? "
                "AND v.outcome IN ('read_error','model_error','model_timeout','invalid_decision') "
                "AND v.failures<? AND v.retry_at>0 AND v.retry_at<=? "
                "AND NOT EXISTS(SELECT 1 FROM attempts a WHERE a.uid=v.uid AND a.bvid=v.bvid) "
                "ORDER BY v.retry_at",
                (uid, mode, MAX_FAILURES, now),
            )
            result = []
            for row in rows:
                detail = json.loads(row["detail"]) if row["detail"].startswith("{") else {}
                # 旧版未保存视频元数据，只能等待该视频再次被推荐。
                if isinstance(detail, dict) and "video" in detail:
                    item = VideoItem(**detail["video"])
                    if eligible is not None and not eligible(item):
                        continue
                    result.append(item)
                    if len(result) == 3:
                        break
            return result

    @staticmethod
    def _candidate_score(row: Dict[str, Any], now: float) -> float:
        """兴趣为主，兼顾新鲜度及等待机会；不保证任何候选一定发送。"""
        try:
            fields = json.loads(row["detail"])
        except (ValueError, TypeError):
            fields = {}
        interest = fields.get("interest", 0) if isinstance(fields, dict) else 0
        interest = float(interest) if type(interest) in (int, float) and 0 <= interest <= 1 else 0
        hours = max(0, now - row["ts"]) / 3600
        return interest * .75 + .15 / (1 + hours / 24) + .10 * min(hours / 12, 1)

    def candidates(
        self, uid: str, eligible: Optional[Callable[[VideoItem], bool]] = None,
        now: Optional[float] = None, limit: int = 24, *, ready_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """只读取从未产生发送意图的候选；清理到期、已屏蔽项后择优排序。"""
        check_deadline = now is not None
        now = time.time() if now is None else now
        with self.connect() as db:
            rows = db.execute(
                "SELECT v.* FROM visits v WHERE v.uid=? AND v.mode='publish' "
                "AND v.outcome IN ('deferred','quota_wait') AND NOT EXISTS "
                "(SELECT 1 FROM attempts a WHERE a.uid=v.uid AND a.bvid=v.bvid)", (uid,),
            ).fetchall()
            result = []
            for raw in rows:
                row = dict(raw)
                if check_deadline and row["expires_at"] and row["expires_at"] <= now:
                    db.execute("UPDATE visits SET outcome='expired',updated_at=? WHERE uid=? AND bvid=? AND mode='publish'",
                               (now, uid, row["bvid"]))
                    continue
                try:
                    fields = json.loads(row["detail"])
                    item = VideoItem(**fields["video"])
                except (ValueError, TypeError, KeyError):
                    continue
                if eligible is not None and not eligible(item):
                    db.execute("UPDATE visits SET outcome='cancelled',updated_at=? WHERE uid=? AND bvid=? AND mode='publish'",
                               (now, uid, row["bvid"]))
                    continue
                if ready_only and check_deadline and row["retry_at"] > now:
                    continue
                row.update(item=item, fields=fields)
                result.append(row)
            result.sort(key=lambda row: (self._candidate_score(row, now), -row["ts"], row["bvid"]), reverse=True)
            return result[:limit]

    def deferred_videos(
        self, uid: str, eligible: Optional[Callable[[VideoItem], bool]] = None,
        now: Optional[float] = None, limit: int = 24,
    ) -> List[VideoItem]:
        return [row["item"] for row in self.candidates(uid, eligible, now, limit)]

    def record_candidate_selection(self, uid: str, candidates: List[Dict[str, Any]], selected: str,
                                   reason: str, now: float, next_check: float) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", ("candidate_check:" + uid, str(next_check)))
            for row in candidates:
                fields = dict(row["fields"])
                fields.update(selection_reason=reason, selection_time=now,
                              selected=row["bvid"] == selected)
                # 选择未命中不是浏览，不更新排序时间、兴趣或分析计数。
                db.execute("UPDATE visits SET detail=? WHERE uid=? AND bvid=? AND mode='publish' "
                           "AND outcome IN ('deferred','quota_wait')",
                           (json.dumps(fields, ensure_ascii=False), uid, row["bvid"]))

    def wait_video(
        self,
        uid: str,
        item: VideoItem,
        now: float,
        detail: str,
        retry_at: float,
        hours: int,
        limit: int,
        outcome: str = "quota_wait",
    ) -> None:
        self.visit(uid, item.bvid, "publish", now, outcome, detail, False)
        with self.connect() as db:
            db.execute(
                "UPDATE visits SET retry_at=?,expires_at=CASE WHEN expires_at>0 THEN expires_at ELSE ts+? END "
                "WHERE uid=? AND bvid=? AND mode='publish'",
                (retry_at, hours * 3600, uid, item.bvid),
            )
            self._trim_waits(db, uid, now, limit)

    @classmethod
    def _trim_waits(cls, db: sqlite3.Connection, uid: str, now: float, limit: int) -> None:
        db.execute("UPDATE visits SET outcome='expired',updated_at=? WHERE uid=? AND mode='publish' "
                   "AND outcome IN ('quota_wait','deferred') AND expires_at>0 AND expires_at<=?", (now, uid, now))
        rows = db.execute("SELECT * FROM visits WHERE uid=? AND mode='publish' "
                          "AND outcome IN ('quota_wait','deferred')", (uid,)).fetchall()
        ranked = sorted(rows, key=lambda row: (cls._candidate_score(row, now), row["ts"], row["bvid"]), reverse=True)
        for row in ranked[limit:]:
            db.execute("UPDATE visits SET outcome='evicted',updated_at=? WHERE uid=? AND bvid=? AND mode='publish'",
                       (now, uid, row["bvid"]))

    def postpone_waits(self, uid: str, until: float) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE visits SET retry_at=max(retry_at,?) WHERE uid=? AND mode='publish' "
                "AND outcome IN ('quota_wait','deferred')",
                (until, uid),
            )

    def read_wait(self, uid: str, stage: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM read_recovery WHERE uid=? AND stage=?", (uid, stage)).fetchone()
            return dict(row) if row else None

    def read_failed(self, uid: str, stage: str, now: float, reason: str, base: int) -> Dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT failures FROM read_recovery WHERE uid=? AND stage=?", (uid, stage)).fetchone()
            failures = row["failures"] + 1 if row else 1
            delay = base * 2 ** (failures - 1) if failures <= 4 else 3600
            db.execute(
                "INSERT OR REPLACE INTO read_recovery VALUES (?,?,?,?,?,?)",
                (uid, stage, failures, now + delay, reason[:500], now),
            )
            return {"failures": failures, "retry_at": now + delay, "reason": reason, "stage": stage}

    def read_succeeded(self, uid: str, stage: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM read_recovery WHERE uid=? AND stage=?", (uid, stage))

    def recoveries(self, uid: str) -> List[Dict[str, Any]]:
        with self.connect() as db:
            return [
                dict(row) for row in db.execute("SELECT * FROM read_recovery WHERE uid=? ORDER BY retry_at", (uid,))
            ]

    def cache_evidence(self, uid: str, bvid: str, payload: Dict[str, Any], now: float, days: int) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM evidence_cache WHERE expires_at<=?", (now,))
            db.execute(
                "INSERT OR REPLACE INTO evidence_cache VALUES (?,?,?,?)",
                (uid, bvid, now + days * 86400, json.dumps(payload, ensure_ascii=False)),
            )

    def content_context(self, uid: str, bvid: str, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute("DELETE FROM evidence_cache WHERE expires_at<=?", (now,))
            row = db.execute("SELECT payload FROM evidence_cache WHERE uid=? AND bvid=?", (uid, bvid)).fetchone()
            return json.loads(row["payload"]) if row else {}

    def clean_evidence(self, now: float) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM evidence_cache WHERE expires_at<=?", (now,))

    def model_call(
        self, uid: str, stage: str, now: float, elapsed: float, outcome: str, usage: Optional[Dict[str, Any]] = None
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO model_calls(uid,stage,ts,elapsed,outcome,usage) VALUES (?,?,?,?,?,?)",
                (uid, stage, now, elapsed, outcome, json.dumps(usage or {}, ensure_ascii=False)),
            )

    def model_usage(self, uid: str, since: float) -> Dict[str, Any]:
        with self.connect() as db:
            rows = db.execute("SELECT elapsed,usage FROM model_calls WHERE uid=? AND ts>=?", (uid, since)).fetchall()
            return {
                "calls": len(rows),
                "elapsed_seconds": round(sum(r["elapsed"] for r in rows), 3),
                "reported_usage": [json.loads(r["usage"]) for r in rows if r["usage"] != "{}"],
            }

    def stats(self, uid: str, day_start: float) -> Dict[str, Any]:
        with self.connect() as db:
            views = db.execute(
                "SELECT count(*) FROM analyses WHERE uid=? AND kind='video' AND ts>=?", (uid, day_start)
            ).fetchone()[0]
            attempts = db.execute("SELECT count(*) FROM attempts WHERE uid=? AND ts>=?", (uid, day_start)).fetchone()[0]
            unresolved = [
                r[0]
                for r in db.execute("SELECT bvid FROM attempts WHERE uid=? AND status IN ('pending','unknown')", (uid,))
            ]
            last = db.execute("SELECT coalesce(max(ts),0) FROM attempts WHERE uid=?", (uid,)).fetchone()[0]
            unresolved += [
                r[0]
                for r in db.execute(
                    "SELECT bvid FROM reply_attempts WHERE uid=? AND status IN ('pending','unknown')", (uid,)
                )
            ]
            reply_attempts = db.execute(
                "SELECT count(*) FROM reply_attempts WHERE uid=? AND ts>=?", (uid, day_start)
            ).fetchone()[0]
            reply_analyses = db.execute(
                "SELECT count(*) FROM analyses WHERE uid=? AND kind='reply' AND ts>=?", (uid, day_start)
            ).fetchone()[0]
            reply_last = db.execute("SELECT coalesce(max(ts),0) FROM reply_attempts WHERE uid=?", (uid,)).fetchone()[0]
            return {
                "views": views,
                "attempts": attempts,
                "unresolved": sorted(set(unresolved)),
                "last_attempt": max(last, reply_last),
                "reply_attempts": reply_attempts,
                "reply_analyses": reply_analyses,
            }

    def own_comments(self, uid: str) -> Dict[str, Dict[str, Any]]:
        with self.connect() as db:
            roots = db.execute(
                "SELECT bvid,rpid,text FROM attempts WHERE uid=? AND status='sent' AND rpid!=''", (uid,)
            ).fetchall()
            own = {
                r["rpid"]: {"bvid": r["bvid"], "root_rpid": int(r["rpid"]), "root_text": r["text"], "text": r["text"]}
                for r in roots
                if r["rpid"].isdigit() and int(r["rpid"]) > 0
            }
            replies = db.execute(
                "SELECT bvid,root_rpid,rpid,text FROM reply_attempts WHERE uid=? AND status='sent' AND rpid!=''", (uid,)
            )
            for row in replies:
                root = own.get(str(row["root_rpid"]))
                if root and row["rpid"].isdigit() and int(row["rpid"]) > 0:
                    own[row["rpid"]] = {
                        "bvid": row["bvid"],
                        "root_rpid": row["root_rpid"],
                        "root_text": root["root_text"],
                        "text": row["text"],
                    }
            return own

    def last_reply_check(self, uid: str) -> float:
        with self.connect() as db:
            row = db.execute("SELECT last_checked FROM reply_poll WHERE uid=?", (uid,)).fetchone()
            return row[0] if row else 0

    def set_reply_check(self, uid: str, now: float) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO reply_poll VALUES (?,?)", (uid, now))

    def receive_replies(self, uid: str, messages: List[IncomingReply]) -> None:
        with self.connect() as db:
            db.executemany(
                "INSERT OR IGNORE INTO reply_inbox VALUES (?,?,?,?,?,?)",
                [
                    (uid, m.source_rpid, m.bvid, m.root_rpid, m.received_at, json.dumps(asdict(m), ensure_ascii=False))
                    for m in messages
                ],
            )

    def pending_replies(self, uid: str, mode: str, since: float, now: Optional[float] = None) -> List[IncomingReply]:
        now = time.time() if now is None else now
        with self.connect() as db:
            rows = db.execute(
                "SELECT i.payload FROM reply_inbox i WHERE i.uid=? AND i.received_at>=? "
                "AND NOT EXISTS(SELECT 1 FROM reply_decisions d WHERE d.uid=i.uid AND d.source_rpid=i.source_rpid AND d.mode=? "
                "AND NOT (d.outcome IN ('model_timeout','model_error','invalid_decision') AND d.failures<? "
                "AND d.retry_at>0 AND d.retry_at<=?) AND NOT (d.outcome='blocked' AND d.reason='cooldown')) "
                "AND NOT EXISTS(SELECT 1 FROM reply_attempts a WHERE a.uid=i.uid AND a.source_rpid=i.source_rpid) "
                "ORDER BY i.received_at,i.source_rpid LIMIT 3",
                (uid, since, mode, MAX_FAILURES, now),
            )
            return [IncomingReply(**json.loads(r[0])) for r in rows]

    def decide_reply(
        self, uid: str, source_rpid: int, mode: str, ts: float, outcome: str, text: str = "", reason: str = ""
    ) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT failures FROM reply_decisions WHERE uid=? AND source_rpid=? AND mode=?",
                (uid, source_rpid, mode),
            ).fetchone()
            failures = previous["failures"] if previous else 0
            if outcome in RETRY_OUTCOMES:
                failures += 1
            retry_at = (
                ts + RETRY_INTERVAL * max(1, failures) if outcome in RETRY_OUTCOMES and failures < MAX_FAILURES else 0
            )
            db.execute(
                "INSERT INTO reply_decisions(uid,source_rpid,mode,ts,outcome,text,reason,updated_at,failures,retry_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(uid,source_rpid,mode) "
                "DO UPDATE SET outcome=excluded.outcome,text=excluded.text,reason=excluded.reason,"
                "updated_at=excluded.updated_at,failures=excluded.failures,retry_at=excluded.retry_at",
                (uid, source_rpid, mode, ts, outcome, text, reason[:500], ts, failures, retry_at),
            )
            if outcome == "reading" or previous is None:
                db.execute(
                    "INSERT INTO analyses(uid,kind,item_id,mode,ts) VALUES (?,'reply',?,?,?)",
                    (uid, str(source_rpid), mode, ts),
                )

    def reply_thread_count(self, uid: str, root_rpid: int) -> int:
        with self.connect() as db:
            return db.execute(
                "SELECT count(*) FROM reply_attempts WHERE uid=? AND root_rpid=?", (uid, root_rpid)
            ).fetchone()[0]

    def conversation(self, uid: str, root_rpid: int) -> List[Dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT i.payload,d.text,d.outcome FROM reply_inbox i JOIN reply_decisions d "
                "ON d.uid=i.uid AND d.source_rpid=i.source_rpid AND d.mode='publish' "
                "WHERE i.uid=? AND i.root_rpid=? ORDER BY i.received_at DESC LIMIT 6",
                (uid, root_rpid),
            ).fetchall()
            return [
                {
                    "incoming": json.loads(r["payload"])["text"][:2000],
                    "my_reply": r["text"] if r["outcome"] == "sent" else "",
                    "outcome": r["outcome"],
                }
                for r in reversed(rows)
            ]

    def recent(self, uid: str) -> List[str]:
        with self.connect() as db:
            return [
                row[0] for row in db.execute("SELECT text FROM attempts WHERE uid=? ORDER BY ts DESC LIMIT 30", (uid,))
            ]

    def browse_context(self, uid: str) -> List[Dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT bvid,mode,updated_at AS ts,outcome,detail FROM visits WHERE uid=? ORDER BY updated_at DESC LIMIT 6",
                (uid,),
            )
            result = []
            for row in rows:
                try:
                    detail = json.loads(row["detail"])
                except (ValueError, TypeError):
                    detail = {}
                if not isinstance(detail, dict):
                    detail = {}
                result.append(
                    {
                        "bvid": row["bvid"],
                        "mode": row["mode"],
                        "timestamp": row["ts"],
                        "outcome": row["outcome"],
                        "title": str(detail.get("title", ""))[:300],
                        "feedback": str(detail.get("reason", row["outcome"]))[:200],
                    }
                )
            return result

    def record_browse_decision(
        self,
        uid: str,
        ts: float,
        should_browse: bool,
        reason: str,
        next_check_at: float,
        selected_bvids: Optional[List[str]] = None,
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO browse_decisions(uid,ts,should_browse,reason,next_check_at,selected_bvids) VALUES (?,?,?,?,?,?)",
                (uid, ts, int(should_browse), reason[:200], next_check_at, json.dumps(selected_bvids or [])),
            )

    def latest_browse_decision(self, uid: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute(
                "SELECT ts,should_browse,reason,next_check_at,selected_bvids FROM browse_decisions "
                "WHERE uid=? ORDER BY id DESC LIMIT 1",
                (uid,),
            ).fetchone()
            return dict(row) if row else None

    def page_records(self, uid: str, kind: str, search: str, page: int) -> Dict[str, Any]:
        """仅返回当前账号的分页记录；动态 SQL 只使用固定的表、列定义。"""
        sources = {
            "visits": ("visits", "bvid || ' ' || detail"),
            "attempts": ("attempts", "bvid || ' ' || text || ' ' || status"),
            "decisions": ("browse_decisions", "reason"),
            "replies": ("reply_inbox", "bvid || ' ' || payload"),
            "reply_attempts": ("reply_attempts", "bvid || ' ' || text || ' ' || status"),
            "shares": ("video_shares", "bvid || ' ' || title || ' ' || group_id || ' ' || detail || ' ' || status"),
        }
        if kind not in sources or type(page) is not int or page < 1 or not isinstance(search, str) or len(search) > 200:
            raise ValueError("记录类型、搜索内容或页码无效")
        table, searchable = sources[kind]
        where, args = "uid=?", [uid]
        if search.strip():
            pattern = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where += f" AND ({searchable}) LIKE ? ESCAPE '\\'"
            args.append("%" + pattern + "%")
        with self.connect() as db:
            total = db.execute(f"SELECT count(*) FROM {table} WHERE {where}", args).fetchone()[0]
            pages = max(1, (total + 19) // 20)
            page = min(page, pages)
            ordering = "received_at" if kind == "replies" else "updated_at" if kind == "visits" else "ts"
            if kind == "replies":
                ordering = (
                    "coalesce((SELECT d.updated_at FROM reply_decisions d WHERE d.uid=reply_inbox.uid "
                    "AND d.source_rpid=reply_inbox.source_rpid ORDER BY CASE d.mode WHEN 'publish' "
                    "THEN 0 ELSE 1 END LIMIT 1),received_at)"
                )
            rows = db.execute(
                f"SELECT * FROM {table} WHERE {where} ORDER BY {ordering} DESC LIMIT 20 OFFSET ?",
                [*args, (page - 1) * 20],
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                if kind in {"visits", "attempts"}:
                    if kind == "attempts":
                        visit = db.execute(
                            "SELECT * FROM visits WHERE uid=? AND bvid=? "
                            "ORDER BY CASE mode WHEN 'publish' THEN 0 ELSE 1 END,updated_at DESC LIMIT 1",
                            (uid, item["bvid"]),
                        ).fetchone()
                        item["detail"] = visit["detail"] if visit else ""
                        if visit:
                            item.update(first_ts=visit["ts"], updated_at=visit["updated_at"])
                        item["send_ts"] = item["ts"]
                    else:
                        item["first_ts"] = item["ts"]
                        if item["mode"] == "publish":
                            attempt = db.execute(
                                "SELECT ts FROM attempts WHERE uid=? AND bvid=?", (uid, item["bvid"])
                            ).fetchone()
                            if attempt:
                                item["send_ts"] = attempt["ts"]
                if kind in {"replies", "reply_attempts"}:
                    if kind == "reply_attempts":
                        inbox = db.execute(
                            "SELECT payload FROM reply_inbox WHERE uid=? AND source_rpid=?", (uid, item["source_rpid"])
                        ).fetchone()
                        item["payload"] = inbox[0] if inbox else "{}"
                    decision = db.execute(
                        "SELECT mode,outcome,text,reason,ts,updated_at,failures,retry_at "
                        "FROM reply_decisions WHERE uid=? AND source_rpid=? "
                        "ORDER BY CASE mode WHEN 'publish' THEN 0 ELSE 1 END LIMIT 1",
                        (uid, item["source_rpid"]),
                    ).fetchone()
                    item["decision"] = dict(decision) if decision else {}
                    if kind == "replies":
                        attempt = db.execute(
                            "SELECT status,rpid,ts FROM reply_attempts WHERE uid=? AND source_rpid=?",
                            (uid, item["source_rpid"]),
                        ).fetchone()
                        if attempt:
                            item.update(status=attempt["status"], rpid=attempt["rpid"], send_ts=attempt["ts"])
                    else:
                        item["send_ts"] = item["ts"]
                result.append(item)
        return {"rows": result, "pagination": {"page": page, "pages": pages, "total": total}}

    def reserve(
        self, uid: str, bvid: str, text: str, now: float, day_start: float, limit: int, cooldown: int, similarity: float
    ) -> str:
        """在同一事务中验证限额并占用发送机会，跨进程也不会重复发送。"""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT value FROM meta WHERE key='paused'").fetchone()[0] != "0":
                return "paused"
            if db.execute("SELECT 1 FROM attempts WHERE uid=? AND status IN ('pending','unknown')", (uid,)).fetchone():
                return "unresolved"
            if db.execute(
                "SELECT 1 FROM reply_attempts WHERE uid=? AND status IN ('pending','unknown')", (uid,)
            ).fetchone():
                return "unresolved"
            if db.execute("SELECT 1 FROM attempts WHERE uid=? AND bvid=?", (uid, bvid)).fetchone():
                return "duplicate_video"
            rows = db.execute("SELECT ts, text FROM attempts WHERE uid=? ORDER BY ts DESC", (uid,)).fetchall()
            if sum(row["ts"] >= day_start for row in rows) >= limit:
                return "daily_limit"
            all_rows = db.execute(
                "SELECT ts,text FROM attempts WHERE uid=? UNION ALL "
                "SELECT ts,text FROM reply_attempts WHERE uid=? ORDER BY ts DESC LIMIT 30",
                (uid, uid),
            ).fetchall()
            if all_rows and now - all_rows[0]["ts"] < cooldown:
                return "cooldown"
            canonical = normalized(text)
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            if (
                db.execute("SELECT 1 FROM attempts WHERE uid=? AND text_hash=?", (uid, digest)).fetchone()
                or db.execute("SELECT 1 FROM reply_attempts WHERE uid=? AND text_hash=?", (uid, digest)).fetchone()
            ):
                return "duplicate_text"
            if any(SequenceMatcher(None, canonical, normalized(row["text"])).ratio() >= similarity for row in all_rows):
                return "similar_text"
            db.execute(
                "INSERT INTO attempts(uid,bvid,ts,text,text_hash,status) VALUES (?,?,?,?,?,'pending')",
                (uid, bvid, now, text, digest),
            )
        return "reserved"

    def reserve_reply(
        self,
        uid: str,
        message: IncomingReply,
        text: str,
        now: float,
        day_start: float,
        limit: int,
        cooldown: int,
        thread_limit: int,
        similarity: float,
    ) -> str:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT value FROM meta WHERE key='paused'").fetchone()[0] != "0":
                return "paused"
            for table in ("attempts", "reply_attempts"):
                if db.execute(
                    f"SELECT 1 FROM {table} WHERE uid=? AND status IN ('pending','unknown')", (uid,)
                ).fetchone():
                    return "unresolved"
            if db.execute(
                "SELECT 1 FROM reply_attempts WHERE uid=? AND source_rpid=?", (uid, message.source_rpid)
            ).fetchone():
                return "duplicate_reply"
            if not db.execute(
                "SELECT 1 FROM attempts WHERE uid=? AND bvid=? AND rpid=? AND status='sent'",
                (uid, message.bvid, str(message.root_rpid)),
            ).fetchone():
                return "untracked_thread"
            if (
                message.target_rpid != message.root_rpid
                and not db.execute(
                    "SELECT 1 FROM reply_attempts WHERE uid=? AND bvid=? AND root_rpid=? AND rpid=? AND status='sent'",
                    (uid, message.bvid, message.root_rpid, str(message.target_rpid)),
                ).fetchone()
            ):
                return "untracked_target"
            if (
                db.execute("SELECT count(*) FROM reply_attempts WHERE uid=? AND ts>=?", (uid, day_start)).fetchone()[0]
                >= limit
            ):
                return "daily_limit"
            if (
                db.execute(
                    "SELECT count(*) FROM reply_attempts WHERE uid=? AND root_rpid=?", (uid, message.root_rpid)
                ).fetchone()[0]
                >= thread_limit
            ):
                return "thread_limit"
            recent = db.execute(
                "SELECT ts,text FROM attempts WHERE uid=? UNION ALL SELECT ts,text FROM reply_attempts "
                "WHERE uid=? ORDER BY ts DESC LIMIT 30",
                (uid, uid),
            ).fetchall()
            if recent and now - recent[0]["ts"] < cooldown:
                return "cooldown"
            canonical = normalized(text)
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            for table in ("attempts", "reply_attempts"):
                if db.execute(f"SELECT 1 FROM {table} WHERE uid=? AND text_hash=?", (uid, digest)).fetchone():
                    return "duplicate_text"
            if any(SequenceMatcher(None, canonical, normalized(r["text"])).ratio() >= similarity for r in recent):
                return "similar_text"
            db.execute(
                "INSERT INTO reply_attempts VALUES (?,?,?,?,?,?,?,'pending','')",
                (uid, message.source_rpid, message.bvid, message.root_rpid, now, text, digest),
            )
        return "reserved"

    def finish_reply(self, uid: str, source_rpid: int, status: str, rpid: str = "") -> None:
        if status not in {"sent", "rejected", "unknown", "cancelled"}:
            raise ValueError("invalid reply attempt status")
        with self.connect() as db:
            db.execute(
                "UPDATE reply_attempts SET status=?,rpid=? WHERE uid=? AND source_rpid=? AND status='pending'",
                (status, rpid, uid, source_rpid),
            )

    def finish(self, uid: str, bvid: str, status: str, rpid: str = "") -> None:
        if status not in {"sent", "rejected", "unknown", "cancelled"}:
            raise ValueError("invalid attempt status")
        with self.connect() as db:
            db.execute(
                "UPDATE attempts SET status=?,rpid=? WHERE uid=? AND bvid=? AND status='pending'",
                (status, rpid, uid, bvid),
            )

    def acknowledge(self, uid: str, bvid: str) -> bool:
        with self.connect() as db:
            result = db.execute(
                "UPDATE attempts SET status='acknowledged' WHERE uid=? AND bvid=? AND status='unknown'", (uid, bvid)
            )
            replies = db.execute(
                "UPDATE reply_attempts SET status='acknowledged' WHERE uid=? AND bvid=? AND status='unknown'",
                (uid, bvid),
            )
            return result.rowcount + replies.rowcount > 0

    def history(self, uid: str) -> str:
        with self.connect() as db:
            rows = db.execute(
                "SELECT bvid,mode,outcome,detail FROM visits WHERE uid=? ORDER BY updated_at DESC LIMIT 8", (uid,)
            )
            visits = (
                "\n".join(f"{r['bvid']} [{r['mode']}/{r['outcome']}] {r['detail'][:200]}" for r in rows)
                or "暂无浏览记录"
            )
            choices = db.execute(
                "SELECT should_browse,reason FROM browse_decisions WHERE uid=? ORDER BY id DESC LIMIT 3", (uid,)
            )
            decisions = "\n".join(
                f"自主决定：{'去浏览' if r['should_browse'] else '暂时不刷'}；{r['reason']}" for r in choices
            )
            reply_rows = db.execute(
                "SELECT i.bvid,d.outcome,d.text,d.reason FROM reply_inbox i JOIN reply_decisions d "
                "ON d.uid=i.uid AND d.source_rpid=i.source_rpid WHERE i.uid=? ORDER BY d.updated_at DESC LIMIT 3",
                (uid,),
            )
            replies = "\n".join(f"{r['bvid']} [跟进/{r['outcome']}] {r['text'] or r['reason']}" for r in reply_rows)
            return "\n".join(part for part in (replies, decisions, visits) if part)
