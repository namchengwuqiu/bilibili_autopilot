"""每次 SQLite 操作有独立连接。异步调用者应使用 asyncio.to_thread。"""

from contextlib import contextmanager
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import hashlib
import json
import sqlite3

from .models import IncomingReply, normalized


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

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
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
            """)
            # 请求发出前已落盘；进程中断后无法知道平台是否接收，绝不重试该视频。
            db.execute("UPDATE attempts SET status='unknown' WHERE status='pending'")
            db.execute("UPDATE reply_attempts SET status='unknown' WHERE status='pending'")
            # 旧版把未发送的模型超时记录为终态失败；允许后续检查重新判断。
            db.execute("UPDATE reply_decisions SET outcome='model_timeout' "
                       "WHERE outcome='model_error' AND reason='模型调用失败：TimeoutError'")
            db.execute("UPDATE video_shares SET status='unknown',detail='进程中断，群分享结果待核对；不会自动重发' WHERE status='pending'")
        self.path.chmod(0o600)

    def set_pause(self, paused: bool, reason: str, kind: str = "error") -> None:
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                           [("paused", "1" if paused else "0"), ("pause_reason", reason[:500]),
                            ("pause_kind", kind if paused else "running")])

    def reserve_share(self, uid: str, bvid: str, group_id: str, ts: float, title: str) -> bool:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM attempts WHERE uid=? AND bvid=? AND status='sent' AND rpid!=''", (uid,bvid)).fetchone():
                return False
            cursor = db.execute("INSERT OR IGNORE INTO video_shares(uid,bvid,group_id,ts,title,status) VALUES (?,?,?,?,?,'pending')",
                                (uid,bvid,group_id,ts,title[:300]))
            return cursor.rowcount == 1

    def finish_share(self, uid: str, bvid: str, group_id: str, status: str,
                     detail: str, message_id: str = "") -> None:
        with self.connect() as db:
            db.execute("UPDATE video_shares SET status=?,detail=?,message_id=? WHERE uid=? AND bvid=? AND group_id=?",
                       (status,detail[:500],message_id,uid,bvid,group_id))

    def latest_share(self, uid: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM video_shares WHERE uid=? ORDER BY ts DESC LIMIT 1", (uid,)).fetchone()
            return dict(row) if row else None

    def share_context(self, uid: str, bvid: str) -> Dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT a.text,v.detail FROM attempts a LEFT JOIN visits v "
                             "ON v.uid=a.uid AND v.bvid=a.bvid AND v.mode='publish' "
                             "WHERE a.uid=? AND a.bvid=? AND a.status='sent' AND a.rpid!=''", (uid,bvid)).fetchone()
            if row is None:
                raise ValueError("没有已提交成功的评论，不能生成群分享")
            detail = json.loads(row['detail']) if row['detail'] else {}
            recent = db.execute("SELECT detail FROM video_shares WHERE uid=? AND status='sent' "
                                "ORDER BY ts DESC LIMIT 20", (uid,)).fetchall()
            recent_texts = list(dict.fromkeys(r['detail'].removeprefix('分享文案：') for r in recent
                                             if r['detail'].startswith('分享文案：')))[:4]
            return {"my_comment": row['text'], "reason": detail.get('reason', ''),
                    "content_evidence": detail.get('evidence_quote', ''), "recent_shares": recent_texts}

    def control(self) -> Dict[str, str]:
        with self.connect() as db:
            return {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM meta")}

    def visit(self, uid: str, bvid: str, mode: str, ts: float, outcome: str, detail: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?) "
                       "ON CONFLICT(uid,bvid,mode) DO UPDATE SET outcome=excluded.outcome, detail=excluded.detail",
                       (uid, bvid, mode, ts, outcome, detail[:1500]))

    def seen(self, uid: str, bvid: str, mode: str) -> bool:
        with self.connect() as db:
            return bool(db.execute("SELECT 1 FROM visits WHERE uid=? AND bvid=? AND mode=?",
                                   (uid, bvid, mode)).fetchone() or
                        db.execute("SELECT 1 FROM attempts WHERE uid=? AND bvid=?", (uid, bvid)).fetchone())

    def stats(self, uid: str, day_start: float) -> Dict[str, Any]:
        with self.connect() as db:
            views = db.execute("SELECT count(*) FROM visits WHERE uid=? AND ts>=?", (uid, day_start)).fetchone()[0]
            attempts = db.execute("SELECT count(*) FROM attempts WHERE uid=? AND ts>=?", (uid, day_start)).fetchone()[0]
            unresolved = [r[0] for r in db.execute("SELECT bvid FROM attempts WHERE uid=? AND status IN ('pending','unknown')", (uid,))]
            last = db.execute("SELECT coalesce(max(ts),0) FROM attempts WHERE uid=?", (uid,)).fetchone()[0]
            unresolved += [r[0] for r in db.execute("SELECT bvid FROM reply_attempts WHERE uid=? AND status IN ('pending','unknown')", (uid,))]
            reply_attempts = db.execute("SELECT count(*) FROM reply_attempts WHERE uid=? AND ts>=?", (uid,day_start)).fetchone()[0]
            reply_analyses = db.execute("SELECT count(*) FROM reply_decisions WHERE uid=? AND ts>=?", (uid,day_start)).fetchone()[0]
            reply_last = db.execute("SELECT coalesce(max(ts),0) FROM reply_attempts WHERE uid=?", (uid,)).fetchone()[0]
            return {"views": views, "attempts": attempts, "unresolved": sorted(set(unresolved)), "last_attempt": max(last, reply_last),
                    "reply_attempts": reply_attempts, "reply_analyses": reply_analyses}

    def own_comments(self, uid: str) -> Dict[str, Dict[str, Any]]:
        with self.connect() as db:
            roots = db.execute("SELECT bvid,rpid,text FROM attempts WHERE uid=? AND status='sent' AND rpid!=''", (uid,)).fetchall()
            own = {r['rpid']: {"bvid": r['bvid'], "root_rpid": int(r['rpid']), "root_text": r['text'], "text": r['text']}
                   for r in roots if r['rpid'].isdigit() and int(r['rpid']) > 0}
            replies = db.execute("SELECT bvid,root_rpid,rpid,text FROM reply_attempts WHERE uid=? AND status='sent' AND rpid!=''", (uid,))
            for row in replies:
                root = own.get(str(row['root_rpid']))
                if root and row['rpid'].isdigit() and int(row['rpid']) > 0:
                    own[row['rpid']] = {"bvid": row['bvid'], "root_rpid": row['root_rpid'],
                                        "root_text": root['root_text'], "text": row['text']}
            return own

    def last_reply_check(self, uid: str) -> float:
        with self.connect() as db:
            row = db.execute("SELECT last_checked FROM reply_poll WHERE uid=?", (uid,)).fetchone()
            return row[0] if row else 0

    def set_reply_check(self, uid: str, now: float) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO reply_poll VALUES (?,?)", (uid,now))

    def receive_replies(self, uid: str, messages: List[IncomingReply]) -> None:
        with self.connect() as db:
            db.executemany("INSERT OR IGNORE INTO reply_inbox VALUES (?,?,?,?,?,?)",
                [(uid,m.source_rpid,m.bvid,m.root_rpid,m.received_at,json.dumps(asdict(m),ensure_ascii=False)) for m in messages])

    def pending_replies(self, uid: str, mode: str, since: float) -> List[IncomingReply]:
        with self.connect() as db:
            rows = db.execute("SELECT i.payload FROM reply_inbox i WHERE i.uid=? AND i.received_at>=? "
                "AND NOT EXISTS(SELECT 1 FROM reply_decisions d WHERE d.uid=i.uid AND d.source_rpid=i.source_rpid AND d.mode=? "
                "AND d.outcome!='model_timeout') "
                "AND NOT EXISTS(SELECT 1 FROM reply_attempts a WHERE a.uid=i.uid AND a.source_rpid=i.source_rpid) "
                "ORDER BY i.received_at,i.source_rpid LIMIT 3", (uid,since,mode))
            return [IncomingReply(**json.loads(r[0])) for r in rows]

    def decide_reply(self, uid: str, source_rpid: int, mode: str, ts: float, outcome: str,
                     text: str = "", reason: str = "") -> None:
        with self.connect() as db:
            db.execute("INSERT INTO reply_decisions VALUES (?,?,?,?,?,?,?) ON CONFLICT(uid,source_rpid,mode) "
                "DO UPDATE SET outcome=excluded.outcome,text=excluded.text,reason=excluded.reason",
                (uid,source_rpid,mode,ts,outcome,text,reason[:500]))

    def reply_thread_count(self, uid: str, root_rpid: int) -> int:
        with self.connect() as db:
            return db.execute("SELECT count(*) FROM reply_attempts WHERE uid=? AND root_rpid=?", (uid,root_rpid)).fetchone()[0]

    def conversation(self, uid: str, root_rpid: int) -> List[Dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT i.payload,d.text,d.outcome FROM reply_inbox i JOIN reply_decisions d "
                "ON d.uid=i.uid AND d.source_rpid=i.source_rpid AND d.mode='publish' "
                "WHERE i.uid=? AND i.root_rpid=? ORDER BY i.received_at DESC LIMIT 6", (uid,root_rpid)).fetchall()
            return [{"incoming":json.loads(r['payload'])['text'][:2000],
                     "my_reply":r['text'] if r['outcome']=='sent' else '', "outcome":r['outcome']} for r in reversed(rows)]

    def recent(self, uid: str) -> List[str]:
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT text FROM attempts WHERE uid=? ORDER BY ts DESC LIMIT 30", (uid,))]

    def browse_context(self, uid: str) -> List[Dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT bvid,mode,ts,outcome,detail FROM visits WHERE uid=? ORDER BY ts DESC LIMIT 6", (uid,))
            result = []
            for row in rows:
                try:
                    detail = json.loads(row["detail"])
                except (ValueError, TypeError):
                    detail = {}
                if not isinstance(detail, dict):
                    detail = {}
                result.append({"bvid": row["bvid"], "mode": row["mode"], "timestamp": row["ts"], "outcome": row["outcome"],
                               "title": str(detail.get("title", ""))[:300],
                               "feedback": str(detail.get("reason", row["outcome"]))[:200]})
            return result

    def record_browse_decision(self, uid: str, ts: float, should_browse: bool,
                               reason: str, next_check_at: float) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO browse_decisions(uid,ts,should_browse,reason,next_check_at) VALUES (?,?,?,?,?)",
                       (uid, ts, int(should_browse), reason[:200], next_check_at))

    def latest_browse_decision(self, uid: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT ts,should_browse,reason,next_check_at FROM browse_decisions "
                             "WHERE uid=? ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
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
            ordering = "received_at" if kind == "replies" else "ts"
            rows = db.execute(f"SELECT * FROM {table} WHERE {where} ORDER BY {ordering} DESC LIMIT 20 OFFSET ?",
                              [*args, (page - 1) * 20]).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                if kind == "attempts":
                    visit = db.execute("SELECT detail FROM visits WHERE uid=? AND bvid=? ORDER BY ts DESC LIMIT 1",
                                       (uid, item["bvid"])).fetchone()
                    item["detail"] = visit["detail"] if visit else ""
                if kind in {"replies", "reply_attempts"}:
                    if kind == "reply_attempts":
                        inbox = db.execute("SELECT payload FROM reply_inbox WHERE uid=? AND source_rpid=?", (uid,item['source_rpid'])).fetchone()
                        item['payload'] = inbox[0] if inbox else '{}'
                    decision = db.execute("SELECT mode,outcome,text,reason FROM reply_decisions WHERE uid=? AND source_rpid=? "
                        "ORDER BY CASE mode WHEN 'publish' THEN 0 ELSE 1 END LIMIT 1", (uid,item['source_rpid'])).fetchone()
                    item['decision'] = dict(decision) if decision else {}
                    if kind == 'replies':
                        attempt = db.execute("SELECT status,rpid FROM reply_attempts WHERE uid=? AND source_rpid=?", (uid,item['source_rpid'])).fetchone()
                        if attempt:
                            item.update(status=attempt['status'],rpid=attempt['rpid'])
                result.append(item)
        return {"rows": result, "pagination": {"page": page, "pages": pages, "total": total}}

    def reserve(self, uid: str, bvid: str, text: str, now: float, day_start: float,
                limit: int, cooldown: int, similarity: float) -> str:
        """在同一事务中验证限额并占用发送机会，跨进程也不会重复发送。"""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT value FROM meta WHERE key='paused'").fetchone()[0] != "0":
                return "paused"
            if db.execute("SELECT 1 FROM attempts WHERE uid=? AND status IN ('pending','unknown')", (uid,)).fetchone():
                return "unresolved"
            if db.execute("SELECT 1 FROM reply_attempts WHERE uid=? AND status IN ('pending','unknown')", (uid,)).fetchone():
                return "unresolved"
            if db.execute("SELECT 1 FROM attempts WHERE uid=? AND bvid=?", (uid, bvid)).fetchone():
                return "duplicate_video"
            rows = db.execute("SELECT ts, text FROM attempts WHERE uid=? ORDER BY ts DESC", (uid,)).fetchall()
            if sum(row["ts"] >= day_start for row in rows) >= limit:
                return "daily_limit"
            all_rows = db.execute("SELECT ts,text FROM attempts WHERE uid=? UNION ALL "
                                  "SELECT ts,text FROM reply_attempts WHERE uid=? ORDER BY ts DESC LIMIT 30", (uid,uid)).fetchall()
            if all_rows and now - all_rows[0]["ts"] < cooldown:
                return "cooldown"
            canonical = normalized(text)
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            if (db.execute("SELECT 1 FROM attempts WHERE uid=? AND text_hash=?", (uid, digest)).fetchone()
                    or db.execute("SELECT 1 FROM reply_attempts WHERE uid=? AND text_hash=?", (uid,digest)).fetchone()):
                return "duplicate_text"
            if any(SequenceMatcher(None, canonical, normalized(row["text"])).ratio() >= similarity for row in all_rows):
                return "similar_text"
            db.execute("INSERT INTO attempts(uid,bvid,ts,text,text_hash,status) VALUES (?,?,?,?,?,'pending')",
                       (uid, bvid, now, text, digest))
        return "reserved"

    def reserve_reply(self, uid: str, message: IncomingReply, text: str, now: float, day_start: float,
                      limit: int, cooldown: int, thread_limit: int, similarity: float) -> str:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT value FROM meta WHERE key='paused'").fetchone()[0] != '0':
                return 'paused'
            for table in ('attempts','reply_attempts'):
                if db.execute(f"SELECT 1 FROM {table} WHERE uid=? AND status IN ('pending','unknown')", (uid,)).fetchone():
                    return 'unresolved'
            if db.execute("SELECT 1 FROM reply_attempts WHERE uid=? AND source_rpid=?", (uid,message.source_rpid)).fetchone():
                return 'duplicate_reply'
            if not db.execute("SELECT 1 FROM attempts WHERE uid=? AND bvid=? AND rpid=? AND status='sent'",
                              (uid,message.bvid,str(message.root_rpid))).fetchone():
                return 'untracked_thread'
            if (message.target_rpid != message.root_rpid and not db.execute(
                    "SELECT 1 FROM reply_attempts WHERE uid=? AND bvid=? AND root_rpid=? AND rpid=? AND status='sent'",
                    (uid,message.bvid,message.root_rpid,str(message.target_rpid))).fetchone()):
                return 'untracked_target'
            if db.execute("SELECT count(*) FROM reply_attempts WHERE uid=? AND ts>=?", (uid,day_start)).fetchone()[0] >= limit:
                return 'daily_limit'
            if db.execute("SELECT count(*) FROM reply_attempts WHERE uid=? AND root_rpid=?", (uid,message.root_rpid)).fetchone()[0] >= thread_limit:
                return 'thread_limit'
            recent = db.execute("SELECT ts,text FROM attempts WHERE uid=? UNION ALL SELECT ts,text FROM reply_attempts "
                                "WHERE uid=? ORDER BY ts DESC LIMIT 30", (uid,uid)).fetchall()
            if recent and now - recent[0]['ts'] < cooldown:
                return 'cooldown'
            canonical = normalized(text)
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            for table in ('attempts','reply_attempts'):
                if db.execute(f"SELECT 1 FROM {table} WHERE uid=? AND text_hash=?", (uid,digest)).fetchone():
                    return 'duplicate_text'
            if any(SequenceMatcher(None,canonical,normalized(r['text'])).ratio() >= similarity for r in recent):
                return 'similar_text'
            db.execute("INSERT INTO reply_attempts VALUES (?,?,?,?,?,?,?,'pending','')",
                       (uid,message.source_rpid,message.bvid,message.root_rpid,now,text,digest))
        return 'reserved'

    def finish_reply(self, uid: str, source_rpid: int, status: str, rpid: str = '') -> None:
        if status not in {'sent','rejected','unknown','cancelled'}:
            raise ValueError('invalid reply attempt status')
        with self.connect() as db:
            db.execute("UPDATE reply_attempts SET status=?,rpid=? WHERE uid=? AND source_rpid=? AND status='pending'",
                       (status,rpid,uid,source_rpid))

    def finish(self, uid: str, bvid: str, status: str, rpid: str = "") -> None:
        if status not in {"sent", "rejected", "unknown", "cancelled"}:
            raise ValueError("invalid attempt status")
        with self.connect() as db:
            db.execute("UPDATE attempts SET status=?,rpid=? WHERE uid=? AND bvid=? AND status='pending'",
                       (status, rpid, uid, bvid))

    def acknowledge(self, uid: str, bvid: str) -> bool:
        with self.connect() as db:
            result = db.execute("UPDATE attempts SET status='acknowledged' WHERE uid=? AND bvid=? AND status='unknown'",
                                (uid, bvid))
            replies = db.execute("UPDATE reply_attempts SET status='acknowledged' WHERE uid=? AND bvid=? AND status='unknown'", (uid,bvid))
            return result.rowcount + replies.rowcount > 0

    def history(self, uid: str) -> str:
        with self.connect() as db:
            rows = db.execute("SELECT bvid,mode,outcome,detail FROM visits WHERE uid=? ORDER BY ts DESC LIMIT 8", (uid,))
            visits = "\n".join(f"{r['bvid']} [{r['mode']}/{r['outcome']}] {r['detail'][:200]}" for r in rows) or "暂无浏览记录"
            choices = db.execute("SELECT should_browse,reason FROM browse_decisions WHERE uid=? ORDER BY id DESC LIMIT 3", (uid,))
            decisions = "\n".join(f"自主决定：{'去浏览' if r['should_browse'] else '暂时不刷'}；{r['reason']}" for r in choices)
            reply_rows = db.execute("SELECT i.bvid,d.outcome,d.text,d.reason FROM reply_inbox i JOIN reply_decisions d "
                "ON d.uid=i.uid AND d.source_rpid=i.source_rpid WHERE i.uid=? ORDER BY d.ts DESC LIMIT 3", (uid,))
            replies = '\n'.join(f"{r['bvid']} [跟进/{r['outcome']}] {r['text'] or r['reason']}" for r in reply_rows)
            return '\n'.join(part for part in (replies,decisions,visits) if part)
