import streamlit as st
import time
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
import google.generativeai as genai
import logging
import traceback
from io import StringIO
import sys
import os

# ========================
# LOGGING SETUP
# ========================
def setup_logging():
    logger = logging.getLogger('josaa_rag')
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fmt = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    string_handler = logging.StreamHandler(st.session_state.get('log_stream', StringIO()))
    string_handler.setLevel(logging.DEBUG)
    string_handler.setFormatter(fmt)
    logger.addHandler(string_handler)
    return logger

if 'log_stream' not in st.session_state:
    st.session_state.log_stream = StringIO()

logger = setup_logging()

def log_performance(name, start):
    dur = time.time() - start
    logger.info(f"⏱️  {name} completed in {dur:.2f}s")
    return dur

def log_error(op, e):
    logger.error(f"❌ {op} failed: {e}")
    logger.debug(f"Traceback:\n{traceback.format_exc()}")

# ========================
# CREDENTIALS (ENV-FIRST)
# ========================
DB = {
    "host": os.getenv("DB_HOST", "DB_HOST_HERE"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "DB_USER_HERE"),
    "password": os.getenv("DB_PASSWORD", "DB_PASSWORD_HERE"),
    "port": os.getenv("DB_PORT", "6543"),
    "sslmode": os.getenv("DB_SSLMODE", "require"),
    "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY_HERE")

logger.info("🔧 Configuration loaded")

# ========================
# POLICY
# ========================
@dataclass
class Policy:
    exclude_pwd_default: bool = True
    final_round_default: bool = True
    apply_open_gn_ai_on_air: bool = True
    apply_open_gn_ai_on_numeric_eligibility: bool = True

# ========================
# CONSTANTS & HELPERS
# ========================
RANK_NUM_EXPR = (
    "CAST(NULLIF(regexp_replace(trim(closing_rank), '[^0-9]', '', 'g'), '') AS INTEGER)"
)
ALLOWED_READ = re.compile(r"^\s*(select|with)\b", re.IGNORECASE | re.DOTALL)
DANGEROUS = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|vacuum|analyze)\b",
    re.IGNORECASE,
)

def single_statement(sql: str) -> bool:
    return sql.strip().count(";") <= 1

def ensure_limit(sql: str, hard_limit: int = 15) -> str:
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip(';')} LIMIT {hard_limit};"

def sanitize_select(sql: str, hard_limit: int = 15) -> str:
    sql = sql.strip()
    if not single_statement(sql):
        raise ValueError("Multiple SQL statements not allowed.")
    if not ALLOWED_READ.match(sql):
        raise ValueError("Only SELECT/CTE read queries are allowed.")
    if DANGEROUS.search(sql):
        raise ValueError("Potentially dangerous SQL token detected.")
    return ensure_limit(sql, hard_limit)

def air_context_hint(text: str) -> str:
    return ("User intent hint: Eligibility by AIR is requested; apply OPEN/GN/AI, "
            "exclude PwD, and latest round unless the user overrides."
            ) if re.search(r'\b(AIR|rank)\b', text, re.I) else ""

def numeric_eligibility_hint(text: str) -> bool:
    return bool(
        re.search(r'\b(under|below|less\s+than|greater\s+than|over|at\s+least|at\s+most)\s*\d+', text, re.I)
        or re.search(r'\b(<=|>=|<|>)\s*\d+', text)
        or re.search(r'\bclosing\s*rank\b.*\d+', text, re.I)
    )

def fix_distinct_orderby(sql: str) -> str:
    if re.search(r"\bselect\s+distinct\b", sql, re.IGNORECASE) and \
       re.search(r"\border\s+by\s+.*closing_rank_num", sql, re.IGNORECASE):
        m = re.search(r"(?is)select\s+distinct\s+(.*?)\s+from\s", sql)
        if m and "closing_rank_num" not in m.group(1):
            start, end = m.span(1)
            sql = sql[:start] + m.group(1).rstrip() + ", closing_rank_num" + sql[end:]
    return sql

def _rank_to_num(v: str) -> int:
    s = re.sub(r'[^0-9]', '', str(v or ''))
    return int(s) if s else 0

def _round_to_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return -1

def _norm(x): return (x or '').strip().lower()

def _key_tuple(r: Dict) -> tuple:
    return (_norm(r.get("institute")), _norm(r.get("program")),
            _norm(r.get("category")), _norm(r.get("quota")), _norm(r.get("gender")))

def _sort_tuple(r: Dict) -> tuple:
    return (_rank_to_num(r.get("closing_rank")),
            -_round_to_int(r.get("round")),
            -(int(r.get("year")) if str(r.get("year") or '').isdigit() else -1))

# ========================
# DB LAYER
# ========================
class Pg:
    def __init__(self, params: Dict[str, str]):
        logger.info("🗃️  Initializing database connection pool")
        start = time.time()
        try:
            self.pool = SimpleConnectionPool(minconn=1, maxconn=6, **params)
            log_performance("Database pool creation", start)
        except Exception as e:
            log_error("Database pool creation", e); raise

    def _conn(self):
        start = time.time()
        try:
            conn = self.pool.getconn()
            with conn.cursor() as c:
                c.execute("SET LOCAL statement_timeout = '12000ms';")
                c.execute("SET LOCAL default_transaction_read_only = on;")
            log_performance("Database connection setup", start)
            return conn
        except Exception as e:
            log_error("Database connection", e); raise

    def fetch_schema_text(self) -> str:
        logger.info("📊 Fetching live database schema")
        start = time.time()
        conn = self._conn()
        try:
            cur = conn.cursor()
            out = ["SCHEMA: public\n"]
            cur.execute("""SELECT table_name FROM information_schema.tables
                           WHERE table_schema='public' ORDER BY table_name;""")
            tables = cur.fetchall()
            for (t,) in tables:
                out.append(f"TABLE: {t}")
                cur.execute("""SELECT column_name, data_type, is_nullable
                               FROM information_schema.columns
                               WHERE table_schema='public' AND table_name=%s
                               ORDER BY ordinal_position;""", (t,))
                for name, dtype, nullable in cur.fetchall():
                    out.append(f"  - {name}: {dtype} {'NULL' if nullable=='YES' else 'NOT NULL'}")
                out.append("")
            schema = "\n".join(out)
            log_performance("Schema fetch", start)
            logger.info(f"✅ Schema fetched successfully ({len(schema)} chars)")
            return schema
        except Exception as e:
            log_error("Schema fetch", e); raise
        finally:
            self.pool.putconn(conn)

    def run(self, sql: str) -> List[Dict]:
        logger.info("🏃‍♀️ Executing SQL query")
        start = time.time()
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchall()]
            log_performance("SQL execution", start)
            logger.info(f"✅ Query executed: {len(rows)} rows")
            return rows
        except Exception as e:
            log_error("SQL execution", e); raise
        finally:
            self.pool.putconn(conn)

# ========================
# STAGE A: QUERY ENHANCER
# ========================
class QueryEnhancer:
    def __init__(self, api_key: str, policy: Policy):
        logger.info("🔍 Initializing Query Enhancer (Gemini Flash)")
        start = time.time()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.policy = policy
        log_performance("Query Enhancer initialization", start)

    def enhance(self, user_query: str) -> List[str]:
        logger.info(f"🔍 Enhancing user query: '{user_query}'")
        start = time.time()
        hints = []
        if self.policy.exclude_pwd_default:
            hints.append("exclude PwD by default unless the user explicitly asks for PwD")
        if self.policy.final_round_default:
            hints.append("assume final (latest) JoSAA round when round is unspecified")
        if self.policy.apply_open_gn_ai_on_numeric_eligibility:
            hints.append("for numeric cutoff queries assume OPEN, Gender-Neutral, quota AI unless specified")
        generic = "; ".join(hints) if hints else "no additional defaults"
        air_hint = ("If the user mentions AIR but not category/gender/quota/round, append "
                    "(assume OPEN, Gender-Neutral, quota AI, exclude PwD, final round)."
                   ) if self.policy.apply_open_gn_ai_on_air else ""
        prompt = f"""
Rewrite the user's database question into 2–3 concise variants.
- Keep the same intent, one line each.
- Do NOT invent constraints.
- When unspecified, {generic}.
- {air_hint}

User query: "{user_query}"
Variants:
"""
        try:
            res = self.model.generate_content(prompt)
            text = (res.text or "").strip()
            variants = [l.strip() for l in text.splitlines() if l.strip()]
            variants = variants[:3] if variants else [user_query]
            log_performance("Query enhancement", start)
            return variants
        except Exception as e:
            log_error("Query enhancement", e)
            return [user_query]

# ========================
# STAGE B: SQL GEN & CRITIQUE
# ========================
class SqlGenPro:
    def __init__(self, api_key: str, schema_text: str, policy: Policy):
        logger.info("🧠 Initializing SQL Generator (Gemini 2.5 Pro)")
        start = time.time()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")
        self.schema_text = schema_text
        self.policy = policy
        log_performance("SQL Generator initialization", start)

    def _rules(self) -> str:
        base = f"""
DATA RULES:
- opening_rank and closing_rank are TEXT. Use:
  {RANK_NUM_EXPR} AS closing_rank_num
- Use ILIKE for case-insensitive matches.
- Prefer table 'josaa_btech_2024' for B.Tech queries.
- IIT filter: institute ILIKE '%Indian Institute of Technology%' OR institute ILIKE 'IIT %'
- Include year and round in SELECT when present.
- If you use closing_rank_num in ORDER BY/filters, include it in SELECT.
- ORDER BY closing_rank_num ASC NULLS LAST.
- ALWAYS LIMIT 15.
"""
        extra = []
        if self.policy.exclude_pwd_default:
            extra.append("If user didn't mention PwD, add: category NOT ILIKE '%PwD%'.")
        if self.policy.final_round_default:
            extra.append("If round unspecified, add: round = (SELECT MAX(round) FROM josaa_btech_2024 WHERE year=2024).")
        if self.policy.apply_open_gn_ai_on_air:
            extra.append("If asking by AIR and unspecified, add quota='AI', category='OPEN', category NOT ILIKE '%PwD%', gender='Gender-Neutral', round=max for 2024; filter closing_rank_num >= AIR.")
        if self.policy.apply_open_gn_ai_on_numeric_eligibility:
            extra.append("If numeric cutoff but no category/gender/quota, add quota='AI', category='OPEN', category NOT ILIKE '%PwD%', gender='Gender-Neutral'.")
        if extra:
            base += "- " + "\n- ".join(extra) + "\n"
        return base

    def to_sql(self, nl_query: str) -> Optional[str]:
        logger.info(f"🛠️  Generating SQL for: '{nl_query}'")
        start = time.time()
        rules = self._rules()
        ctx = []
        if air_context_hint(nl_query):
            ctx.append(air_context_hint(nl_query))
        if self.policy.apply_open_gn_ai_on_numeric_eligibility and numeric_eligibility_hint(nl_query):
            ctx.append("Numeric cutoff intent detected; apply OPEN/GN/AI, exclude PwD, latest round.")
        ctx = "\n".join(ctx)

        prompt = f"""
You are an expert PostgreSQL writer.

DATABASE SCHEMA (public):
{self.schema_text}

{rules}

{ctx}

Return ONLY a single SQL SELECT (WITH allowed).

User query: "{nl_query}"
SQL:
"""
        try:
            res = self.model.generate_content(prompt)
            sql = (res.text or "").strip()
            if sql.startswith("```"):
                sql = sql.replace("```sql", "").replace("```", "").strip()
            if not sql:
                return None
            sql = fix_distinct_orderby(sql)
            # Critique/fix
            fix = self.critique_fix(sql, nl_query, rules)
            sql_out = fix or sql
            log_performance("SQL generation", start)
            return sql_out
        except Exception as e:
            log_error("SQL generation", e)
            return None

    def critique_fix(self, sql: str, nl_query: str, rules: str) -> Optional[str]:
        start = time.time()
        critic = f"""
You are a PostgreSQL critic. Ensure:
- closing_rank_num derived & in SELECT if used,
- exclude PwD by default,
- final round if unspecified,
- OPEN/GN/AI defaults when AIR/eligibility unspecified,
- IIT name via ILIKE,
- include year & round,
- ORDER BY closing_rank_num ASC NULLS LAST,
- LIMIT 15,
- single SELECT/WITH.
If OK, return original; else corrected.

User: "{nl_query}"
SQL:
{sql}

Return ONLY SQL:
"""
        try:
            res = self.model.generate_content(critic)
            fix = (res.text or "").strip()
            if fix and fix != sql:
                if fix.startswith("```"):
                    fix = fix.replace("```sql", "").replace("```", "").strip()
                log_performance("SQL critique", start)
                return fix
            log_performance("SQL critique", start)
            return None
        except Exception as e:
            log_error("SQL critique", e)
            return None

# ========================
# STAGE C: CHAT ANSWER (NO TABLES)
# ========================
class Answerer:
    def __init__(self, api_key: str):
        logger.info("💬 Initializing Answer Generator")
        start = time.time()
        genai.configure(api_key=api_key)
        self.sim = genai.GenerativeModel("gemini-2.0-flash-exp")
        log_performance("Answer Generator initialization", start)

    def answer(self, user_query: str, rows: List[Dict]) -> str:
        logger.info(f"💬 Generating answer for {len(rows)} rows")
        start = time.time()
        if not rows:
            return ("### 🔍 No matches yet\n\n"
                    "Try widening the rank window (±1000), consider more categories, "
                    "check earlier rounds, and add NIT/IIIT backups.")
        data = self._analyze(rows)
        txt = self._compose(user_query, data, rows)
        log_performance("Answer generation", start)
        return txt

    def _analyze(self, rows: List[Dict]) -> Dict:
        inst = {}
        progs = set()
        rmin, rmax = float('inf'), 0
        for r in rows:
            i = r.get('institute', 'Unknown')
            p = r.get('program', 'Unknown')
            crs = str(r.get('closing_rank', '0'))
            cr = int(''.join(filter(str.isdigit, crs)) or '0')
            if cr > 0:
                rmin = min(rmin, cr); rmax = max(rmax, cr)
            progs.add(p)
            inst.setdefault(i, []).append({
                'program': p, 'closing_rank': crs, 'rank_num': cr,
                'category': r.get('category',''), 'quota': r.get('quota',''),
                'gender': r.get('gender',''), 'round': r.get('round',''),
                'year': r.get('year','')
            })
        for k in inst: inst[k].sort(key=lambda x: x['rank_num'] or 10**9)
        return {'institutes': inst, 'programs': list(progs),
                'rank_range': (rmin if rmin != float('inf') else 0, rmax),
                'total': len(rows)}

    def _extract_rank(self, q: str) -> Optional[int]:
        for p in [r'\bAIR\s*(\d+)\b', r'\brank\s*(\d+)\b', r'\bof\s*(\d+)\b', r'\b(\d{3,6})\b']:
            m = re.search(p, q, re.I)
            if m: return int(m.group(1))
        return None

    def _bucket(self, ur: Optional[int], cr: int) -> str:
        if not ur or not cr: return ""
        diff = cr - ur
        if diff >= 1500: return "Likely"
        if diff > -1500: return "Balanced"
        return "Reach"

    def _compose(self, user_query: str, data: Dict, raw: List[Dict]) -> str:
        ur = self._extract_rank(user_query)
        rmin, rmax = data['rank_range']
        flat = []
        for inst, plist in data['institutes'].items():
            for p in plist:
                flat.append({
                    "inst": inst, "prog": p['program'], "cr": p['closing_rank'],
                    "rn": p['rank_num'], "bucket": self._bucket(ur, p['rank_num']),
                    "rd": p['round'], "yr": p['year']
                })

        def sort_key(x):
            order = {"Likely":0, "Balanced":1, "Reach":2, "":3}
            margin = (x['rn'] - ur) if (ur and x['rn']) else 10**9
            round_num = int(x['rd']) if str(x['rd']).isdigit() else -1
            return (order.get(x['bucket'],3), abs(margin), -round_num, x['rn'] or 10**9)

        flat = sorted(flat, key=sort_key)

        lines = []
        lines.append("## 🎓 Your JoSAA Options — Chat Summary")
        chips = []
        if ur: chips.append(f"Rank **{ur:,}**")
        chips.append("Assumed **OPEN · Gender-Neutral · AI · Final round**")
        lines.append(" • ".join(chips))
        lines.append("")
        lines.append(f"Found **{data['total']}** matching seats across **{len(data['institutes'])}** institutes.")
        if rmin and rmax: lines.append(f"Closing ranks range roughly **{rmin:,} → {rmax:,}**.")

        if flat:
            lines.append("")
            lines.append("### ⭐ Top matches for you")
            for x in flat[:6]:
                tag = f" — *{x['bucket']}*" if x['bucket'] else ""
                lines.append(f"- **{x['inst']}**, {x['prog']} (closing rank **{x['cr']}**, R{x['rd']}){tag}")

        # Strategy
        lines.append("")
        lines.append("### 🧭 Strategy")
        if ur and rmin:
            if ur <= rmin + 500:
                lines.append("- You’re **well inside** several cutoffs — prioritise campus/branch fit.")
            elif ur <= rmax:
                lines.append("- You’re in a **competitive** band — order carefully and track later rounds.")
            else:
                lines.append("- It’s **tight** — add more backups and watch special/spot rounds.")
        lines.append("- Build a 3–4–3 stack: **3 Reach • 4–6 Balanced • 3–4 Likely**.")
        lines.append("- Compare curriculum, location, internships/placements from official pages & alumni.")
        lines.append("- Round-to-round swings of ±300–600 are common in popular branches.")

        return "\n".join(lines)

# ========================
# ORCHESTRATOR
# ========================
class Pipeline:
    def __init__(self):
        logger.info("🚀 Initializing JoSAA RAG Pipeline")
        start = time.time()
        self.pg = Pg(DB)
        self.schema_text = self.pg.fetch_schema_text()
        self.policy = Policy()
        self.enhancer = QueryEnhancer(GEMINI_API_KEY, self.policy)
        self.sqlpro = SqlGenPro(GEMINI_API_KEY, self.schema_text, self.policy)
        self.answerer = Answerer(GEMINI_API_KEY)
        log_performance("Pipeline initialization", start)
        logger.info("🎉 JoSAA RAG Pipeline initialized successfully!")

    def _dedup_rows(self, rows: List[Dict]) -> List[Dict]:
        start = time.time()
        rows = sorted(rows, key=_sort_tuple)
        seen, out = set(), []
        for r in rows:
            k = _key_tuple(r)
            if k in seen: continue
            seen.add(k); out.append(r)
        log_performance("Row deduplication", start)
        return out

    @staticmethod
    def _pick_best(cands: List[Tuple[str, List[Dict]]]) -> Tuple[Optional[str], List[Dict]]:
        ne = [c for c in cands if len(c[1]) > 0]
        return (sorted(ne, key=lambda x: len(x[1]), reverse=True)[0]
                if ne else (cands[0] if cands else (None, [])))

    def run(self, user_query: str) -> Tuple[str, str, List[Dict]]:
        logger.info(f"🚀 Starting pipeline execution for query: '{user_query}'")
        start = time.time()
        try:
            variants = self.enhancer.enhance(user_query) or [user_query]
            variants = [re.sub(r'^\s*[-*•]+\s*', '', v) for v in variants] or [user_query]

            sqls = []
            for v in variants:
                s = self.sqlpro.to_sql(v)
                if not s: continue
                try:
                    sqls.append(sanitize_select(s, 15))
                except Exception:
                    continue

            if not sqls:
                return ("I couldn't produce a safe SQL for that.", "", [])

            # unique sqls
            u, seen = [], set()
            for s in sqls:
                h = hash(s.strip().lower())
                if h in seen: continue
                seen.add(h); u.append(s)
            sqls = u

            # Execute
            cands = []
            for s in sqls:
                try:
                    rows = self.pg.run(s)
                    cands.append((s, rows))
                except Exception:
                    pass
            if not cands:
                return ("All generated SQLs failed to run.", sqls[0], [])

            best_sql, best_rows = self._pick_best(cands)
            best_rows = self._dedup_rows(best_rows)
            answer = self.answerer.answer(user_query, best_rows)
            total = log_performance("Complete pipeline execution", start)
            logger.info(f"[Pipeline] rows={len(best_rows)} total={total:.1f}s")
            return answer, best_sql, best_rows
        except Exception as e:
            log_error("Pipeline execution", e)
            return (f"Pipeline error: {e}", "", [])

# ========================
# STREAMLIT UI
# ========================
def init_session_state():
    if 'pipeline' not in st.session_state:
        try:
            st.session_state.pipeline = Pipeline()
            st.session_state.pipeline_loaded = True
            logger.info("✅ Pipeline loaded in session state")
        except Exception as e:
            st.session_state.pipeline_loaded = False
            st.session_state.pipeline_error = str(e)
            log_error("Session state pipeline initialization", e)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    if 'pending_query' not in st.session_state:
        st.session_state.pending_query = None

def get_logs_tail():
    logs = st.session_state.log_stream.getvalue() if hasattr(st.session_state, 'log_stream') else ""
    return logs.split("\n")[-60:] if logs else []

def main():
    st.set_page_config(page_title="JoSAA AI Assistant", page_icon="🎓", layout="wide")

    st.markdown("""
    <style>
    .bubble { padding: 1rem 1.2rem; border-radius: 14px; margin: 0.8rem 0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06); line-height: 1.55; font-size: 0.98rem; }
    .user { background: #e8f0fe; border-left: 5px solid #1a73e8; }
    .assistant { background: #eef7ee; border-left: 5px solid #34a853; }
    .header { text-align:center; padding: 1rem; background: linear-gradient(45deg,#667eea,#764ba2);
      color:#fff; border-radius: 12px; margin-bottom: 0.5rem; }
    .log { background:#0f172a; color:#e2e8f0; padding:.6rem; border-radius:10px; font:12px ui-monospace; max-height:300px; overflow:auto;}
    </style>
    """, unsafe_allow_html=True)

    init_session_state()

    st.markdown("<div class='header'><h3 style='margin:0'>🎓 JoSAA Counselling Assistant</h3><div>Chat-style summaries, no tables.</div></div>", unsafe_allow_html=True)

    col_chat, col_side = st.columns([2,1])

    with col_side:
        st.subheader("🔧 Status")
        if st.session_state.get('pipeline_loaded', False):
            st.success("Pipeline & DB ready")
        else:
            st.error("Pipeline failed")
            st.write(st.session_state.get('pipeline_error',''))

        st.subheader("📝 Logs (tail)")
        st.markdown(f"<div class='log'>{'<br>'.join(get_logs_tail())}</div>", unsafe_allow_html=True)

        st.subheader("💡 Try one")
        samples = [
            "Which colleges are good for Mechanical around AIR 7500?",
            "Electrical engineering options near rank 8000",
            "I have AIR 6000, which IIT programs can I get?",
            "Which college will I get if my rank is 5000?"
        ]
        for q in samples:
            if st.button(q, use_container_width=True, key=f"s_{hash(q)}"):
                st.session_state.pending_query = q
                st.experimental_rerun()

    with col_chat:
        st.subheader("💬 Chat")

        # Chat history (render using wrapper div, then markdown content)
        for m in st.session_state.messages:
            role = m["role"]
            bubble = "user" if role == "user" else "assistant"
            st.markdown(f"<div class='bubble {bubble}'><strong>{'You' if role=='user' else 'Assistant'}:</strong></div>", unsafe_allow_html=True)
            st.markdown(m["content"])  # markdown parsed properly
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if role == "assistant" and m.get("sql"):
                with st.expander("SQL (optional)"):
                    st.code(m["sql"], language="sql")
            if role == "assistant" and m.get("data"):
                with st.expander("Raw rows (optional)"):
                    st.dataframe(pd.DataFrame(m["data"]), use_container_width=True)

        # Input form (clears on submit so safe to rerun)
        with st.form("chat_form", clear_on_submit=True):
            user_text = st.text_input("Ask about JoSAA cutoffs/options (I'll answer in clean paragraphs)…", "")
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)

        # Decide what to run
        query = None
        if submitted and st.session_state.get('pipeline_loaded', False) and user_text.strip():
            query = user_text.strip()
        elif st.session_state.get("pending_query"):
            query = st.session_state.pending_query
            st.session_state.pending_query = None

        # Run once; then show instantly and rerun to update history area at the top
        if query:
            logger.info(f"🎯 User submitted query: '{query}'")
            st.session_state.messages.append({"role":"user","content":query})

            with st.spinner("Working — enhancing → generating SQL → executing → composing…"):
                answer, sql, rows = st.session_state.pipeline.run(query)

            # Show immediately
            st.markdown("<div class='bubble assistant'><strong>Assistant:</strong></div>", unsafe_allow_html=True)
            st.markdown(answer)

            # Save to history and bump counters
            st.session_state.messages.append({"role":"assistant","content":answer,"sql":sql,"data":rows})
            st.session_state.query_count += 1

            # Now refresh so the conversation up top includes this turn
            st.experimental_rerun()

if __name__ == "__main__":
    main()
