"""
Hand-written deterministic baseline solver for the environment-setup domain.

This is NOT an induced solver: it is the bare-Ubuntu analogue of EnvBench's
deterministic baseline script, written by hand as (a) a lower bound for the
benchmark tables and (b) a stand-in for testing the harness.  It follows the same
contract as an induced solver (see domains/envsetup/solver_interface.md): it only
talks to the container through ``shell.run`` and never sees the hidden verifier.

Strategy (repo setup / dependency resolution): probe the container, install the
base toolchain for the detected ecosystem, run the ecosystem's standard install
commands, then self-check (import the package / build).  Database and background
service tasks get minimal, parameter-extracted recipes for the common engines.
"""
from __future__ import annotations

import re
import shlex

APT_BASE = "ca-certificates curl git build-essential pkg-config"


# ------------------------------------------------------------------ helpers
def _ok(shell, cmd, timeout=None):
    r = shell.run(cmd, timeout=timeout)
    return r.exit_code == 0, r.output


def _has(shell, binary):
    return _ok(shell, f"command -v {shlex.quote(binary)} >/dev/null 2>&1")[0]


def _exists(shell, path):
    return _ok(shell, f"test -e {shlex.quote(path)}")[0]


def _apt(shell, pkgs, timeout=900):
    ok, out = _ok(shell, "test -f /var/lib/apt/lists/lock && test -n \"$(ls /var/lib/apt/lists 2>/dev/null | grep -v lock)\"")
    if not ok:
        _ok(shell, "apt-get update -qq", timeout=600)
    return _ok(shell, f"apt-get install -y -qq --no-install-recommends {pkgs}", timeout=timeout)


def _listing(shell):
    ok, out = _ok(shell, "ls -a")
    return set(out.split()) if ok else set()


def _detect_ecosystem(files, hint):
    if {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"} & files:
        return "python"
    if "package.json" in files:
        return "node"
    if "Gemfile" in files:
        return "ruby"
    if "go.mod" in files:
        return "go"
    if "Cargo.toml" in files:
        return "rust"
    if {"pom.xml", "build.gradle", "build.gradle.kts", "mvnw", "gradlew"} & files:
        return "java"
    if hint in ("python", "javascript", "typescript", "ruby", "go", "rust", "java", "c++"):
        return {"javascript": "node", "typescript": "node"}.get(hint, hint)
    return "unknown"


def _python_package_name(shell):
    ok, out = _ok(shell, "grep -m1 -E '^\\s*name\\s*=' pyproject.toml setup.cfg 2>/dev/null; "
                         "grep -m1 -oE \"name\\s*=\\s*['\\\"][A-Za-z0-9_.-]+['\\\"]\" setup.py 2>/dev/null")
    m = re.search(r"name\s*=\s*['\"]?([A-Za-z0-9_.-]+)", out or "")
    return m.group(1).replace("-", "_") if m else None


# ------------------------------------------------------------- ecosystems
def _setup_python(shell, files):
    if not _has(shell, "python3"):
        _apt(shell, f"{APT_BASE} python3 python3-pip python3-dev python3-venv python3-setuptools")
    elif not _has(shell, "pip3"):
        _apt(shell, "python3-pip python3-dev")
    _ok(shell, "python3 -m pip install --no-input -q --upgrade pip setuptools wheel", timeout=600)
    installed = False
    if {"pyproject.toml", "setup.py", "setup.cfg"} & files:
        for extras in ("[dev,test,tests]", "[test]", "[dev]", ""):
            ok, _ = _ok(shell, f"python3 -m pip install --no-input -q -e '.{extras}'", timeout=1800)
            if ok:
                installed = True
                break
        if not installed:
            installed, _ = _ok(shell, "python3 -m pip install --no-input -q .", timeout=1800)
    for req in ("requirements.txt", "requirements-dev.txt", "requirements_dev.txt", "dev-requirements.txt",
                "requirements-test.txt", "test-requirements.txt"):
        if req in files:
            ok, _ = _ok(shell, f"python3 -m pip install --no-input -q -r {req}", timeout=1800)
            installed = installed or ok
    if "Pipfile" in files and not installed:
        _ok(shell, "python3 -m pip install --no-input -q pipenv && pipenv install --system --deploy", timeout=1800)
    _ok(shell, "python3 -m pip install --no-input -q pytest tox", timeout=600)
    pkg = _python_package_name(shell)
    checks = []
    if pkg:
        checks.append(_ok(shell, f"python3 -c 'import {pkg}'")[0])
    checks.append(_ok(shell, "python3 -m pytest --collect-only -q >/dev/null 2>&1 || python3 -m pip check")[0])
    return installed and any(checks)


def _setup_node(shell, files):
    if not _has(shell, "node"):
        _apt(shell, APT_BASE)
        ok, _ = _ok(shell, "curl -fsSL https://deb.nodesource.com/setup_18.x | bash - >/dev/null 2>&1", timeout=600)
        _apt(shell, "nodejs")
    _ok(shell, "corepack enable >/dev/null 2>&1 || npm install -g corepack >/dev/null 2>&1; true")
    if "pnpm-lock.yaml" in files:
        _ok(shell, "npm install -g pnpm >/dev/null 2>&1; true", timeout=600)
        ok, _ = _ok(shell, "pnpm install --frozen-lockfile || pnpm install", timeout=1800)
    elif "yarn.lock" in files:
        _ok(shell, "npm install -g yarn >/dev/null 2>&1; true", timeout=600)
        ok, _ = _ok(shell, "yarn install --frozen-lockfile --non-interactive || yarn install --non-interactive", timeout=1800)
    elif "package-lock.json" in files:
        ok, _ = _ok(shell, "npm ci || npm install", timeout=1800)
    else:
        ok, _ = _ok(shell, "npm install", timeout=1800)
    _ok(shell, "npm run build --if-present", timeout=1800)
    return ok


def _setup_ruby(shell, files):
    if not _has(shell, "ruby"):
        _apt(shell, f"{APT_BASE} ruby-full")
    if not _has(shell, "bundle"):
        _ok(shell, "gem install bundler --no-document", timeout=600)
    ok, _ = _ok(shell, "bundle install --jobs 4", timeout=1800)
    return ok


def _setup_go(shell, files):
    if not _has(shell, "go"):
        _apt(shell, f"{APT_BASE} golang-go")
    ok, _ = _ok(shell, "go build ./...", timeout=1800)
    return ok


def _setup_rust(shell, files):
    if not _has(shell, "cargo"):
        _apt(shell, APT_BASE)
        _ok(shell, "curl -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null 2>&1", timeout=900)
        _ok(shell, 'export PATH="$HOME/.cargo/bin:$PATH"')
    ok, _ = _ok(shell, "cargo build", timeout=3600)
    return ok


def _setup_java(shell, files):
    if not _has(shell, "java"):
        _apt(shell, f"{APT_BASE} default-jdk maven")
    if "mvnw" in files:
        ok, _ = _ok(shell, "chmod +x mvnw && ./mvnw -q -B -DskipTests package", timeout=3600)
    elif "pom.xml" in files:
        ok, _ = _ok(shell, "mvn -q -B -DskipTests package", timeout=3600)
    elif "gradlew" in files:
        ok, _ = _ok(shell, "chmod +x gradlew && ./gradlew build -x test --no-daemon -q", timeout=3600)
    else:
        ok = False
    return ok


def _resolve_dependencies(shell, files, ecosystem_hint):
    """SetupBench dependency-resolution tasks: npm ERESOLVE or bundler compatibility conflicts."""
    if "package.json" in files or (ecosystem_hint or "").startswith("npm"):
        ok, _ = _ok(shell, "npm install --ignore-scripts --package-lock-only --legacy-peer-deps", timeout=1800)
        ok2, _ = _ok(shell, "npm ci --ignore-scripts", timeout=1800)
        return ok2
    if "Gemfile" in files or (ecosystem_hint or "").startswith("bundler"):
        ok, out = _ok(shell, "grep -A1 'BUNDLED WITH' Gemfile.lock | tail -1 | tr -d ' '")
        ver = (out or "").strip()
        if ok and re.match(r"^\d+(\.\d+)+$", ver):
            _ok(shell, f"gem install bundler -v {ver} --no-document", timeout=600)
            ok, _ = _ok(shell, f"bundle _{ver}_ install --jobs 4", timeout=1800)
            if ok:
                return True
        ok, _ = _ok(shell, "bundle install --jobs 4 || (bundle update --conservative && bundle install)", timeout=1800)
        return ok
    return False


# ---------------------------------------------------------- services / dbs
def _param(text, *patterns, default=None):
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return default


def _setup_database(shell, task):
    ps = task["problem_statement"]
    db = _param(ps, r"database named ['`\"]?([A-Za-z0-9_]+)", r"database ['`\"]?([A-Za-z0-9_]+)['`\"]? ")
    user = _param(ps, r"user named ['`\"]?([A-Za-z0-9_]+)", r"user ['`\"]?([A-Za-z0-9_]+)['`\"]? with password")
    pw = _param(ps, r"password ['`\"]?([A-Za-z0-9_!@#$%^&*-]+)")
    port = _param(ps, r"port (\d{3,5})")
    ok, out = _ok(shell, "ls -R . | head -50")
    sql_files = re.findall(r"[\w./-]+\.sql", out or "")
    engine = _detect_engine(ps)
    _apt(shell, APT_BASE)
    if engine == "postgresql":
        _apt(shell, "postgresql postgresql-contrib")
        if port:
            _ok(shell, f"sed -i \"s/^#\\?port = .*/port = {port}/; s/^#\\?listen_addresses = .*/listen_addresses = '*'/\" /etc/postgresql/*/main/postgresql.conf")
        _ok(shell, "service postgresql start", timeout=300)
        p = f"-p {port}" if port else ""
        if user and pw:
            _ok(shell, f"su - postgres -c \"psql {p} -c \\\"CREATE USER {user} WITH PASSWORD '{pw}' SUPERUSER;\\\"\"")
        if db:
            _ok(shell, f"su - postgres -c \"psql {p} -c \\\"CREATE DATABASE {db} OWNER {user or 'postgres'};\\\"\"")
        for f in _ordered_sql(sql_files):
            _ok(shell, f"PGPASSWORD={pw or ''} psql {p} -h 127.0.0.1 -U {user or 'postgres'} -d {db or 'postgres'} -f {f}", timeout=600)
        return _ok(shell, f"pg_isready {p}")[0]
    if engine == "mysql":
        _apt(shell, "mysql-server")
        _ok(shell, "service mysql start", timeout=300)
        if db:
            _ok(shell, f"mysql -e \"CREATE DATABASE IF NOT EXISTS {db};\"")
        if user and pw:
            _ok(shell, f"mysql -e \"CREATE USER IF NOT EXISTS '{user}'@'%' IDENTIFIED BY '{pw}'; CREATE USER IF NOT EXISTS '{user}'@'localhost' IDENTIFIED BY '{pw}'; GRANT ALL ON *.* TO '{user}'@'%'; GRANT ALL ON *.* TO '{user}'@'localhost'; FLUSH PRIVILEGES;\"")
        for f in _ordered_sql(sql_files):
            _ok(shell, f"mysql {db or ''} < {f}", timeout=600)
        return _ok(shell, "mysqladmin ping")[0]
    if engine == "redis":
        _apt(shell, "redis-server redis-tools")
        _ok(shell, f"redis-server --daemonize yes {'--port ' + port if port else ''}", timeout=120)
        return _ok(shell, f"redis-cli {'-p ' + port if port else ''} ping | grep -q PONG")[0]
    if engine == "sqlite":
        _apt(shell, "sqlite3")
        target = _param(ps, r"([\w./-]+\.(?:db|sqlite3?))", default="/testbed/database.db")
        for f in _ordered_sql(sql_files):
            _ok(shell, f"sqlite3 {target} < {f}", timeout=300)
        return _exists(shell, target)
    if engine == "mongodb":
        _apt(shell, "gnupg curl")
        _ok(shell, "curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg && "
                   "echo 'deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse' > /etc/apt/sources.list.d/mongodb-org-7.0.list && apt-get update -qq", timeout=600)
        _apt(shell, "mongodb-org")
        _ok(shell, f"mkdir -p /data/db && mongod --fork --logpath /var/log/mongod.log --dbpath /data/db {'--port ' + port if port else ''}", timeout=120)
        return _ok(shell, "mongosh --quiet --eval 'db.runCommand({ping:1}).ok' | grep -q 1")[0]
    return False


def _detect_engine(text):
    t = text.lower()
    for eng in ("postgresql", "postgres", "mysql", "mariadb", "redis", "mongodb", "mongo", "sqlite"):
        if eng in t:
            return {"postgres": "postgresql", "mariadb": "mysql", "mongo": "mongodb"}.get(eng, eng)
    return None


def _ordered_sql(files):
    uniq = sorted(set(files))
    # schema before seed, migrations in name order
    return sorted(uniq, key=lambda f: (0 if "schema" in f else 1 if "migrat" in f else 2, f))


def _setup_background_service(shell, task):
    ps = task["problem_statement"]
    _apt(shell, f"{APT_BASE} python3 python3-pip supervisor redis-server nginx")
    _ok(shell, "python3 -m pip install --no-input -q gunicorn flask celery redis watchdog", timeout=900)
    _ok(shell, "service redis-server start || redis-server --daemonize yes", timeout=120)
    sock = _param(ps, r"(/[\w./-]+\.sock)")
    if "gunicorn" in ps.lower() and _exists(shell, "app.py"):
        bind = f"unix:{sock}" if sock else "0.0.0.0:8000"
        conf = ("[program:gunicorn]\ncommand=/usr/local/bin/gunicorn --bind " + bind + " app:app\n"
                "directory=" + task["workdir"] + "\nautostart=true\nautorestart=true\n")
        _ok(shell, f"mkdir -p /etc/supervisor/conf.d && printf %s {shlex.quote(conf)} > /etc/supervisor/conf.d/gunicorn.conf")
    if "celery" in ps.lower() and _exists(shell, "tasks.py"):
        conf = ("[program:celery_worker]\ncommand=/usr/local/bin/celery -A tasks worker --loglevel=info\n"
                "directory=" + task["workdir"] + "\nautostart=true\nautorestart=true\n"
                "[program:celery_beat]\ncommand=/usr/local/bin/celery -A tasks beat --loglevel=info\n"
                "directory=" + task["workdir"] + "\nautostart=true\nautorestart=true\n")
        _ok(shell, f"mkdir -p /etc/supervisor/conf.d && printf %s {shlex.quote(conf)} > /etc/supervisor/conf.d/celery.conf")
    _ok(shell, "supervisord -c /etc/supervisor/supervisord.conf || supervisorctl reread && supervisorctl update", timeout=120)
    _ok(shell, "sleep 5")
    ok, out = _ok(shell, "supervisorctl status")
    return ok and "RUNNING" in (out or "")


# ------------------------------------------------------------------- entry
def solve_envsetup(task, shell):
    category = task.get("category", "reposetup")
    _ok(shell, f"cd {shlex.quote(task.get('workdir', '/testbed'))}")
    if category == "dbsetup":
        return {"success": _setup_database(shell, task), "notes": "baseline: database recipe"}
    if category == "bgsetup":
        return {"success": _setup_background_service(shell, task), "notes": "baseline: service recipe"}
    files = _listing(shell)
    if category == "dependency_resolution":
        return {"success": _resolve_dependencies(shell, files, task.get("ecosystem")),
                "notes": "baseline: dependency-resolution recipe"}
    eco = _detect_ecosystem(files, task.get("language"))
    handler = {"python": _setup_python, "node": _setup_node, "ruby": _setup_ruby, "go": _setup_go,
               "rust": _setup_rust, "java": _setup_java}.get(eco)
    if handler is None:
        return {"success": False, "notes": f"baseline: unknown ecosystem (files={sorted(files)[:12]})"}
    if eco != "python" and not _has(shell, "git"):
        _apt(shell, APT_BASE)
    return {"success": bool(handler(shell, files)), "notes": f"baseline: {eco} recipe"}
