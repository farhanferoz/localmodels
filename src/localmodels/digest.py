"""`localmodels digest run`: the monthly check of the sources that can change a serving or training rule.

No language model is involved: every source is checked with a fixed local check (``gh api``, PyPI JSON, plain
HTTP requests, host files) and filtered by paths and keywords, so a run is reproducible from ``.raw``.
Sources: strand B's table D1-D13 (``research-2026-10-18/b-community/report.md`` section 3.2), D15 (new
llama-cpp-python releases, D24), and D14, a
local one (no network): the newest ``localmodels refresh`` run.

Output, in the output directory: ``<date>.md`` (the note), ``.raw/<date>.json`` (every finding and failure)
and ``.state.json`` (what the next run compares with: ETags, versions, the last run time).

A finding line is ``- [K1,K2] D1 llama.cpp: <one-line finding> (<link>)``; ``rules show`` parses it.
A source whose check raises is listed under "Failed sources" and never stops the run.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .atomic import write_text_atomically
from .enums import WorkloadKind

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

EXIT_OK = 0
EXIT_ERROR = 1

DEFAULT_OUT_DIR = Path.home() / "cache" / "localmodels" / "digests"
LLAMA_CPP_DIR = Path.home() / "dev" / "llama.cpp"
FIRST_RUN_DAYS = 35
HTTP_TIMEOUT_S = 30
BROWSER_AGENT = "Mozilla/5.0"
MAX_LINES_PER_SOURCE = 15
SHORT_SHA = 7

LLAMA_REPO = "ggml-org/llama.cpp"
LLAMA_PATHS = (
    "ggml/src/ggml-cuda",
    "src/llama-context.cpp",
    "common/fit.cpp",
    "tools/server",
    "tools/llama-bench",
)
LLAMA_TITLE_WORDS = re.compile(
    r"nvfp4|mxfp4|blackwell|sm_12|gb10|spark|unified|fit|kv|spec|mtp|embed|rerank|load|deprecat",
    re.IGNORECASE,
)
DEPRECATION = re.compile("deprecat", re.IGNORECASE)
LLAMA_DISCUSSIONS = (16578, 16514)
GB10_WORDS = re.compile(r"sm_?12|GB10|DGX Spark|nvfp4|mxfp", re.IGNORECASE)
FORUM_WORDS = re.compile(
    r"llama\.cpp|GB10|sm_121|thermal|driver|kernel|regression|freeze|unified|NGC|vLLM|SGLang", re.IGNORECASE
)
FORUM_URL = "https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719.json"
FORUM_TOPIC_URL = "https://forums.developer.nvidia.com/t/{slug}/{id}"
SPARK_DOCS = (
    "https://docs.nvidia.com/dgx/dgx-spark/release-notes.html",
    "https://docs.nvidia.com/dgx/dgx-spark/known-issues.html",
)
NGC_VLLM_NOTES = "https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html"
PYPI_JSON = "https://pypi.org/pypi/{package}/json"
PYPI_TORCH = PYPI_JSON.format(package="torch")
#: Its GitHub releases are one tag per hardware variant (-vulkan, -metal, ...); PyPI names the one version.
PYPI_LLAMA_CPP_PYTHON = PYPI_JSON.format(package="llama-cpp-python")
TORCH_ARCH_FILE = ".ci/manywheel/build_env_setup.py"
AARCH64_ARCHS = re.compile(r'"aarch64"\s*:\s*\{([^}]*)\}')
SM121 = "121"
LIBRARY_REPOS = (
    "pytorch/ao",
    "NVIDIA/TransformerEngine",
    "flashinfer-ai/flashinfer",
    "huggingface/transformers",
)
ENGINE_REPOS = ("vllm-project/vllm", "sgl-project/sglang", "NVIDIA/TensorRT-LLM")
PLAYBOOK_REPO = "NVIDIA/dgx-spark-playbooks"
PLAYBOOK_DIRS = (
    "vllm",
    "sglang",
    "trt-llm",
    "llama-cpp",
    "pytorch-fine-tune",
    "nvfp4-quantization",
    "speculative-decoding",
)
ARENA_REPO = "spark-arena/recipe-registry"
ARENA_TOOLS = (
    "eugr/llama-benchy",
    "spark-arena/sparkrun",
    "eugr/spark-vllm-docker",
    "timothystewart6/vllm-gb10",
)
AWESOME_REPO = "bidual/awesome-dgx-spark"
# Organisations whose new models are checked, and the model families we serve (extend both here).
HF_ORGS = ("Qwen", "unsloth", "google", "microsoft", "openai", "nvidia", "ggml-org")
HF_FAMILIES = re.compile(r"qwen|phi|gemma|gpt-oss|nemotron|glm|llama", re.IGNORECASE)
HF_LIST_URL = "https://huggingface.co/api/models?author={org}&sort=lastModified&direction=-1&limit=20"
HOST_UPGRADE_WORDS = re.compile(r"nvidia|cuda|dgx", re.IGNORECASE)
DGX_RELEASE = "/etc/dgx-release"
SPARK_TABLE = re.compile(r"DGX OS\s+([\d.]+)\s+NVIDIA GPU Driver\s+([\d.]+)\s+NVIDIA CUDA Toolkit\s+([\d.]+)")
TAGS = re.compile(r"<[^>]+>|\s+")


class Source(StrEnum):
    """The digest's sources: D1-D13 and D15 look outside, D14 is this host's own refresh run."""

    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"
    D5 = "D5"
    D6 = "D6"
    D7 = "D7"
    D8 = "D8"
    D9 = "D9"
    D10 = "D10"
    D11 = "D11"
    D12 = "D12"
    D13 = "D13"
    D14 = "D14"
    D15 = "D15"


ALL_KINDS = tuple(WorkloadKind)
LLAMA_KINDS = (WorkloadKind.K1, WorkloadKind.K2, WorkloadKind.K3)
TORCH_KINDS = (
    WorkloadKind.K4,
    WorkloadKind.K5,
    WorkloadKind.K6,
    WorkloadKind.K8,
    WorkloadKind.K9,
    WorkloadKind.K10,
)
SERVING_KINDS = (WorkloadKind.K1, WorkloadKind.K2)
MODEL_KINDS = (WorkloadKind.K1, WorkloadKind.K2, WorkloadKind.K4, WorkloadKind.K5, WorkloadKind.K6)

LINE_RE = re.compile(
    r"^- \[(?P<kinds>K\d+(?:,K\d+)*)\] (?P<source>D\d+) (?P<name>.+?): (?P<text>.*) \((?P<link>.*)\)$"
)


@dataclass(frozen=True)
class Finding:
    """One finding of a source: a one-line text and the link to read."""

    text: str
    link: str


class Net(Protocol):
    """What a check may touch: GitHub, plain HTTP, host files and commands (a test replaces it)."""

    def gh_json(self, endpoint: str) -> Any:
        """The JSON of a GitHub REST endpoint."""
        ...

    def gh_text(self, endpoint: str) -> str:
        """The raw text of a GitHub contents endpoint."""
        ...

    def get(self, url: str, *, browser: bool = False) -> str:
        """The body of a GET; ``browser`` sends a browser user agent."""
        ...

    def head(self, url: str) -> dict[str, str]:
        """The response headers of a HEAD, names in lower case."""
        ...

    def read(self, path: str) -> str:
        """A host file's text."""
        ...

    def run(self, argv: list[str]) -> str:
        """A host command's output."""
        ...


class LiveNet:
    """The real thing: ``gh api``, urllib, host files and commands."""

    def gh_json(self, endpoint: str) -> Any:  # noqa: D102
        return json.loads(self.run(["gh", "api", endpoint]))

    def gh_text(self, endpoint: str) -> str:  # noqa: D102
        return self.run(["gh", "api", endpoint, "-H", "Accept: application/vnd.github.raw"])

    def get(self, url: str, *, browser: bool = False) -> str:  # noqa: D102
        request = urllib.request.Request(url, headers={"User-Agent": BROWSER_AGENT} if browser else {})  # noqa: S310
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310
            return str(response.read().decode("utf-8", "replace"))

    def head(self, url: str) -> dict[str, str]:  # noqa: D102
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": BROWSER_AGENT})  # noqa: S310
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310
            return {k.lower(): v for k, v in response.headers.items()}

    def read(self, path: str) -> str:  # noqa: D102
        return Path(path).read_text(encoding="utf-8")

    def run(self, argv: list[str]) -> str:  # noqa: D102
        done = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=120)  # noqa: S603
        return done.stdout


@dataclass
class Context:
    """What one check gets: the network, the time of the last run, and its own state from the last run."""

    net: Net
    since: str
    mem: dict[str, Any] = field(default_factory=dict)

    def commits(self, repo: str, path: str | None = None) -> list[Any]:
        """The commits of ``repo`` (under ``path``) since the last run."""
        query = f"since={self.since}&per_page=100" + (f"&path={path}" if path else "")
        return list(self.net.gh_json(f"repos/{repo}/commits?{query}"))

    def new_releases(self, repo: str) -> list[Any]:
        """The releases of ``repo`` published since the last run."""
        releases = self.net.gh_json(f"repos/{repo}/releases?per_page=10")
        return [r for r in releases if (r.get("published_at") or "") > self.since]

    def changed(self, key: str, value: str) -> bool:
        """Whether ``value`` differs from what the last run stored under ``key``; stores it."""
        previous = self.mem.get(key)
        self.mem[key] = value
        return previous is not None and previous != value


def _title(commit: Any) -> str:
    return str(commit["commit"]["message"]).splitlines()[0]


def _commit_findings(
    commits: list[Any], prefix: str, *, keywords: re.Pattern[str] | None = None
) -> list[Finding]:
    return [
        Finding(f"{prefix}{_title(c)}", c["html_url"])
        for c in commits
        if keywords is None or keywords.search(_title(c))
    ]


def _release_findings(
    ctx: Context, repos: tuple[str, ...], *, pattern: re.Pattern[str] | None
) -> list[Finding]:
    """A line per new release (all of them with no ``pattern``), else per match in the body."""
    found: list[Finding] = []
    for repo in repos:
        for release in ctx.new_releases(repo):
            tag = release.get("tag_name", "?")
            if pattern is None:
                found.append(Finding(f"{repo} {tag}", release["html_url"]))
                continue
            hits = sorted({m.group(0).lower() for m in pattern.finditer(release.get("body") or "")})
            if hits:
                found.append(Finding(f"{repo} {tag} mentions {', '.join(hits)}", release["html_url"]))
    return found


def _installed_llama_commit(net: Net) -> str:
    """The build ``current`` points to (what projects run); else the hand-made build's commit."""
    from .llama_builds import current_commit, llama_home  # noqa: PLC0415 - llama_builds does not import this

    return (
        current_commit(llama_home())
        or net.run(["git", "-C", str(LLAMA_CPP_DIR), "rev-parse", "HEAD"]).strip()
    )


def check_llama_cpp(ctx: Context) -> list[Finding]:
    """D1: commits ahead of the installed build, and the keyword-matching ones on the watched paths."""
    installed = _installed_llama_commit(ctx.net)
    compare = ctx.net.gh_json(f"repos/{LLAMA_REPO}/compare/{installed}...master?per_page=1")
    found = [
        Finding(
            f"master is {compare['ahead_by']} commits ahead of the installed {installed[:SHORT_SHA]}",
            compare["html_url"],
        )
    ]
    seen: set[str] = set()
    for path in LLAMA_PATHS:
        for commit in ctx.commits(LLAMA_REPO, path):
            if commit["sha"] in seen or not LLAMA_TITLE_WORDS.search(_title(commit)):
                continue
            seen.add(commit["sha"])
            red = "RED " if DEPRECATION.search(_title(commit)) else ""
            found.append(Finding(f"{red}{_title(commit)} [{path}]", commit["html_url"]))
    return found


def check_llama_bench_numbers(ctx: Context) -> list[Finding]:
    """D2: any new commit under the maintainer's Spark numbers."""
    return _commit_findings(ctx.commits(LLAMA_REPO, "benches/dgx-spark"), "maintainer's Spark numbers: ")


def check_llama_discussions(ctx: Context) -> list[Finding]:
    """D3: discussions #16578 and #16514 updated since the last run."""
    found = []
    for number in LLAMA_DISCUSSIONS:
        item = ctx.net.gh_json(f"repos/{LLAMA_REPO}/discussions/{number}")
        if item["updated_at"] > ctx.since:
            found.append(Finding(f"#{number} updated: {item['title']}", item["html_url"]))
    return found


def _pypi_version(ctx: Context, url: str) -> str:
    """The newest release PyPI lists at ``url`` (a package's JSON page)."""
    return str(json.loads(ctx.net.get(url))["info"]["version"])


def check_pytorch(ctx: Context) -> list[Finding]:
    """D4: a new PyTorch minor, 121 in the aarch64 arch set, and new GB10 issues."""
    version = _pypi_version(ctx, PYPI_TORCH)
    minor = ".".join(version.split(".")[:2])
    found = []
    if ctx.changed("minor", minor):
        found.append(Finding(f"new minor {minor} (release {version}): PyTorch rules DUE", PYPI_TORCH))
    source = ctx.net.gh_text(f"repos/pytorch/pytorch/contents/{TORCH_ARCH_FILE}?ref=v{version}")
    arch_sets = [set(re.findall(r"\d+", m)) for m in AARCH64_ARCHS.findall(source)]
    if not arch_sets:
        msg = f"no aarch64 arch set found in {TORCH_ARCH_FILE} at v{version}"
        raise ValueError(msg)
    if any(SM121 in archs for archs in arch_sets):
        found.append(
            Finding(
                f"{SM121} is in the aarch64 arch set of v{version}: the warning caveat can be retired",
                f"https://github.com/pytorch/pytorch/blob/v{version}/{TORCH_ARCH_FILE}",
            )
        )
    query = urllib.parse.quote(f'repo:pytorch/pytorch "DGX Spark" created:>={ctx.since[:10]}')
    for issue in ctx.net.gh_json(f"search/issues?q={query}")["items"]:
        found.append(Finding(f"issue: {issue['title']}", issue["html_url"]))  # noqa: PERF401
    return found


def check_llama_cpp_python(ctx: Context) -> list[Finding]:
    """D15: a new llama-cpp-python release; projects upgrade it themselves (D24), then re-tune (D2)."""
    version = _pypi_version(ctx, PYPI_LLAMA_CPP_PYTHON)
    if not ctx.changed("version", version):
        return []
    text = f"release {version}: a project that upgrades it gets its models re-tuned (D2, D24)"
    return [Finding(text, PYPI_LLAMA_CPP_PYTHON)]


def check_libraries(ctx: Context) -> list[Finding]:
    """D5: torchao, TransformerEngine, FlashInfer, transformers releases that mention GB10."""
    return _release_findings(ctx, LIBRARY_REPOS, pattern=GB10_WORDS)


def check_engines(ctx: Context) -> list[Finding]:
    """D6: a new vLLM, SGLang or TRT-LLM version, and a change of the NGC vLLM release-notes index."""
    found = _release_findings(ctx, ENGINE_REPOS, pattern=None)
    if ctx.changed("ngc_etag", ctx.net.head(NGC_VLLM_NOTES).get("etag", "")):
        found.append(Finding("NGC vLLM release-notes index changed", NGC_VLLM_NOTES))
    return found


def check_spark_docs(ctx: Context) -> list[Finding]:
    """D7: the two NVIDIA Spark pages changed (ETag), and the OS/driver/CUDA row of the release notes."""
    found = []
    for url in SPARK_DOCS:
        headers = ctx.net.head(url)
        stamp = headers.get("etag", "") or headers.get("last-modified", "")
        if ctx.changed(f"etag:{url}", stamp):
            found.append(Finding(f"page changed (last modified {headers.get('last-modified', '?')})", url))
    text = TAGS.sub(" ", ctx.net.get(SPARK_DOCS[0]))
    if (row := SPARK_TABLE.search(text)) and ctx.changed("table", "/".join(row.groups())):
        found.append(
            Finding("release table now: DGX OS {}, driver {}, CUDA {}".format(*row.groups()), SPARK_DOCS[0])
        )
    return found


def check_forum(ctx: Context) -> list[Finding]:
    """D8: new topics of the DGX Spark category whose title matches the keywords."""
    topics = json.loads(ctx.net.get(FORUM_URL, browser=True))["topic_list"]["topics"]
    return [
        Finding(t["title"], FORUM_TOPIC_URL.format(slug=t["slug"], id=t["id"]))
        for t in topics
        if t["created_at"] > ctx.since and FORUM_WORDS.search(t["title"])
    ]


def check_playbooks(ctx: Context) -> list[Finding]:
    """D9: new commits under the playbook folders we follow."""
    found: list[Finding] = []
    for folder in PLAYBOOK_DIRS:
        found += _commit_findings(ctx.commits(PLAYBOOK_REPO, f"nvidia/{folder}"), f"playbook {folder}: ")
    return found


def check_arena(ctx: Context) -> list[Finding]:
    """D10: Spark Arena benchmarking profile commits and new releases of its tools."""
    found = _commit_findings(ctx.commits(ARENA_REPO, "benchmarking"), "benchmarking profile: ")
    return found + _release_findings(ctx, ARENA_TOOLS, pattern=None)


def check_awesome(ctx: Context) -> list[Finding]:
    """D11: new commits of the curated index."""
    return _commit_findings(ctx.commits(AWESOME_REPO), "")


def check_models(ctx: Context) -> list[Finding]:
    """D12: models created since the last run, by the organisations and in the families we serve."""
    found = []
    for org in HF_ORGS:
        for model in json.loads(ctx.net.get(HF_LIST_URL.format(org=org))):
            if model.get("createdAt", "") > ctx.since and HF_FAMILIES.search(model["id"]):
                found.append(Finding(f"new model {model['id']}", f"https://huggingface.co/{model['id']}"))  # noqa: PERF401
    return found


def check_refresh(_ctx: Context) -> list[Finding]:
    """D14: what the newest ``localmodels refresh`` run did with its candidate, and the rules it measured."""
    from .refresh import headline, newest_view, rules_line  # noqa: PLC0415 - refresh imports rules, which imports this

    view = newest_view()
    if view is None:
        return []
    link = str(view.folder)
    return [Finding(text, link) for text in (headline(view), rules_line(view)) if text]


def check_host(ctx: Context) -> list[Finding]:
    """D13: a changed host component (DGX OS, driver, kernel) and the upgradable NVIDIA/CUDA/DGX packages."""
    now = {
        "dgx_os": ctx.net.read(DGX_RELEASE).strip(),
        "driver": ctx.net.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip(),
        "kernel": ctx.net.run(["uname", "-r"]).strip(),
    }
    found = [
        Finding(f"{name} changed to {value}: rules that depend on it are STALE", "host")
        for name, value in now.items()
        if ctx.changed(name, value)
    ]
    upgrades = ctx.net.run(["apt", "list", "--upgradable"]).splitlines()
    found += [
        Finding(f"upgradable: {u.split('/')[0]}", "apt list --upgradable")
        for u in upgrades
        if HOST_UPGRADE_WORDS.search(u)
    ]
    return found


@dataclass(frozen=True)
class SourceSpec:
    """A source's display name, the workload kinds it touches and its check."""

    name: str
    kinds: tuple[WorkloadKind, ...]
    check: Callable[[Context], list[Finding]]
    local: bool = False  # reads this host's files only, so it cannot fail with the network down


SOURCES: dict[Source, SourceSpec] = {
    Source.D1: SourceSpec("llama.cpp", LLAMA_KINDS, check_llama_cpp),
    Source.D2: SourceSpec("llama.cpp", LLAMA_KINDS, check_llama_bench_numbers),
    Source.D3: SourceSpec("llama.cpp", LLAMA_KINDS, check_llama_discussions),
    Source.D4: SourceSpec("PyTorch", TORCH_KINDS, check_pytorch),
    Source.D5: SourceSpec("PyTorch libraries", TORCH_KINDS, check_libraries),
    Source.D6: SourceSpec("vLLM/SGLang/TRT-LLM", (WorkloadKind.K2,), check_engines),
    Source.D7: SourceSpec("NVIDIA docs", ALL_KINDS, check_spark_docs),
    Source.D8: SourceSpec("NVIDIA forum", SERVING_KINDS, check_forum),
    Source.D9: SourceSpec("NVIDIA playbooks", SERVING_KINDS, check_playbooks),
    Source.D10: SourceSpec("Spark Arena", SERVING_KINDS, check_arena),
    Source.D11: SourceSpec("awesome-dgx-spark", SERVING_KINDS, check_awesome),
    Source.D12: SourceSpec("model releases", MODEL_KINDS, check_models),
    Source.D13: SourceSpec("this host", ALL_KINDS, check_host),
    Source.D14: SourceSpec("llama.cpp refresh", LLAMA_KINDS, check_refresh, local=True),
    Source.D15: SourceSpec("llama-cpp-python", LLAMA_KINDS, check_llama_cpp_python),
}


def format_line(source: Source, finding: Finding) -> str:
    """The note's finding line, the format ``localmodels rules show`` parses."""
    spec = SOURCES[source]
    kinds = ",".join(spec.kinds)
    return f"- [{kinds}] {source} {spec.name}: {finding.text} ({finding.link})"


@dataclass
class Outcome:
    """One source's result: its findings, or why its check failed."""

    findings: list[Finding] = field(default_factory=list)
    failed: str | None = None


def run_sources(net: Net, state: dict[str, Any], since: str) -> dict[Source, Outcome]:
    """Check every source; a source that raises is recorded as failed, its stored state left as it was."""
    outcomes: dict[Source, Outcome] = {}
    for source, spec in SOURCES.items():
        ctx = Context(net, since, dict(state.get(source, {})))
        try:
            findings = spec.check(ctx)
        except Exception as exc:  # noqa: BLE001  one broken source never stops the run
            outcomes[source] = Outcome(failed=f"{type(exc).__name__}: {exc}")
            continue
        outcomes[source] = Outcome(findings=findings)
        state[source] = ctx.mem
    return outcomes


def render_note(day: str, since: str, outcomes: dict[Source, Outcome]) -> str:
    """The dated note: finding lines, then quiet sources, then failed sources."""
    lines = [f"# localmodels digest {day}", "", f"Changes since {since}.", "", "## Findings", ""]
    body: list[str] = []
    for source, outcome in outcomes.items():
        body += [format_line(source, f) for f in outcome.findings[:MAX_LINES_PER_SOURCE]]
        if (hidden := len(outcome.findings) - MAX_LINES_PER_SOURCE) > 0:
            body.append(format_line(source, Finding(f"{hidden} more in the raw file", ".raw")))
    lines += body or ["None."]
    quiet = [f"{s} {SOURCES[s].name}" for s, o in outcomes.items() if o.failed is None and not o.findings]
    if quiet:
        lines += ["", "## Quiet sources", "", *[f"- {q}" for q in quiet]]
    failed = [f"- {s} {SOURCES[s].name}: {o.failed}" for s, o in outcomes.items() if o.failed is not None]
    if failed:
        lines += ["", "## Failed sources", "", *failed]
    return "\n".join(lines) + "\n"


def run_digest(net: Net, out_dir: Path, now: datetime) -> int:
    """Run all sources, write the note, the raw file and the state; exit 1 if every network source failed."""
    state_path = out_dir / ".state.json"
    state: dict[str, Any] = json.loads(state_path.read_text()) if state_path.is_file() else {}
    since = state.get("last_run") or (now - timedelta(days=FIRST_RUN_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    outcomes = run_sources(net, state, since)
    day = now.strftime("%Y-%m-%d")
    state["last_run"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    write_text_atomically(out_dir / f"{day}.md", render_note(day, since, outcomes))
    raw = {"date": day, "since": since, "sources": {s: asdict(o) for s, o in outcomes.items()}}
    write_text_atomically(out_dir / ".raw" / f"{day}.json", json.dumps(raw, indent=2) + "\n")
    write_text_atomically(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    failed = sum(o.failed is not None for o in outcomes.values())
    network = [o for s, o in outcomes.items() if not SOURCES[s].local]
    findings = sum(len(o.findings) for o in outcomes.values())
    note = out_dir / f"{day}.md"
    print(f"digest {day}: {findings} findings, {failed} of {len(outcomes)} sources failed -> {note}")  # noqa: T201
    return EXIT_ERROR if all(o.failed is not None for o in network) else EXIT_OK


def cmd_digest_run(args: argparse.Namespace) -> int:
    """Run the digest against the live sources."""
    return run_digest(LiveNet(), Path(args.out).expanduser(), datetime.now(UTC))


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``digest run`` to the command line."""
    digest_p = subparsers.add_parser("digest", help="monthly check of the sources that can change a rule")
    digest_sub = digest_p.add_subparsers(dest="digest_command", required=True)
    run_p = digest_sub.add_parser("run", help="check the sources and write the dated note")
    run_p.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output directory (default: %(default)s)")
    run_p.set_defaults(fn=cmd_digest_run)
