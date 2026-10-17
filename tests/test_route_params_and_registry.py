"""Request-param coercion on the chat route + atomic provider-registry loading.

Five defects, one file each section:

* **R1** ``_parse_reasoning_effort`` returned ``body["reasoning_effort"]`` on
  the invalid path (server_routes.py:984-993). The nested ``{"reasoning":
  {"effort": ...}}`` spelling that agent harnesses send therefore warned
  about nothing: ``unsupported["reasoning_effort"]`` stayed unset, so the
  per-request "ignoring unsupported OpenAI params" line never mentioned it.
* **R2** ``stream`` was read as ``bool(body.get("stream"))``, so
  ``{"stream": "false"}`` -- JSON-quoted booleans, what shell/YAML wrappers
  and hand-rolled clients send -- streamed SSE at a caller that asked for
  JSON.
* **R3** ``parallel_tool_calls`` appeared nowhere in ``src/``, and the
  ``tool_choice`` variants that ask for a FORCED call ("required", a named
  function) were indistinguishable from "auto". Both were dropped with no log
  line at all, unlike temperature/top_p/n/stop.
* **R4** ``registry._LOADED`` was published BEFORE the imports and without a
  lock, so a request arriving during ``warmup_on_startup`` saw a partial
  registry and answered 404 "model not available" (or 500 from
  ``default()``). A failed import used to be retried on every access, which
  turned one broken module into a warning per request; it is now skipped for
  the process lifetime, like every other failing import.
* **R5** ``n`` and ``stream_options`` had no hostile-input coverage. They are
  already safe (int() wrapped in ``except (TypeError, ValueError)``, the
  dict check on stream_options); these tests pin that down so it stays true.
"""

from __future__ import annotations

import logging
import threading

import pytest
from fastapi.testclient import TestClient

from deepseaport.config import AccountConfig, Settings
from deepseaport.core import registry
from deepseaport.core.provider import ModelSpec, Provider
from deepseaport.server import create_app
from support import settings as _settings
from deepseaport.server_routes import (
    _extract_passthrough,
    _parse_reasoning_effort,
    _prepare,
)

MODEL = "deepseek-flash"
MSGS = [{"role": "user", "content": "hi"}]



def _body(**kw) -> dict:
    body = {"model": MODEL, "messages": MSGS}
    body.update(kw)
    return body


class _FakeProvider(Provider):
    """Route seam provider: no accounts, no engine, records what it was given."""

    id = "fakep"
    display_name = "FakeP"
    requires_accounts = False

    def __init__(self) -> None:
        self.preps: list[dict] = []

    def models(self) -> dict[str, ModelSpec]:
        return {"fakep-x": ModelSpec(id="fakep-x")}

    def complete_request(self, app, pool, prep, item, cancel_event):
        self.preps.append(prep)
        return {"content": "P", "thinking": "", "usage_total": 1}

    async def stream_response(self, app, pool, item, prep):
        self.preps.append(prep)
        yield b'data: {"choices": [{"delta": {"content": "P"}}]}\n\n'
        yield b"data: [DONE]\n\n"


def _fake_app(monkeypatch) -> tuple[TestClient, _FakeProvider]:
    """App wired to the seam provider. Two or more ids in ``providers`` are
    required: a one-entry list is ignored by design (pools.active_provider_ids),
    so single mode would resolve the model against deepseek and 404."""
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    provider = _FakeProvider()
    registry._REGISTRY[registry._key(provider)] = provider
    settings = _settings(providers=["deepseek", "fakep"],
                         accounts=[AccountConfig(email="f@x.com", token="t",
                                                 provider="fakep")])
    return TestClient(create_app(settings)), provider


# --- R1: an invalid reasoning_effort must warn in BOTH spellings ---------------


@pytest.mark.parametrize("field,body_patch,expected", [
    ("flat", {"reasoning_effort": "ultra"}, "ultra"),
    ("flat", {"reasoning_effort": " Ultra "}, "ultra"),  # normalized before return
    ("nested", {"reasoning": {"effort": "ultra"}}, "ultra"),
    ("nested", {"reasoning": {"effort": "ULTRA"}}, "ultra"),
])
def test_invalid_reasoning_effort_returns_a_marker_in_both_spellings(
        field, body_patch, expected):
    effort, invalid_raw = _parse_reasoning_effort(_body(**body_patch))

    assert effort is None, f"{field}: an unknown variant must not be forwarded"
    assert invalid_raw == expected, (
        f"{field}: the invalid value was dropped instead of reported "
        f"(got {invalid_raw!r})")


@pytest.mark.parametrize("body_patch", [
    {"reasoning_effort": "ultra"},
    {"reasoning": {"effort": "ultra"}},
])
def test_invalid_reasoning_effort_is_reported_in_unsupported_params(body_patch):
    prep = _prepare(_body(**body_patch), _settings())

    assert prep["reasoning_effort"] is None
    assert prep["unsupported_params"].get("reasoning_effort") == "ultra", (
        "the nested spelling left unsupported_params empty, so nothing warned")


@pytest.mark.parametrize("body_patch", [
    {"reasoning_effort": "ultra"},
    {"reasoning": {"effort": "ultra"}},
])
def test_invalid_reasoning_effort_logs_the_warning(body_patch, caplog):
    with caplog.at_level(logging.WARNING, logger="deepseaport.server"):
        _prepare(_body(**body_patch), _settings())

    assert "ignoring unsupported OpenAI params" in caplog.text
    assert "reasoning_effort" in caplog.text


def test_valid_reasoning_effort_still_not_warned():
    prep = _prepare(_body(reasoning={"effort": "High"}), _settings())
    assert prep["reasoning_effort"] == "high"
    assert "reasoning_effort" not in prep["unsupported_params"]


# --- R2: `stream` is parsed, not bool()-coerced --------------------------------


@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False),
    ("true", True), ("false", False),
    ("TRUE", True), ("False", False),
    ("  true  ", True), ("\tfalse\n", False),
    ("1", True), ("0", False),
    ("yes", True), ("no", False),
    ("on", True), ("off", False),
])
def test_stream_spellings(raw, expected):
    assert _prepare(_body(stream=raw), _settings())["stream"] is expected


def test_stream_missing_defaults_to_false():
    assert "stream" not in _body()
    assert _prepare(_body(), _settings())["stream"] is False


@pytest.mark.parametrize("raw", ["maybe", "", "  ", "2", 2, {}, [], None])
def test_unrecognisable_stream_keeps_the_default(raw):
    """Odd-but-previously-accepted input must not become a 400."""
    prep = _prepare(_body(stream=raw), _settings())
    assert prep["stream"] is False


@pytest.mark.parametrize("raw,expect_sse", [
    ("false", False), ("no", False), ("0", False), ("off", False),
    ("true", True), ("1", True), (True, True), (False, False),
])
def test_stream_string_drives_the_actual_response_type(raw, expect_sse, monkeypatch):
    client, provider = _fake_app(monkeypatch)

    resp = client.post("/v1/chat/completions",
                       json={"model": "fakep-x", "messages": MSGS, "stream": raw})

    assert resp.status_code == 200, resp.text
    got_sse = "text/event-stream" in resp.headers["content-type"]
    assert got_sse is expect_sse, (
        f"stream={raw!r} produced {resp.headers['content-type']}")
    assert provider.preps and provider.preps[0]["stream"] is expect_sse


def test_unrecognisable_stream_does_not_400(monkeypatch):
    client, _ = _fake_app(monkeypatch)

    resp = client.post("/v1/chat/completions",
                       json={"model": "fakep-x", "messages": MSGS,
                             "stream": "eventually"})

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" not in resp.headers["content-type"]
    assert resp.json()["choices"][0]["message"]["content"] == "P"


# --- R3: unhonored tool params are warn-and-drop, like the others ---------------


@pytest.mark.parametrize("value,expected", [
    (False, False), (True, True),
])
def test_parallel_tool_calls_present_is_reported(value, expected):
    prep = _prepare(_body(parallel_tool_calls=value), _settings())
    assert prep["unsupported_params"].get("parallel_tool_calls") is value


def test_parallel_tool_calls_absent_is_not_reported():
    prep = _prepare(_body(), _settings())
    assert "parallel_tool_calls" not in prep["unsupported_params"]


def test_parallel_tool_calls_logs_like_the_other_dropped_params(caplog):
    with caplog.at_level(logging.WARNING, logger="deepseaport.server"):
        _prepare(_body(parallel_tool_calls=False), _settings())

    assert "parallel_tool_calls" in caplog.text


def test_tool_choice_none_still_disables_tools_and_is_not_reported():
    tools = [{"function": {"name": "get_time"}}]
    prep = _prepare(_body(tools=tools, tool_choice="none"), _settings())
    assert prep["tools"] == [], "tool_choice=none stopped gating tools off"
    assert "tool_choice" not in prep["unsupported_params"]


def test_tool_choice_auto_is_not_reported():
    prep = _prepare(_body(tools=[{"function": {"name": "get_time"}}],
                          tool_choice="auto"), _settings())
    assert prep["unsupported_params"].get("tool_choice") is None


@pytest.mark.parametrize("choice", [
    "required",
    {"type": "function", "function": {"name": "get_time"}},
])
def test_tool_choice_forcing_a_call_is_reported(choice):
    tools = [{"function": {"name": "get_time"}}]
    prep = _prepare(_body(tools=tools, tool_choice=choice), _settings())

    assert prep["unsupported_params"].get("tool_choice") == choice
    # Reporting must not change whether tools are enabled: the tools are still
    # rendered, the bridge just cannot guarantee a tool_call comes back.
    assert len(prep["tools"]) == 1


# --- R5: hostile n / stream_options -------------------------------------------


@pytest.mark.parametrize("value,reported", [
    # The coerced int itself has no reader in prep; what a caller can observe
    # is the warn-and-drop accounting: a value that reads as something other
    # than 1 is named there, everything else is silently read as 1.
    ("many", False), ({}, False), ([], False), (2.7, True), (None, False),
    ("2", True), (0, False), (True, False)])
def test_hostile_n_is_coerced_not_raised(value, reported):
    prep = _prepare(_body(n=value), _settings())
    assert ("n" in prep["unsupported_params"]) is reported


@pytest.mark.parametrize("value", ["x", [], 5, 0, "", {"include_usage": True}])
def test_hostile_stream_options_is_coerced_not_raised(value):
    prep = _prepare(_body(stream_options=value), _settings())
    assert isinstance(prep["stream_options"], dict)
    if value != {"include_usage": True}:
        assert prep["stream_options"] == {}


@pytest.mark.parametrize("n,opts", [("many", "x"), ({}, []), (2.7, 5), (None, None)])
def test_hostile_n_and_stream_options_answer_over_http(n, opts, monkeypatch):
    client, provider = _fake_app(monkeypatch)

    resp = client.post("/v1/chat/completions",
                       json={"model": "fakep-x", "messages": MSGS,
                             "n": n, "stream_options": opts})

    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "P"
    # A numeric n is truncated (2.7 -> 2) and warned about, as it always was;
    # what matters here is that nothing raised out of _prepare as a 500.
    assert ("n" in provider.preps[0]["unsupported_params"]) is (n == 2.7)
    assert provider.preps[0]["stream_options"] == {}


# --- R4: registry loading is atomic, and a broken module is skipped once -------


class _StubProvider(Provider):
    """Module-scope stub for registry loading tests (needs only .id + models)."""

    def __init__(self, provider_id: str, model_id: str) -> None:
        self.id = provider_id
        self._model_id = model_id

    def models(self) -> dict[str, ModelSpec]:
        return {self._model_id: ModelSpec(id=self._model_id)}


class _StubModule:
    """Stands in for an imported provider package exposing create_provider()."""

    def __init__(self, name: str) -> None:
        pid = name.split(".")[-1]
        self.create_provider = lambda: _StubProvider(pid, f"{pid}-x")


class _StubImportlib:
    """Import shim: one module object per name, built by ``factory``."""

    def __init__(self, factory) -> None:
        self._factory = factory
        self.calls: list[str] = []

    def import_module(self, name: str):
        self.calls.append(name)
        return _StubModule(name) if self._factory(name) is None else self._factory(name)


@pytest.fixture
def isolated_registry(monkeypatch):
    """Give each test its own registry state (and an empty module list).

    Rebinding ``_REGISTRY`` here (rather than clearing it) is safe precisely
    because monkeypatch restores it, and it keeps the production in-place
    ``update()`` contract under test.
    """
    monkeypatch.setattr(registry, "_BUILTIN_MODULES", ())
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_LOADED", False)
    return registry


def test_concurrent_reader_never_sees_a_partial_registry(isolated_registry,
                                                         monkeypatch):
    """A reader during a slow load blocks; it never 404s on a valid model."""
    importlib = _StubImportlib(lambda _name: None)
    loading = threading.Event()      # the loader is inside the lock
    release = threading.Event()      # lets the imports finish
    entered = threading.Event()      # the reader is about to call resolve_model

    def slow_import(_name: str):
        loading.set()
        assert release.wait(30), "test bug: the loader was never released"
        return None

    monkeypatch.setattr(isolated_registry, "_BUILTIN_MODULES", ("ds", "qw", "oc"))
    monkeypatch.setattr(isolated_registry, "importlib",
                        _StubImportlib(lambda name: slow_import(name)))

    loader = threading.Thread(target=isolated_registry.registered)
    loader.start()
    assert loading.wait(30), "the loader never reached the imports"

    # Nothing is published yet. Pre-fix, _LOADED was already True and _REGISTRY
    # was already growing while these imports ran.
    assert isolated_registry._REGISTRY == {}, (
        "a provider became visible before the load finished")
    assert isolated_registry._LOADED is False

    seen: dict = {}

    def reader() -> None:
        entered.set()
        seen["resolved"] = isolated_registry.resolve_model("qw-x")
        seen["ids"] = sorted(isolated_registry.registered())

    rt = threading.Thread(target=reader)
    rt.start()
    assert entered.wait(30)

    # Deterministic, not sleep-based: release is still unset, so the loader
    # still holds _LOAD_LOCK and the reader CANNOT have finished.
    assert "resolved" not in seen, "the reader passed a load in progress"
    assert rt.is_alive()

    release.set()
    loader.join(30)
    rt.join(30)
    assert not loader.is_alive() and not rt.is_alive(), "a thread hung"

    resolved = seen["resolved"]
    assert resolved is not None, (
        "resolve_model returned None mid-load -> HTTP 404 for a valid model")
    assert resolved[0].id == "qw"
    assert seen["ids"] == ["ds", "oc", "qw"], "the reader saw a partial load"
    assert isolated_registry._LOADED is True
    assert importlib.calls == [], "the stand-in importlib was never installed"


def test_failed_import_is_skipped_without_taking_the_others_down(isolated_registry,
                                                                monkeypatch):
    """One broken module costs only that provider, and logs once."""
    def flaky(name: str):
        if name == "boom":
            raise ImportError("permanently broken")
        return None

    monkeypatch.setattr(isolated_registry, "_BUILTIN_MODULES", ("ds", "boom"))
    monkeypatch.setattr(isolated_registry, "importlib",
                        _StubImportlib(flaky))

    loaded = isolated_registry.registered()
    assert "ds" in loaded, "one bad module took the good providers with it"
    assert "boom" not in loaded
    # The load latched: no retry, so no second warning and no re-import per
    # request. Every built-in module is one line in _BUILTIN_MODULES away.
    assert isolated_registry._LOADED is True
    assert sorted(isolated_registry.registered()) == ["ds"]


def test_failed_import_warns_exactly_once_per_process(isolated_registry,
                                                      monkeypatch, caplog):
    """The latch is the anti-flood: one broken module, one warning, ever."""
    def always_bad(_name: str):
        raise ImportError("permanently broken")

    monkeypatch.setattr(isolated_registry, "_BUILTIN_MODULES", ("ds",))
    monkeypatch.setattr(isolated_registry, "importlib",
                        _StubImportlib(always_bad))

    with caplog.at_level(logging.DEBUG, logger="deepseaport.providers"):
        isolated_registry.registered()
        isolated_registry.registered()
        isolated_registry.registered()

    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert sum("failed to load" in m for m in warnings) == 1, caplog.text


def test_all_providers_failing_still_reports_the_real_problem(isolated_registry,
                                                              monkeypatch):
    def always_bad(_name: str):
        raise ImportError("nope")

    monkeypatch.setattr(isolated_registry, "_BUILTIN_MODULES", ("ds", "qw"))
    monkeypatch.setattr(isolated_registry, "importlib",
                        _StubImportlib(always_bad))

    # never raises out of the loader...
    assert isolated_registry.registered() == {}
    assert isolated_registry._REGISTRY == {}
    # ...and default() still reports the real problem rather than returning None.
    with pytest.raises(RuntimeError, match="no providers registered"):
        isolated_registry.default()


def test_successful_load_is_published_once(isolated_registry, monkeypatch):
    importlib = _StubImportlib(lambda _name: None)
    monkeypatch.setattr(isolated_registry, "_BUILTIN_MODULES", ("ds", "qw"))
    monkeypatch.setattr(isolated_registry, "importlib", importlib)

    assert sorted(isolated_registry.registered()) == ["ds", "qw"]
    assert isolated_registry._LOADED is True
    assert importlib.calls == ["ds", "qw"]

    # Loaded is a latch: a second call imports nothing.
    assert sorted(isolated_registry.registered()) == ["ds", "qw"]
    assert importlib.calls == ["ds", "qw"]


def test_registered_entries_survive_the_load(isolated_registry, monkeypatch):
    """The load updates _REGISTRY in place: it must not drop entries added
    before it ran (tests and embedders write into _REGISTRY directly)."""
    stub = _StubProvider("fakep", "fakep-x")
    isolated_registry._REGISTRY[isolated_registry._key(stub)] = stub
    monkeypatch.setattr(isolated_registry, "_BUILTIN_MODULES", ("ds",))
    monkeypatch.setattr(isolated_registry, "importlib",
                        _StubImportlib(lambda _name: None))

    assert sorted(isolated_registry.registered()) == ["ds", "fakep"]


def test_passthrough_map_has_no_stray_keys_for_a_bare_request():
    """Guard against the accounting leaking params that ARE honored."""
    unsupported = _extract_passthrough(_body())["unsupported"]
    assert unsupported == {}


# --- R6: resolve_model is guarded and order-INDEPENDENT -----------------------


class _SlashCatalogue(Provider):
    """A provider whose real catalogue ids contain a slash."""

    id = "slashp"
    display_name = "slashp"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def models(self) -> dict[str, ModelSpec]:
        return {"vendor/shared-model": ModelSpec(id="vendor/shared-model")}


class _BoomSpec(Provider):
    """A provider whose model_spec raises, as a half-broken one can."""

    id = "boom"

    def models(self) -> dict[str, ModelSpec]:
        return {}

    def model_spec(self, model):
        raise RuntimeError("model_spec exploded")


def test_resolve_model_skips_a_provider_whose_model_spec_raises():
    """Its two siblings already guard this exact call; it did not."""
    registry._REGISTRY["boom"] = _BoomSpec()
    try:
        found = registry.resolve_model("deepseek-flash", provider_ids=["boom", "deepseek"])
    finally:
        registry._REGISTRY.pop("boom", None)

    assert found is not None, "one broken provider took the resolution down"
    assert found[0].id == "deepseek"


def test_a_real_catalogue_beats_the_opencode_passthrough_in_either_order():
    """opencode's model_spec returns a synthetic spec for ANY id with a slash.

    Left as the first entry in ``providers``, that swallowed every slash id --
    including models a real upstream publishes -- so the same config resolved
    differently depending on the order its operator typed. The registry now
    asks whether a provider actually LISTS the model before honouring its
    answer, so the order cannot matter.
    """
    provider = _SlashCatalogue()
    registry._REGISTRY[registry._key(provider)] = provider
    try:
        oc_first = registry.resolve_model("vendor/shared-model",
                                          provider_ids=["opencode", "slashp"])
        oc_last = registry.resolve_model("vendor/shared-model",
                                         provider_ids=["slashp", "opencode"])
    finally:
        registry._REGISTRY.pop("slashp", None)

    assert oc_first is not None and oc_first[0].id == "slashp", (
        "opencode's slash passthrough outranked the provider that lists the model")
    assert oc_first[1].id == oc_last[1].id
    assert oc_first[0].id == oc_last[0].id


def test_the_passthrough_still_serves_a_model_no_provider_lists():
    """The guard demotes synthesised specs, it does not delete the feature."""
    provider = _SlashCatalogue()
    registry._REGISTRY[registry._key(provider)] = provider
    try:
        found = registry.resolve_model("anthropic/some-other-model",
                                       provider_ids=["opencode", "slashp"])
    finally:
        registry._REGISTRY.pop("slashp", None)

    assert found is not None
    assert found[0].id == "opencode"
    assert found[1].id == "anthropic/some-other-model"


def test_resolve_model_is_stable_for_every_permutation_of_the_active_set():
    """The strongest form: two orderings of the SAME set, one answer."""
    provider = _SlashCatalogue()
    registry._REGISTRY[registry._key(provider)] = provider
    ids = ["opencode", "slashp", "deepseek", "qwen"]
    try:
        forward = {m: registry.resolve_model(m, provider_ids=ids) for m in
                   ("vendor/shared-model", "deepseek-flash", "anthropic/other")}
        reverse = {m: registry.resolve_model(m, provider_ids=list(reversed(ids)))
                   for m in forward}
    finally:
        registry._REGISTRY.pop("slashp", None)

    for model, found in forward.items():
        assert found is not None, model
        assert forward[model][0].id == reverse[model][0].id, (
            f"{model}: order decided the owner "
            f"({forward[model][0].id} vs {reverse[model][0].id})")


# --- R7: the running server reports its own version ---------------------------


def test_both_public_endpoints_report_the_runtime_version():
    """``openapi_url`` is None, so before this a running server had NO version
    surface at all: a user genuinely could not ask what they were talking to.
    One source (``deepseaport.__version__``), read from both endpoints.
    """
    import deepseaport

    client = TestClient(create_app(_settings()))
    health = client.get("/health").json()
    models = client.get("/v1/models").json()

    assert health["version"] == deepseaport.__version__
    assert models["version"] == deepseaport.__version__
    # A source checkout reads pyproject; "0.0.0" is the frozen-exe fallback.
    assert deepseaport.__version__ not in ("", "0.0.0")


def test_the_version_endpoints_keep_their_existing_shape():
    """A new field must not disturb what the pollers and clients read."""
    client = TestClient(create_app(_settings()))
    health = client.get("/health").json()
    models = client.get("/v1/models").json()

    assert health["ok"] is True and "ready" in health and "waf" in health
    assert models["object"] == "list"
    assert isinstance(models["data"], list) and models["data"]
