"""Search public Kernel Hub metadata for leads on observed operator pairs."""

from datetime import datetime, timezone
from http.client import HTTPException
import json
import os
import re
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen


CATALOG = "https://huggingface.co/api/kernels"
# Kernel Hub spellings for search, without implying semantic equivalence.
PAIR_TERMS = {
    ("silu", "mul"): ("silu_and_mul", "swiglu"),
    ("gelu", "mul"): ("gelu_and_mul", "geglu"),
    ("mul", "add"): ("mul_add", "multiply_add", "fma"),
}
ACTIVATIONS = {"gelu", "relu", "silu", "tanh", "sigmoid"}
GENERIC_OPS = {"add", "mul", "mm", "addmm", "linear", "mean", "pow"}
ERRORS = (OSError, HTTPException, ValueError)


MANIFEST_ERRORS = ERRORS + (KeyError, TypeError, AttributeError)


def _json(url):
    with urlopen(url, timeout=6) as response:
        return json.load(response)


def _norm(text):
    return re.sub("[^a-z0-9]", "", text.lower())


def _terms(operators):
    left, right = [op.split(".")[1] for op in operators]
    pair = {left + "_" + right, *PAIR_TERMS.get((left, right), ())}
    if left in {"linear", "mm", "addmm"}:
        pair.update({"gemm_" + right, "matmul_" + right})
    component = {left, right} - GENERIC_OPS
    family = {"activation"} if component & ACTIVATIONS else set()
    return {
        "pair": sorted(pair),
        "component": sorted(component),
        "family": sorted(family),
    }


def _match(terms, fields):
    for kind, words in terms.items():
        for word in words:
            for field in fields:
                if _norm(word) in _norm(field):
                    return {"kind": kind, "term": word, "field": field}
    return None


def _symbols(url):
    try:
        data = _json(url)
        return sorted(
            {f["name"] for f in data["functions"] if isinstance(f.get("name"), str)}
        )
    except HTTPError as error:
        if error.code == 404:
            return None
        raise


def search(discovery, rejected=()):
    pairs = sorted({tuple(c["operators"]) for c in discovery["candidates"]})
    device = discovery["device"].split(":")[0]
    driver = "metal" if device == "mps" else device
    result = {
        "schema_version": 1,
        "endpoint": CATALOG,
        "target_driver": driver,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "catalog_size": None,
        "queries": [],
        "layers": layer_leads(discovery),
        "status": "not_needed",
    }
    if not pairs and not result["layers"]:
        return result
    if os.getenv("HF_HUB_OFFLINE", "").upper() in {"1", "TRUE", "ON", "YES"}:
        result["status"] = "offline"
        return result
    # Ask for one bounded public catalog snapshot. Record the limit explicitly
    # so an empty hit list never implies complete coverage as the Hub grows.
    limit = 1000
    result["catalog_limit"] = limit
    try:
        catalog = _json(
            f"{CATALOG}?limit={limit}&expand=sha&expand=tags&expand=supportedDriverFamilies"
        )
        if not isinstance(catalog, list) or any(
            not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in catalog
        ):
            raise ValueError("Unexpected Kernel Hub listing")
    except ERRORS as error:
        result.update(status="unavailable", error=str(error))
        return result
    result.update(
        status="searched",
        catalog_size=len(catalog),
        limit_reached=len(catalog) >= limit,
    )
    version = re.match(r"(\d+)\.(\d+)", discovery["torch"])
    manifests = {}
    for pair in pairs:
        terms = _terms(pair)
        scopes = {
            obs[side]
            for c in discovery["candidates"]
            if tuple(c["operators"]) == pair
            for obs in c["observations"]
            for side in ("producer_scope", "consumer_scope")
        }
        mapped = {}
        for layer in result["layers"]:
            for state, names in layer["sites"].items():
                linked = scopes.intersection(names)
                if linked:
                    mapped.setdefault(layer["repo_id"], []).append(
                        {
                            "symbol": layer["symbol"],
                            "binding": state,
                            "scope_count": len(linked),
                        }
                    )
        hits = []
        for repo in catalog:
            repo_id = repo["id"]
            match = _match(terms, [repo_id.split("/")[-1], *(repo.get("tags") or [])])
            if repo_id in mapped:
                match = {
                    "kind": "mapping",
                    "term": "Transformers declaration",
                    "field": mapped[repo_id][0]["symbol"],
                }
            if not match:
                continue
            drivers = sorted(repo.get("supportedDriverFamilies") or [])
            url = "https://huggingface.co/kernels/" + quote(repo_id, safe="/")
            hit = {
                "repo_id": repo_id,
                "url": url,
                "revision": repo.get("sha"),
                "advertised_drivers": drivers,
                "catalog_match": match,
                "observed_layer_bindings": mapped.get(repo_id, []),
                "compatibility": "unverified",
            }
            hits.append(hit)
        hits.sort(
            key=lambda h: (
                driver not in h["advertised_drivers"],
                list(("mapping", *terms)).index(h["catalog_match"]["kind"]),
                h["repo_id"],
            )
        )
        # Inspect a small number of target builds. Search hits stay available
        # when older or differently packaged kernels omit symbol inventories.
        for hit in hits[:3]:
            if (
                driver != "metal"
                or driver not in hit["advertised_drivers"]
                or not version
                or not hit["revision"]
            ):
                continue
            path = (
                f"build/torch{version[1]}{version[2]}-metal-aarch64-darwin/symbols.json"
            )
            url = f"{hit['url']}/resolve/{quote(hit['revision'], safe='')}/{path}"
            if url not in manifests:
                try:
                    names = _symbols(url)
                    manifests[url] = (
                        "read" if names is not None else "absent",
                        names or [],
                    )
                except MANIFEST_ERRORS:
                    manifests[url] = ("unavailable", [])
            status, names = manifests[url]
            hit["symbols"] = {
                "url": url,
                "status": status,
                "matches": [
                    {"name": name, "kind": match["kind"]}
                    for name in names
                    if (match := _match(terms, [name]))
                ],
            }
        result["queries"].append(
            {"operators": list(pair), "terms": terms, "hits": hits}
        )
    inspect_layers(result, discovery, rejected)
    return result


def summarize_search(snapshot, tasks):
    """Count distinct catalog hits and attach whole-pair leads to matching tasks."""
    driver = snapshot["target_driver"]
    summary = {
        "status": snapshot["status"],
        "target_driver": driver,
        "timestamp": snapshot.get("timestamp"),
    }
    if snapshot["status"] != "searched":
        return summary
    queries = snapshot["queries"]
    repos = set()
    target_repos = set()
    pairs = {}
    for query in queries:
        leads = []
        for hit in query["hits"]:
            repo = hit["repo_id"]
            repos.add(repo)
            if driver not in hit["advertised_drivers"]:
                continue
            target_repos.add(repo)
            symbols = hit.get("symbols", {})
            matches = sorted(
                {s["name"] for s in symbols.get("matches", []) if s["kind"] == "pair"}
            )
            if matches or hit["catalog_match"]["kind"] == "pair":
                leads.append(
                    {
                        "repo_id": repo,
                        "revision": hit.get("revision"),
                        "basis": "published_symbol" if matches else "catalog_metadata",
                        "symbols": matches,
                        "source": symbols.get("url") if matches else hit["url"],
                        "compatibility": "unverified",
                    }
                )
        pairs[tuple(query["operators"])] = sorted(
            leads, key=lambda h: (not h["symbols"], h["repo_id"])
        )
    for task in tasks:
        task["hub_leads"] = pairs.get(tuple(task["operators"]), [])
        if task["hub_leads"]:
            task["next_step"] = (
                "Check the existing Hub implementation's signature, layout and numerics, then benchmark it before writing a replacement."
            )
    return {
        **summary,
        "catalog_size": snapshot["catalog_size"],
        "limit_reached": snapshot["limit_reached"],
        "operator_pairs": len(queries),
        "matching_repositories": len(repos),
        "target_repositories": sorted(target_repos),
    }


def layer_leads(discovery):
    """Summarize actual forward bindings for literal inference-mode mappings."""
    integration = discovery.get("transformers_integrations") or {}
    sites = integration.get("sites", {})
    groups = {}
    for declaration in integration.get("declarations", []):
        mode = declaration["mode"]
        if mode and "Mode.INFERENCE" not in mode:
            continue
        layer, repo, symbol = (
            declaration[k] for k in ("layer", "repo_id", "layer_name")
        )
        key = (
            layer,
            repo,
            symbol,
            declaration.get("version"),
            declaration.get("revision"),
        )
        if key not in groups:
            usage = {"already_bound": [], "original_forward": [], "unknown": []}
            for scope, site in sites.items():
                if site["layer"] == layer:
                    state = site["binding"]
                    if state == "hub_bound":
                        state = (
                            "already_bound"
                            if (site["bound_repo"], site["bound_symbol"])
                            == (repo, symbol)
                            else "unknown"
                        )
                    usage[state].append(scope)
            groups[key] = {
                "layer": layer,
                "repo_id": repo,
                "symbol": symbol,
                "version": key[3],
                "revision": key[4],
                "mappings": [],
                "sites": usage,
                "example_status": "unverified",
            }
        groups[key]["mappings"].append(
            {k: declaration[k] for k in ("device", "mode", "source")}
        )
    return list(groups.values())


def inspect_layers(result, discovery, rejected):
    """Select candidate adapters using versioned upstream source and observed sites."""
    import ast

    device = discovery["device"].split(":")[0]
    version = re.match(r"(\d+)\.(\d+)", discovery["torch"])
    cache = {}
    for lead in result["layers"]:
        repo, symbol = lead["repo_id"], lead["symbol"]
        if not lead["sites"]["original_forward"]:
            lead["example_status"] = "no_observed_original_sites"
            continue
        if device != "mps" or not version or not repo.startswith("kernels-community/"):
            continue
        ref = lead["revision"] or (f"v{lead['version']}" if lead["version"] else None)
        if not ref:
            continue
        key = (repo, ref)
        if key not in cache:
            try:
                info = _json(f"{CATALOG}/{quote(repo)}/revision/{quote(ref, safe='')}")
                if (
                    info["id"] != repo
                    or not re.fullmatch(r"[a-f0-9]{40}", info["sha"])
                    or "metal" not in (info.get("supportedDriverFamilies") or [])
                ):
                    cache[key] = None
                else:
                    url = (
                        f"https://huggingface.co/kernels/{quote(repo)}/resolve/{info['sha']}/"
                        f"build/torch{version[1]}{version[2]}-metal-aarch64-darwin/layers.py"
                    )
                    with urlopen(url, timeout=6) as response:
                        source = ast.parse(response.read(1024 * 1024))
                    classes = {
                        node.name
                        for node in source.body
                        if isinstance(node, ast.ClassDef)
                        and any(
                            isinstance(fn, ast.FunctionDef) and fn.name == "forward"
                            for fn in node.body
                        )
                    }
                    cache[key] = info["sha"], url, classes
            except (
                OSError,
                ValueError,
                HTTPException,
                SyntaxError,
                KeyError,
                TypeError,
            ):
                cache[key] = None
        artifact = cache[key]
        if artifact:
            lead["source_url"] = artifact[1]
            if symbol in artifact[2]:
                lead.update(
                    example_status=(
                        "picked_for_trial"
                        if any(m["device"] == device for m in lead["mappings"])
                        else "requires_device_integration"
                    ),
                    pinned_revision=artifact[0],
                )
            if (repo, artifact[0]) in rejected:
                lead["example_status"] = "previously_rejected"
    # A site can receive one adapter. Prefer an explicit mapping for this device.
    for layer in {lead["layer"] for lead in result["layers"]}:
        choices = [
            lead
            for lead in result["layers"]
            if lead["layer"] == layer and lead["example_status"] == "picked_for_trial"
        ]
        targeted = [
            lead
            for lead in choices
            if any(m["device"] == device for m in lead["mappings"])
        ]
        best = targeted or choices
        for lead in choices:
            if len(best) != 1 or lead is not best[0]:
                lead["example_status"] = "alternative_mapping"
