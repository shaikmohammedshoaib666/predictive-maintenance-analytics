"""19-stage industrial data-quality checks (ported from Analytics Forge v2)."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def find_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for n in names:
        for k, real in lower.items():
            if n.lower() in k:
                return real
    return None


def _zscore_iqr_flags(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number])
    z_hits, iqr_hits = 0, 0
    details: list[str] = []
    for c in num.columns:
        s = num[c].dropna()
        if len(s) < 5:
            continue
        z = (s - s.mean()) / (s.std() + 1e-9)
        zc = int((z.abs() > 3).sum())
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        ic = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr else 0
        z_hits += zc
        iqr_hits += ic
        if zc or ic:
            details.append(f"{c}:z={zc},iqr={ic}")
    return {"z_hits": z_hits, "iqr_hits": iqr_hits, "details": details[:8]}


def _isolation_forest_flags(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 15:
        return {"ok": False, "reason": "need >=2 numeric cols & 15 rows"}
    iso = IsolationForest(contamination=0.08, random_state=42)
    labels = iso.fit_predict(num.values)
    n = int((labels == -1).sum())
    return {"ok": True, "anomalies": n, "rate_pct": round(100.0 * n / len(num), 2)}


def _dbscan_noise(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 20:
        return {"ok": False, "reason": "need more numeric rows"}
    X = StandardScaler().fit_transform(num.values[: min(2000, len(num))])
    labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)
    noise = int((labels == -1).sum())
    return {
        "ok": True,
        "noise_points": noise,
        "clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
    }


def _kmeans_clean_proxy(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 20:
        return {"ok": False}
    X = StandardScaler().fit_transform(num.values[: min(2000, len(num))])
    km = KMeans(n_clusters=min(3, max(2, len(X) // 5)), random_state=42, n_init=10)
    labels = km.fit_predict(X)
    dists = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    far = int((dists > dists.mean() + 2 * dists.std()).sum())
    return {"ok": True, "far_from_cluster": far}


def _rolling_impossible_jumps(df: pd.DataFrame) -> dict[str, Any]:
    flags: list[str] = []
    total = 0
    for c in df.select_dtypes(include=[np.number]).columns:
        cl = str(c).lower()
        if not any(h in cl for h in ("temp", "vib", "pressure", "speed", "current", "rpm")):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        delta = s.diff().abs()
        thr = (
            50
            if "temp" in cl
            else (5 if "vib" in cl else (30 if "pressure" in cl else float(s.std() or 1) * 4))
        )
        n = int((delta > thr).fillna(False).sum())
        if n:
            flags.append(f"{c}:{n} jumps>{thr}")
            total += n
    return {"ok": True, "impossible_jumps": flags[:10], "count": total}


def _lag_correlation_break(df: pd.DataFrame) -> dict[str, Any]:
    t = find_col(df, "temperature", "temp")
    p = find_col(df, "pressure")
    v = find_col(df, "vibration", "vib")
    pairs = []
    for a, b in [(t, p), (t, v), (p, v)]:
        if a and b:
            corr = pd.to_numeric(df[a], errors="coerce").corr(pd.to_numeric(df[b], errors="coerce"))
            pairs.append({"pair": f"{a}|{b}", "corr": None if pd.isna(corr) else round(float(corr), 3)})
    broken = [x for x in pairs if x["corr"] is not None and abs(x["corr"]) < 0.05]
    return {"pairs": pairs, "dead_correlations": broken}


def _domain_opc_rules(df: pd.DataFrame) -> list[str]:
    flags: list[str] = []
    t = find_col(df, "temperature", "temp")
    v = find_col(df, "vibration", "vib")
    p = find_col(df, "pressure")
    r = find_col(df, "rul", "failure_within_days")
    fcol = find_col(df, "failure", "fault")
    if t and v:
        tt = pd.to_numeric(df[t], errors="coerce")
        vv = pd.to_numeric(df[v], errors="coerce")
        stuck = int(((tt > 150) & (vv < 0.1)).fillna(False).sum())
        if stuck:
            flags.append(f"Sensor stuck pattern: {stuck} rows (temp>150 & vib<0.1)")
    if p:
        speed = find_col(df, "speed", "flow", "load", "rpm")
        if speed:
            pp = pd.to_numeric(df[p], errors="coerce")
            ss = pd.to_numeric(df[speed], errors="coerce")
            n = int(((pp.diff() < -10) & (ss.diff().abs() < 0.5)).fillna(False).sum())
            if n:
                flags.append(f"Leak/sensor fault suspect: {n} rows")
    if r:
        rr = pd.to_numeric(df[r], errors="coerce")
        d = rr.diff().dropna()
        if len(d) and d.gt(0).mean() > 0.6:
            flags.append("RUL trend unusual: RUL increases over time often")
    if fcol and v:
        ff = pd.to_numeric(df[fcol], errors="coerce").fillna(0)
        vv = pd.to_numeric(df[v], errors="coerce")
        missed = int(((ff == 0) & (vv > vv.mean() + 3 * (vv.std() or 1))).fillna(False).sum())
        if missed:
            flags.append(f"Possible missed failures: {missed} rows")
    return flags


def _run_great_expectations(df: pd.DataFrame) -> dict[str, Any]:
    results = []
    for col in df.columns:
        null_pct = float(df[col].isna().mean())
        results.append(
            {
                "expectation": "expect_column_values_to_not_be_null",
                "column": col,
                "success": null_pct < 0.2,
                "detail": f"null_pct={null_pct:.3f}",
            }
        )
    t = find_col(df, "temperature", "temp")
    if t:
        s = pd.to_numeric(df[t], errors="coerce")
        ok = bool(((s.dropna() >= 0) & (s.dropna() <= 200)).all()) if s.notna().any() else False
        results.append(
            {
                "expectation": "expect_column_values_to_be_between",
                "column": t,
                "success": ok,
                "detail": "temp in [0,200]",
            }
        )
    r = find_col(df, "rul", "failure_within_days")
    if r:
        s = pd.to_numeric(df[r], errors="coerce").dropna()
        success = bool(s.diff().dropna().le(0).mean() >= 0.5) if len(s) > 3 else True
        results.append(
            {
                "expectation": "expect_rul_mostly_non_increasing",
                "column": r,
                "success": success,
                "detail": "RUL mostly non-increasing",
            }
        )
    ts = find_col(df, "timestamp", "time", "datetime", "date")
    mid = find_col(df, "machine_id", "machine", "asset_id")
    if ts and mid:
        dup = int(df.duplicated([ts, mid]).sum())
        results.append(
            {
                "expectation": "expect_compound_columns_to_be_unique",
                "column": f"{ts}+{mid}",
                "success": dup == 0,
                "detail": f"dup_keys={dup}",
            }
        )
    results.append(
        {
            "expectation": "expect_table_row_count_to_be_between",
            "column": "*",
            "success": 1 <= len(df) <= 5_000_000,
            "detail": f"rows={len(df)}",
        }
    )
    ge_available = False
    try:
        import great_expectations as gx  # noqa: F401

        ge_available = True
    except Exception:
        ge_available = False
    passed = sum(1 for r in results if r["success"])
    return {
        "engine": "great_expectations",
        "available": ge_available,
        "passed": passed,
        "total": len(results),
        "results": results[:40],
        "ok": True,
    }


def _run_ydata(df: pd.DataFrame) -> dict[str, Any]:
    high_card = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun > max(50, int(0.5 * len(df))):
            high_card.append(c)
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(df.head(min(400, len(df))), minimal=True, progress_bar=False)
        desc = profile.get_description()
        return {
            "engine": "ydata-profiling",
            "ok": True,
            "variables": len(desc.get("variables", {})),
            "alerts": len(desc.get("alerts", [])),
            "high_cardinality": high_card[:8],
        }
    except Exception as exc:
        return {
            "engine": "ydata-profiling",
            "ok": False,
            "error": str(exc),
            "high_cardinality": high_card[:8],
        }


def _run_cleanlab(df: pd.DataFrame) -> dict[str, Any]:
    fcol = find_col(df, "failure", "fault", "label", "churn", "default")
    num = df.select_dtypes(include=[np.number])
    out: dict[str, Any] = {"engine": "cleanlab"}
    try:
        from cleanlab import Datalab

        work = num.dropna()
        if work.shape[1] >= 2 and len(work) >= 15:
            lab = Datalab(data=work.reset_index(drop=True))
            lab.find_issues(features=work.values)
            issues = lab.get_issues()
            n_out = int(issues["is_outlier_issue"].sum()) if "is_outlier_issue" in issues.columns else 0
            out.update({"ok": True, "outlier_issues": n_out})
        else:
            out.update({"ok": True, "skipped": "numeric too small"})
    except Exception as exc:
        out.update({"ok": False, "error": str(exc)})
    dirty = []
    vib = find_col(df, "vibration", "vib")
    if fcol and vib:
        v = pd.to_numeric(df[vib], errors="coerce")
        f = pd.to_numeric(df[fcol], errors="coerce").fillna(0)
        if v.notna().any():
            false_pos = int(((f == 1) & (v < v.quantile(0.2))).sum())
            false_neg = int(((f == 0) & (v > v.quantile(0.95))).sum())
            if false_pos:
                dirty.append(f"{false_pos} rows failure=1 but low vibration")
            if false_neg:
                dirty.append(f"{false_neg} rows failure=0 but extreme vibration")
    out["dirty_label_flags"] = dirty
    return out


def _pca_drift(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 30:
        return {"ok": False}
    half = len(num) // 2
    X1 = StandardScaler().fit_transform(num.iloc[:half])
    X2 = StandardScaler().fit_transform(num.iloc[half:])
    ncomp = min(3, X1.shape[1])
    r1 = float(PCA(n_components=ncomp).fit(X1).explained_variance_ratio_.sum())
    r2 = float(PCA(n_components=min(3, X2.shape[1])).fit(X2).explained_variance_ratio_.sum())
    drift = abs(r1 - r2)
    return {
        "ok": True,
        "pca_var_early": round(r1, 3),
        "pca_var_late": round(r2, 3),
        "drift_score": round(drift, 3),
        "concept_drift": drift > 0.15,
    }


def _association_rules_proxy(df: pd.DataFrame) -> dict[str, Any]:
    try:
        items: list[set[str]] = []
        cats = [
            c
            for c in df.select_dtypes(include=["object", "category"]).columns
            if df[c].nunique(dropna=True) <= 12
        ][:4]
        nums = list(df.select_dtypes(include=[np.number]).columns)[:6]
        sample = df.tail(min(800, len(df)))
        for _, row in sample.iterrows():
            basket: set[str] = set()
            for c in cats:
                val = row[c]
                if pd.notna(val):
                    basket.add(f"{c}={val}")
            for c in nums:
                s = pd.to_numeric(sample[c], errors="coerce")
                thr = s.quantile(0.9)
                v = pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v) and pd.notna(thr) and v >= thr:
                    basket.add(f"{c}=HIGH")
            if len(basket) >= 2:
                items.append(basket)
        if len(items) < 20:
            return {"ok": True, "skipped": "too few baskets", "suspicious_rules": []}
        pair_counts: Counter[tuple[str, str]] = Counter()
        item_counts: Counter[str] = Counter()
        for basket in items:
            for a in basket:
                item_counts[a] += 1
            bl = sorted(basket)
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    pair_counts[(bl[i], bl[j])] += 1
        n = len(items)
        rules = []
        for (a, b), cnt in pair_counts.most_common(30):
            support = cnt / n
            conf_ab = cnt / max(1, item_counts[a])
            conf_ba = cnt / max(1, item_counts[b])
            if support >= 0.05 and max(conf_ab, conf_ba) >= 0.55:
                rules.append(
                    {
                        "rule": f"{a} => {b}",
                        "support": round(support, 3),
                        "confidence": round(max(conf_ab, conf_ba), 3),
                    }
                )
        suspicious = [r for r in rules if "HIGH" in r["rule"] and r["confidence"] >= 0.7][:8]
        return {
            "ok": True,
            "rules_found": len(rules),
            "top_rules": rules[:5],
            "suspicious_rules": suspicious,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "suspicious_rules": []}


def build_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Run all 19 industrial quality checks and return a structured report."""
    checks: list[dict[str, Any]] = []
    n, m = df.shape
    miss = float(df.isna().sum().sum() / max(1, df.size))
    checks.append(
        {
            "check": "NULLS / Missing%",
            "status": "FAIL" if miss > 0.2 else ("WARN" if miss > 0.05 else "PASS"),
            "detail": f"{miss * 100:.2f}% missing",
        }
    )
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    checks.append(
        {
            "check": "CONSTANT",
            "status": "FAIL" if const_cols else "PASS",
            "detail": str(const_cols[:6]) if const_cols else "none",
        }
    )
    num = df.select_dtypes(include=[np.number])
    zero_ratio = float((num == 0).sum().sum() / max(1, num.size)) if num.size else 0
    checks.append(
        {
            "check": "ZEROS",
            "status": "WARN" if zero_ratio > 0.3 else "PASS",
            "detail": f"{zero_ratio * 100:.1f}% zeros",
        }
    )
    dups = int(df.duplicated().sum())
    checks.append(
        {
            "check": "DUPLICATES",
            "status": "WARN" if dups else "PASS",
            "detail": f"{dups} dup rows",
        }
    )
    zi = _zscore_iqr_flags(df)
    checks.append(
        {
            "check": "Z-SCORE (>3σ)",
            "status": "WARN" if zi["z_hits"] else "PASS",
            "detail": f"{zi['z_hits']} hits; {zi['details'][:3]}",
        }
    )
    checks.append(
        {
            "check": "IQR OUTLIER",
            "status": "WARN" if zi["iqr_hits"] else "PASS",
            "detail": f"{zi['iqr_hits']} hits",
        }
    )
    iso = _isolation_forest_flags(df)
    checks.append(
        {
            "check": "ISOLATION FOREST",
            "status": "WARN" if iso.get("anomalies", 0) else ("PASS" if iso.get("ok") else "INFO"),
            "detail": json.dumps({k: iso[k] for k in iso if k != "ok"})[:160],
        }
    )
    db = _dbscan_noise(df)
    checks.append(
        {
            "check": "DBSCAN NOISE",
            "status": "WARN" if db.get("noise_points", 0) else ("PASS" if db.get("ok") else "INFO"),
            "detail": json.dumps(db)[:160],
        }
    )
    km = _kmeans_clean_proxy(df)
    checks.append(
        {
            "check": "KMEANS DISTANCE",
            "status": "WARN" if km.get("far_from_cluster", 0) else ("PASS" if km.get("ok") else "INFO"),
            "detail": json.dumps(km)[:160],
        }
    )
    jumps = _rolling_impossible_jumps(df)
    checks.append(
        {
            "check": "ROLLING IMPOSSIBLE JUMP",
            "status": "FAIL" if jumps.get("count", 0) else "PASS",
            "detail": str(jumps.get("impossible_jumps") or "none")[:160],
        }
    )
    lag = _lag_correlation_break(df)
    checks.append(
        {
            "check": "LAG / SENSOR CORRELATION",
            "status": "WARN" if lag.get("dead_correlations") else "PASS",
            "detail": json.dumps(lag)[:160],
        }
    )
    ge = _run_great_expectations(df)
    checks.append(
        {
            "check": "GE EXPECTATIONS",
            "status": "PASS" if ge["passed"] == ge["total"] else "WARN",
            "detail": f"{ge['passed']}/{ge['total']} passed; available={ge['available']}",
        }
    )
    yd = _run_ydata(df)
    checks.append(
        {
            "check": "YDATA CARDINALITY",
            "status": "WARN" if yd.get("high_cardinality") else ("PASS" if yd.get("ok") else "INFO"),
            "detail": json.dumps(yd.get("high_cardinality") or yd)[:160],
        }
    )
    cl = _run_cleanlab(df)
    checks.append(
        {
            "check": "CLEANLAB / DIRTY LABELS",
            "status": "WARN" if cl.get("dirty_label_flags") else ("PASS" if cl.get("ok") else "INFO"),
            "detail": str(cl.get("dirty_label_flags") or cl)[:160],
        }
    )
    pca = _pca_drift(df)
    checks.append(
        {
            "check": "PCA / CONCEPT DRIFT",
            "status": "WARN" if pca.get("concept_drift") else ("PASS" if pca.get("ok") else "INFO"),
            "detail": json.dumps(pca)[:160],
        }
    )
    domain_flags = _domain_opc_rules(df)
    checks.append(
        {
            "check": "DOMAIN OPC / PHYSICS RULES",
            "status": "FAIL" if domain_flags else "PASS",
            "detail": "; ".join(domain_flags) if domain_flags else "no domain violations",
        }
    )
    assoc = _association_rules_proxy(df)
    checks.append(
        {
            "check": "ASSOCIATION RULE MINING",
            "status": "WARN" if assoc.get("suspicious_rules") else ("PASS" if assoc.get("ok") else "INFO"),
            "detail": json.dumps(assoc)[:160],
        }
    )
    checks.append(
        {
            "check": "SCHEMA / ROWCOUNT",
            "status": "PASS" if n > 0 else "FAIL",
            "detail": f"{n} rows × {m} cols",
        }
    )
    checks.append(
        {
            "check": "TIMESTAMP PRESENT",
            "status": "PASS" if find_col(df, "timestamp", "date", "time", "datetime") else "WARN",
            "detail": str(find_col(df, "timestamp", "date", "time", "datetime") or "missing"),
        }
    )
    return {
        "checks": checks,
        "ge": ge,
        "ydata": yd,
        "cleanlab": cl,
        "pca": pca,
        "domain_flags": domain_flags,
        "association": assoc,
    }


QUALITY_STAGE_COUNT = 19
