import ast
import re
from collections import Counter
from datetime import datetime

import pandas as pd

from rules import resolve_zone_context


GENERIC_LOCATION_TOKENS = {
    "and",
    "area",
    "bangalore",
    "bengaluru",
    "circle",
    "cross",
    "india",
    "junction",
    "karnataka",
    "main",
    "near",
    "pin",
    "road",
    "signal",
    "station",
    "street",
}

HISTORICAL_LABEL_TO_OUTPUT = {
    "NO PARKING": "No Parking",
    "WRONG PARKING": "Wrong Parking",
    "PARKING IN A MAIN ROAD": "Parking In A Main Road",
    "PARKING ON FOOTPATH": "Parking On Footpath",
    "DOUBLE PARKING": "Double Parking",
    "PARKING NEAR ROAD CROSSING": "Parking Near Road Crossing",
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": "Parking Near Bus Stop/School/Hospital",
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": "Parking Near Traffic Light Or Zebra Cross",
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": "Parking Opposite Another Parked Vehicle",
}

RULE_TO_HISTORICAL_LABELS = {
    "Parking on Footpath": ["PARKING ON FOOTPATH", "WRONG PARKING"],
    "Active Carriageway Obstruction": [
        "PARKING IN A MAIN ROAD",
        "DOUBLE PARKING",
        "WRONG PARKING",
    ],
    "Illegal Parking": [
        "NO PARKING",
        "WRONG PARKING",
        "PARKING NEAR ROAD CROSSING",
        "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC",
    ],
    "Restricted Zone Violation": [
        "WRONG PARKING",
        "NO PARKING",
        "PARKING IN A MAIN ROAD",
    ],
}

INCIDENT_CAUSE_WEIGHTS = {
    "congestion": 18,
    "vehicle_breakdown": 12,
    "accident": 10,
    "construction": 8,
    "water_logging": 6,
    "others": 4,
}


def _tokenize_location(location_name):
    raw_tokens = re.findall(r"[a-z0-9]+", str(location_name).lower())
    return [
        token
        for token in raw_tokens
        if len(token) >= 3 and token not in GENERIC_LOCATION_TOKENS
    ][:6]


def _normalize_violation_labels(raw_value):
    if pd.isna(raw_value):
        return []

    try:
        parsed = ast.literal_eval(str(raw_value))
        if isinstance(parsed, list):
            return [str(label).strip().upper() for label in parsed if str(label).strip()]
    except (ValueError, SyntaxError):
        pass

    normalized = str(raw_value).strip().upper()
    return [normalized] if normalized else []


def _format_historical_label(label):
    return HISTORICAL_LABEL_TO_OUTPUT.get(label, label.replace("_", " ").title())


def _load_violation_dataframe():
    try:
        df = pd.read_csv(
            "historical_violations.csv",
            low_memory=False,
            usecols=[
                "location",
                "junction_name",
                "police_station",
                "vehicle_type",
                "violation_type",
                "created_datetime",
            ],
        )
    except Exception as exc:
        print(f"Historical dataset load failed: {exc}")
        return None

    df["created_datetime"] = pd.to_datetime(
        df["created_datetime"],
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=["created_datetime"]).copy()
    df["hour"] = df["created_datetime"].dt.hour
    df["normalized_labels"] = df["violation_type"].apply(_normalize_violation_labels)
    df["search_text"] = (
        df["location"].fillna("").astype(str).str.lower()
        + " "
        + df["junction_name"].fillna("").astype(str).str.lower()
        + " "
        + df["police_station"].fillna("").astype(str).str.lower()
    )
    return df


def _load_event_dataframe():
    try:
        df = pd.read_csv(
            "event_data_traffic_violations.csv",
            low_memory=False,
            usecols=[
                "address",
                "junction",
                "police_station",
                "event_cause",
                "priority",
                "start_datetime",
            ],
        )
    except Exception as exc:
        print(f"Event dataset load failed: {exc}")
        return None

    df["start_datetime"] = pd.to_datetime(
        df["start_datetime"],
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=["start_datetime"]).copy()
    df["hour"] = df["start_datetime"].dt.hour
    df["search_text"] = (
        df["address"].fillna("").astype(str).str.lower()
        + " "
        + df["junction"].fillna("").astype(str).str.lower()
        + " "
        + df["police_station"].fillna("").astype(str).str.lower()
    )
    return df


VIOLATION_DF = _load_violation_dataframe()
EVENT_DF = _load_event_dataframe()


def _build_location_mask(df, location_name):
    tokens = _tokenize_location(location_name)
    if df is None or not tokens:
        return None, tokens

    token_hits = None
    for token in tokens:
        contains_token = df["search_text"].str.contains(token, regex=False, na=False).astype(int)
        token_hits = contains_token if token_hits is None else token_hits + contains_token

    min_hits = 2 if len(tokens) >= 3 else 1
    return token_hits >= min_hits, tokens


def _extract_local_violation_slice(location_name, current_hour):
    if VIOLATION_DF is None:
        return None

    location_mask, tokens = _build_location_mask(VIOLATION_DF, location_name)
    hour_mask = VIOLATION_DF["hour"] == current_hour

    if location_mask is not None:
        location_hour = VIOLATION_DF[location_mask & hour_mask]
        if len(location_hour) >= 15:
            return {
                "data": location_hour,
                "scope": "location_and_hour",
                "tokens": tokens,
                "location_hour_count": len(location_hour),
                "location_total_count": int(location_mask.sum()),
            }

        location_all = VIOLATION_DF[location_mask]
        if len(location_all) >= 15:
            return {
                "data": location_all,
                "scope": "location_only",
                "tokens": tokens,
                "location_hour_count": len(location_hour),
                "location_total_count": len(location_all),
            }

    hour_only = VIOLATION_DF[hour_mask]
    if len(hour_only) > 0:
        return {
            "data": hour_only,
            "scope": "hour_only",
            "tokens": tokens,
            "location_hour_count": 0,
            "location_total_count": 0,
        }

    return {
        "data": VIOLATION_DF,
        "scope": "global",
        "tokens": tokens,
        "location_hour_count": 0,
        "location_total_count": 0,
    }


def _extract_local_event_slice(location_name, current_hour):
    if EVENT_DF is None:
        return None

    location_mask, _ = _build_location_mask(EVENT_DF, location_name)
    hour_mask = EVENT_DF["hour"] == current_hour

    if location_mask is not None:
        local_events = EVENT_DF[location_mask & hour_mask]
        if len(local_events) >= 5:
            return {
                "data": local_events,
                "scope": "location_and_hour",
            }

        broader_events = EVENT_DF[location_mask]
        if len(broader_events) >= 5:
            return {
                "data": broader_events,
                "scope": "location_only",
            }

    hour_events = EVENT_DF[hour_mask]
    if len(hour_events) > 0:
        return {
            "data": hour_events,
            "scope": "hour_only",
        }

    return None


def _count_historical_labels(local_df):
    counts = Counter()
    if local_df is None:
        return counts

    for labels in local_df["normalized_labels"]:
        counts.update(labels)

    return counts


def _derive_incident_signal(event_slice):
    if event_slice is None:
        return {
            "incident_count": 0,
            "incident_scope": "none",
            "top_event_causes": [],
            "incident_risk_bonus": 0,
        }

    cause_counts = Counter(
        event_slice["data"]["event_cause"].fillna("unknown").astype(str).str.lower()
    )
    top_causes = [
        {"event_cause": cause, "count": count}
        for cause, count in cause_counts.most_common(3)
    ]

    incident_bonus = 0
    for cause, count in cause_counts.items():
        incident_bonus += min(count, 3) * INCIDENT_CAUSE_WEIGHTS.get(cause, 3)

    return {
        "incident_count": int(len(event_slice["data"])),
        "incident_scope": event_slice["scope"],
        "top_event_causes": top_causes,
        "incident_risk_bonus": min(25, incident_bonus),
    }


def _build_recommendation(risk_score):
    if risk_score >= 85:
        return "CRITICAL: Recommend tow-truck dispatch and on-ground intervention"
    if risk_score >= 65:
        return "WARNING: Alert nearest patrol and monitor queue buildup"
    if risk_score >= 45:
        return "ELEVATED: Monitor closely and keep response unit on standby"
    return "NORMAL: Monitor via CCTV and log recurring patterns"


def _choose_historical_label(rule_violation_type, label_counts):
    preferred_labels = RULE_TO_HISTORICAL_LABELS.get(rule_violation_type, [])
    for label in preferred_labels:
        if label_counts.get(label, 0) > 0:
            return label

    if label_counts:
        return label_counts.most_common(1)[0][0]

    return None


def _enrich_violations(violations, label_counts, sample_count):
    if not violations:
        return []

    total_labels = sum(label_counts.values()) or 1
    enriched = []

    for violation in violations:
        preferred_labels = RULE_TO_HISTORICAL_LABELS.get(
            violation["violation_type"],
            [],
        )
        support_count = sum(label_counts.get(label, 0) for label in preferred_labels)
        support_ratio = support_count / total_labels if total_labels else 0.0
        historical_weight = min(0.30, sample_count / 200.0)
        historical_confidence = round(
            ((0.60 * support_ratio) + (0.40 * min(1.0, sample_count / 50.0))) * 100,
            2,
        )

        predicted_label = _choose_historical_label(
            violation["violation_type"],
            label_counts,
        )
        predicted_violation_type = (
            _format_historical_label(predicted_label)
            if predicted_label
            else violation["violation_type"]
        )

        final_confidence = violation["violation_confidence"]
        if support_count > 0:
            final_confidence = round(
                (violation["violation_confidence"] * (1 - historical_weight))
                + (historical_confidence * historical_weight),
                2,
            )

        enriched_violation = dict(violation)
        enriched_violation["rule_based_violation_type"] = violation["violation_type"]
        enriched_violation["violation_type"] = predicted_violation_type
        enriched_violation["violation_confidence"] = final_confidence
        enriched_violation["historical_support_count"] = support_count
        enriched_violation["historical_support_ratio"] = round(support_ratio, 3)
        enriched_violation["historical_confidence"] = historical_confidence

        enriched.append(enriched_violation)

    return enriched



def _synthesize_risk_from_context(location_name, current_time, violations=None):
    """Fallback risk score when historical CSVs are unavailable.

    Uses the current zone type, time-of-day, and the number of violations
    already inferred by the rule engine so the score still carries meaning.
    """
    violations = violations or []
    zone_context = resolve_zone_context(location_name, current_time)
    current_hour = datetime.strptime(current_time, "%H:%M").hour

    zone_base = {
        "active_carriageway": 78,
        "no_parking_zone": 66,
        "footpath": 72,
        "restricted_zone": 64,
    }.get(zone_context["zone_type"], 60)

    # Peak traffic windows are higher risk because dwell-time violations are
    # more operationally sensitive there.
    is_peak_hour = 7 <= current_hour <= 10 or 17 <= current_hour <= 21
    peak_bonus = 10 if is_peak_hour else 0

    # Night-time parking can be less congested but still operationally risky.
    if current_hour >= 22 or current_hour <= 5:
        time_bonus = 4
    else:
        time_bonus = 0

    violation_bonus = min(18, len(violations) * 6)

    # If the rule window itself is inactive, the risk should stay lower even
    # when the zone type is normally sensitive.
    activation_penalty = 0 if zone_context["rule_active"] else -8

    risk_score = zone_base + peak_bonus + time_bonus + violation_bonus + activation_penalty
    risk_score = max(20, min(95, risk_score))

    if risk_score >= 85:
        recommendation = "CRITICAL: Recommend tow-truck dispatch and on-ground intervention"
    elif risk_score >= 65:
        recommendation = "WARNING: Alert nearest patrol and monitor queue buildup"
    elif risk_score >= 45:
        recommendation = "ELEVATED: Monitor closely and keep response unit on standby"
    else:
        recommendation = "NORMAL: Monitor via CCTV and log recurring patterns"

    return {
        "risk_score": int(risk_score),
        "historical_count": 0,
        "location_match_scope": f"synthetic:{zone_context['zone_type']}",
        "matched_location_tokens": [],
        "top_historical_violation_types": [],
        "predicted_congestion_hotspot": risk_score >= 70,
        "incident_count": len(violations),
        "incident_match_scope": "synthetic",
        "top_event_causes": [],
        "recommendation": recommendation,
        "enriched_violations": violations,
        "zone_context": zone_context,
        "risk_basis": {
            "zone_type": zone_context["zone_type"],
            "is_peak_hour": is_peak_hour,
            "current_hour": current_hour,
            "violation_count": len(violations),
        },
    }


def get_historical_context(location_name, current_time, violations=None):
    if VIOLATION_DF is None:
        return _synthesize_risk_from_context(location_name, current_time, violations)

    current_hour = datetime.strptime(current_time, "%H:%M").hour
    violation_slice = _extract_local_violation_slice(location_name, current_hour)
    event_slice = _extract_local_event_slice(location_name, current_hour)

    local_data = violation_slice["data"]
    historical_count = int(len(local_data))
    label_counts = _count_historical_labels(local_data)
    incident_signal = _derive_incident_signal(event_slice)

    top_violation_types = [
        {"violation_type": _format_historical_label(label), "count": count}
        for label, count in label_counts.most_common(5)
    ]

    base_risk = min(55, violation_slice["location_hour_count"])
    density_bonus = min(20, violation_slice["location_total_count"] // 250)
    risk_score = min(
        95,
        25 + base_risk + density_bonus + incident_signal["incident_risk_bonus"],
    )

    enriched_violations = _enrich_violations(
        violations or [],
        label_counts,
        historical_count,
    )

    return {
        "risk_score": int(risk_score),
        "historical_count": historical_count,
        "location_match_scope": violation_slice["scope"],
        "matched_location_tokens": violation_slice["tokens"],
        "top_historical_violation_types": top_violation_types,
        "predicted_congestion_hotspot": risk_score >= 70,
        "incident_count": incident_signal["incident_count"],
        "incident_match_scope": incident_signal["incident_scope"],
        "top_event_causes": incident_signal["top_event_causes"],
        "recommendation": _build_recommendation(risk_score),
        "enriched_violations": enriched_violations,
    }
