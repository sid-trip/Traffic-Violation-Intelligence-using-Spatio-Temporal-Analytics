from datetime import datetime

import cv2
import numpy as np


LOCATION_PROFILES = [
    {
        "name": "KR Market",
        "aliases": ["kr market", "kalasipalyam", "mothinagar"],
        "zone_type": "active_carriageway",
        "violation_type": "Active Carriageway Obstruction",
        "restricted_hours": None,
    },
    {
        "name": "Safina Plaza",
        "aliases": ["safina plaza", "tasker town", "main guard cross", "dispensary road"],
        "zone_type": "no_parking_zone",
        "violation_type": "Illegal Parking",
        "restricted_hours": [("08:00", "23:00")],
    },
    {
        "name": "Elite Junction",
        "aliases": ["elite junction", "kempe gowda circle", "gandhi nagar"],
        "zone_type": "no_parking_zone",
        "violation_type": "Illegal Parking",
        "restricted_hours": [("08:00", "22:00")],
    },
    {
        "name": "Sagar Theatre Junction",
        "aliases": ["sagar theatre", "hospital road", "balepet", "chickpete"],
        "zone_type": "no_parking_zone",
        "violation_type": "Illegal Parking",
        "restricted_hours": [("08:00", "22:00")],
    },
    {
        "name": "Hosahalli Metro Station",
        "aliases": ["hosahalli metro", "chord road", "manuvana", "vijaya nagar"],
        "zone_type": "no_parking_zone",
        "violation_type": "Illegal Parking",
        "restricted_hours": [("07:00", "22:00")],
    },
    {
        "name": "Central Street Junction",
        "aliases": ["central street", "lady curzon road", "chandni chowk road", "shivaji nagar"],
        "zone_type": "active_carriageway",
        "violation_type": "Active Carriageway Obstruction",
        "restricted_hours": None,
    },
    {
        "name": "Subbanna Junction",
        "aliases": ["subbanna junction", "r subanna circle", "seshadri road"],
        "zone_type": "no_parking_zone",
        "violation_type": "Illegal Parking",
        "restricted_hours": [("08:00", "22:00")],
    },
]

FALLBACK_KEYWORD_RULES = [
    {
        "keywords": ["footpath", "sidewalk", "pedestrian"],
        "zone_type": "footpath",
        "violation_type": "Parking on Footpath",
        "restricted_hours": None,
    },
    {
        "keywords": ["junction", "signal", "intersection", "main road"],
        "zone_type": "active_carriageway",
        "violation_type": "Active Carriageway Obstruction",
        "restricted_hours": None,
    },
    {
        "keywords": ["school", "hospital", "market", "metro"],
        "zone_type": "no_parking_zone",
        "violation_type": "Illegal Parking",
        "restricted_hours": [("08:00", "22:00")],
    },
]


def _time_to_minutes(time_text):
    parsed = datetime.strptime(time_text, "%H:%M")
    return parsed.hour * 60 + parsed.minute


def _is_time_in_window(current_time, start_time, end_time):
    current_minutes = _time_to_minutes(current_time)
    start_minutes = _time_to_minutes(start_time)
    end_minutes = _time_to_minutes(end_time)

    if start_minutes <= end_minutes:
        return start_minutes <= current_minutes <= end_minutes

    return current_minutes >= start_minutes or current_minutes <= end_minutes


def _build_zone_context(rule, current_time, source, matched_on):
    windows = rule["restricted_hours"]
    if not windows:
        return {
            "zone_type": rule["zone_type"],
            "violation_type": rule["violation_type"],
            "rule_active": True,
            "rule_window": "Always Active",
            "rule_source": source,
            "matched_on": matched_on,
        }

    active_window = None
    for start_time, end_time in windows:
        if _is_time_in_window(current_time, start_time, end_time):
            active_window = f"{start_time}-{end_time}"
            break

    return {
        "zone_type": rule["zone_type"],
        "violation_type": rule["violation_type"],
        "rule_active": active_window is not None,
        "rule_window": active_window or f"{windows[0][0]}-{windows[0][1]}",
        "rule_source": source,
        "matched_on": matched_on,
    }


def _match_location_profile(location_text):
    best_profile = None
    best_score = 0

    for profile in LOCATION_PROFILES:
        score = sum(1 for alias in profile["aliases"] if alias in location_text)
        if score > best_score:
            best_profile = profile
            best_score = score

    if best_profile and best_score > 0:
        return best_profile

    return None


def resolve_zone_context(location_name, current_time):
    location_text = str(location_name).lower()

    matched_profile = _match_location_profile(location_text)
    if matched_profile:
        return _build_zone_context(
            matched_profile,
            current_time,
            source="location_profile",
            matched_on=matched_profile["name"],
        )

    for rule in FALLBACK_KEYWORD_RULES:
        matched_keyword = next(
            (keyword for keyword in rule["keywords"] if keyword in location_text),
            None,
        )
        if matched_keyword:
            return _build_zone_context(
                rule,
                current_time,
                source="keyword_fallback",
                matched_on=matched_keyword,
            )

    return _build_zone_context(
        {
            "zone_type": "restricted_zone",
            "violation_type": "Restricted Zone Violation",
            "restricted_hours": [("08:00", "20:00")],
        },
        current_time,
        source="default",
        matched_on="default_restricted_zone",
    )


class ParkingRuleEngine:
    def __init__(
        self,
        restricted_polygon,
        location_name,
        current_time,
        overlap_threshold=0.10,
    ):
        self.restricted_polygon = np.array(restricted_polygon, dtype=np.int32)
        if self.restricted_polygon.ndim == 2:
            self.restricted_polygon = self.restricted_polygon.reshape((-1, 1, 2))

        self.overlap_threshold = overlap_threshold
        self.zone_context = resolve_zone_context(location_name, current_time)

    def _bbox_overlap_ratio(self, bbox, image_shape):
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox

        # Use only the bottom portion of the bounding box (bottom 20%) to represent the contact area on the ground.
        # This prevents perspective-based false positives for tall vehicles.
        box_h = y2 - y1
        y1_contact = int(y2 - 0.20 * box_h)

        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        zone_mask = np.zeros((h, w), dtype=np.uint8)

        cv2.rectangle(bbox_mask, (x1, y1_contact), (x2, y2), 255, -1)
        cv2.fillPoly(zone_mask, [self.restricted_polygon], 255)

        intersection = cv2.bitwise_and(bbox_mask, zone_mask)

        bbox_area = np.count_nonzero(bbox_mask)
        if bbox_area == 0:
            return 0.0

        overlap_area = np.count_nonzero(intersection)
        return overlap_area / bbox_area

    def _calculate_violation_confidence(self, detection_confidence, overlap_ratio):
        detection_score = max(0.0, min(1.0, detection_confidence / 100.0))
        overlap_score = max(
            0.0,
            min(1.0, overlap_ratio / max(self.overlap_threshold * 2.0, 0.01)),
        )
        rule_score = 1.0 if self.zone_context["rule_active"] else 0.35
        source_score = 1.0 if self.zone_context["rule_source"] == "location_profile" else 0.80

        blended_score = (
            (0.40 * detection_score)
            + (0.35 * overlap_score)
            + (0.15 * rule_score)
            + (0.10 * source_score)
        )
        return round(blended_score * 100, 2)

    def _build_reason(self, overlap_ratio):
        base_reason = (
            f"Vehicle overlaps the {self.zone_context['zone_type']} zone by "
            f"{overlap_ratio:.2f}."
        )

        source_reason = (
            f" Zone assumption came from {self.zone_context['rule_source']}: "
            f"{self.zone_context['matched_on']}."
        )

        if self.zone_context["rule_active"]:
            return (
                f"{base_reason} Restriction is active during "
                f"{self.zone_context['rule_window']}.{source_reason}"
            )

        return (
            f"{base_reason} Restriction window is "
            f"{self.zone_context['rule_window']}, so this is marked as low-confidence."
            f"{source_reason}"
        )

    def evaluate(self, detections, image_shape):
        violations = []

        for det in detections:
            if det["confidence"] < 35:
                continue

            bbox = det["bbox"]
            overlap_ratio = self._bbox_overlap_ratio(bbox, image_shape)

            if overlap_ratio < self.overlap_threshold:
                continue

            violation_confidence = self._calculate_violation_confidence(
                det["confidence"],
                overlap_ratio,
            )

            violations.append(
                {
                    "vehicle_type": det["vehicle_type"],
                    "detection_confidence": det["confidence"],
                    "violation_type": self.zone_context["violation_type"],
                    "violation_confidence": violation_confidence,
                    "reason": self._build_reason(overlap_ratio),
                    "bbox": bbox,
                    "overlap_ratio": round(overlap_ratio, 3),
                    "zone_type": self.zone_context["zone_type"],
                    "rule_active": self.zone_context["rule_active"],
                    "rule_window": self.zone_context["rule_window"],
                    "rule_source": self.zone_context["rule_source"],
                    "matched_on": self.zone_context["matched_on"],
                }
            )

        return violations
