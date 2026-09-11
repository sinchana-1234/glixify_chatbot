#!/usr/bin/env python3
"""
Health Progress Tool
Fetches daily vitals (glucose, BP, HR, stress, HRV, activity, sleep) for a
patient from the healthProgress endpoint.

The base URL is read from HEALTH_PROGRESS_BASE_URL in .env (loaded into
os.environ by load_dotenv() at app startup) so this tool points at whichever
environment (QA / production) the rest of the backend is configured for,
without needing a code change to switch. No fallback default is used —
if the setting is missing, the app fails loudly at import time rather than
silently querying the wrong environment.

In addition to the text summary returned to the LLM, this tool stashes a
chart-ready payload on `self.last_chart_data` — a side-channel the chat API
route reads after the agent finishes, since a LangChain tool can only return
text to the LLM itself. `last_chart_data` is cleared after each read so a
stale chart never leaks into an unrelated later answer.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Optional
import httpx
from langchain.tools import BaseTool
from dal.database import DatabaseManager

logger = logging.getLogger(__name__)

HEALTH_PROGRESS_BASE = os.getenv("HEALTH_PROGRESS_BASE_URL")
if not HEALTH_PROGRESS_BASE:
    raise RuntimeError(
        "HEALTH_PROGRESS_BASE_URL is not set in .env — HealthProgressTool cannot function without it."
    )


class HealthProgressTool(BaseTool):
    name: str = "get_health_progress"
    description: str = (
        "Get a patient's daily health data over a date range: glucose analytics "
        "(mean, TIR/TAR/TBR %, HbA1c estimate, FBS), blood pressure, heart rate, "
        "stress, HRV, activity (steps/calories), sleep breakdown, medications, and "
        "assigned doctor/DHA. Use for: 'glucose trend', 'BP for patient X', 'sleep "
        "pattern', 'activity summary', 'time in range', 'what medications', 'who is "
        "their doctor', 'graphs for patient X'. PREFER this tool over any single-"
        "value lookup tool whenever the user wants a trend, pattern, or chart across "
        "multiple days — this tool renders a real chart, single-value tools do not. "
        "Requires EITHER patient_id (int) OR patient_name (str) — pass whichever the "
        "user gave you; do NOT guess a patient_id when only a name was given, this "
        "tool resolves the name itself. If multiple patients match the name, the "
        "result will list them — relay that list and ask the user to pick one, do "
        "not choose for them. from_date/to_date (YYYY-MM-DD) are OPTIONAL — if the "
        "user doesn't mention a date range, omit them entirely and the tool will "
        "default to the last 7 days. Pass 'metric' to choose which chart to build: "
        "'glucose' (default, mean glucose trend), 'tir' (time in range breakdown — "
        "above/in-range/below), 'sleep' (deep/light/REM sleep minutes per day), or "
        "'activity' (daily step count). Match 'metric' to what the user actually "
        "asked about — e.g. 'sleep quality' -> metric='sleep', 'time in range' -> "
        "metric='tir', 'activity/steps' -> metric='activity'. The result includes a "
        "'status' field: 'ok' (data found), 'ambiguous_name' (multiple patients "
        "matched — ask which one), 'not_found' (no patient matched the name), "
        "'no_data_in_range' (explicit dates given, none found — tell the user "
        "plainly, do not substitute other dates), 'fallback_most_recent' (no dates "
        "given, none in the last 7 days, showing older data instead), or "
        "'no_data_at_all' (nothing found in 180 days). Always relay the 'notice' "
        "field to the user when present, verbatim or close to it — do not invent "
        "your own wording. IMPORTANT: only report the data relevant to what the "
        "user actually asked about (e.g. only activity/steps if they asked for "
        "activity) — do NOT dump unrelated fields like medications, supplements, "
        "or full glucose readings unless the user specifically asked for those too. "
        "If the requested metric has no data for the period shown (see "
        "'metric_notice' in the result if present), say so plainly and briefly — "
        "do not pad the answer with unrelated data instead. When the result has "
        "'ok' status and more than one data point, a chart is ALREADY being shown "
        "to the user automatically — do NOT also write out a table or list of the "
        "individual daily values in your text reply. Instead give a brief 1-3 "
        "sentence summary (e.g. average, highest, lowest, or a notable trend) and "
        "let the chart carry the detail."
    )

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)
        object.__setattr__(self, 'last_chart_data', None)

    def _fetch(self, patient_id: int, from_date: str, to_date: str, auth_token: str) -> dict:
        """Single raw call to the healthProgress endpoint. Raises on failure."""
        resp = httpx.get(
            f"{HEALTH_PROGRESS_BASE}/{patient_id}/{from_date}/{to_date}",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=15.0
        )
        resp.raise_for_status()
        return resp.json()

    def _build_chart(self, metric: str, glucose_daily: list, data: dict, from_date: str, to_date: str) -> dict:
        """Builds the chart_data payload for the requested metric. Each metric
        has a different natural chart shape, so the frontend switches on
        'type' to pick the right Recharts component."""

        if metric == "tir":
            return {
                "type": "overlayArea",
                "title": f"Time in Range - {from_date} to {to_date}",
                "x_labels": [d["glucoseDate"] for d in glucose_daily],
                "series": [
                    {"name": "Above (%)", "values": [d["tarPercent"] for d in glucose_daily]},
                    {"name": "In Range (%)", "values": [d["tirPercent"] for d in glucose_daily]},
                    {"name": "Below (%)", "values": [d["tbrPercent"] for d in glucose_daily]},
                ],
                "stats": {
                    "avg_tir_pct": round(sum(d["tirPercent"] for d in glucose_daily) / len(glucose_daily), 1),
                    "avg_bp": None,
                    "avg_hba1c": None,
                }
            }

        if metric == "sleep":
            sleep = data.get("sleep") or []
            if not sleep:
                return None
            return {
                "type": "line",
                "title": f"Sleep Quality - {from_date} to {to_date}",
                "x_labels": [d["readingDate"] for d in sleep],
                "series": [
                    {"name": "Light Sleep (hrs)", "values": [round(d.get("lightSleep", 0) / 60, 1) for d in sleep]},
                    {"name": "Deep Sleep (min)", "values": [d.get("deepSleep", 0) for d in sleep]},
                    {"name": "REM Sleep (min)", "values": [d.get("remSleep", 0) for d in sleep]},
                ],
                "stats": {"avg_tir_pct": None, "avg_bp": None, "avg_hba1c": None}
            }

        if metric == "activity":
            activity = data.get("activity") or []
            if not activity:
                return None
            return {
                "type": "line",
                "title": f"Activity - {from_date} to {to_date}",
                "x_labels": [d["readingDate"] for d in activity],
                "series": [
                    {"name": "Steps", "values": [d.get("totalSteps", 0) for d in activity]},
                    {"name": "Calories (kcal)", "values": [d.get("calories", 0) for d in activity]},
                ],
                "dual_axis": True,
                "stats": {"avg_tir_pct": None, "avg_bp": None, "avg_hba1c": None}
            }

        chart_data = {
            "type": "line",
            "title": f"Mean glucose - {from_date} to {to_date}",
            "x_labels": [d["glucoseDate"] for d in glucose_daily],
            "series": [{
                "name": "Mean glucose (mg/dL)",
                "values": [d["meanGlucose"] for d in glucose_daily]
            }],
            "stats": {
                "avg_tir_pct": round(sum(d["tirPercent"] for d in glucose_daily) / len(glucose_daily), 1),
                "avg_bp": None,
                "avg_hba1c": round(
                    sum(d["estimatedHba1c"] for d in glucose_daily if d.get("estimatedHba1c")) /
                    max(1, len([d for d in glucose_daily if d.get("estimatedHba1c")])), 1
                )
            }
        }
        bp = data.get("bp") or []
        if bp:
            avg_sys = round(sum(d["averageSystolic"] for d in bp) / len(bp))
            avg_dia = round(sum(d["averageDiastolic"] for d in bp) / len(bp))
            chart_data["stats"]["avg_bp"] = f"{avg_sys}/{avg_dia}"
        return chart_data

    def _summarize(self, metric: str, glucose_daily: list, data: dict) -> dict:
        """Small numeric summary used INSTEAD of the raw per-day array once a
        chart is already being shown -- the chart carries the daily detail,
        so the LLM is given nothing left to re-render as a table."""
        if metric == "tir":
            return {
                "days_count": len(glucose_daily),
                "avg_tir_pct": round(sum(d["tirPercent"] for d in glucose_daily) / len(glucose_daily), 1),
                "avg_tar_pct": round(sum(d["tarPercent"] for d in glucose_daily) / len(glucose_daily), 1),
                "avg_tbr_pct": round(sum(d["tbrPercent"] for d in glucose_daily) / len(glucose_daily), 1),
            }
        if metric == "sleep":
            sleep = data.get("sleep") or []
            if not sleep:
                return {"days_count": 0}
            return {
                "days_count": len(sleep),
                "avg_light_hrs": round(sum(d.get("lightSleep", 0) for d in sleep) / len(sleep) / 60, 1),
                "avg_deep_min": round(sum(d.get("deepSleep", 0) for d in sleep) / len(sleep)),
                "avg_rem_min": round(sum(d.get("remSleep", 0) for d in sleep) / len(sleep)),
            }
        if metric == "activity":
            activity = data.get("activity") or []
            if not activity:
                return {"days_count": 0}
            steps = [d.get("totalSteps", 0) for d in activity]
            return {
                "days_count": len(activity),
                "avg_steps": round(sum(steps) / len(steps)),
                "max_steps": max(steps),
                "min_steps": min(steps),
                "avg_calories": round(sum(d.get("calories", 0) for d in activity) / len(activity)),
            }
        return {}

    def _resolve_patient_id(self, patient_id: Optional[int], patient_name: Optional[str]) -> tuple:
        """Resolves patient_id from patient_name if needed, using the same
        substring-match pattern already proven working in UserProfileTool.
        Returns (resolved_id, error_response_or_None). If error_response is
        not None, _run should return it immediately without proceeding."""
        if patient_id:
            return patient_id, None
        if not patient_name:
            return None, json.dumps({
                "status": "not_found",
                "notice": "Either patient_id or patient_name is required."
            })

        user_context = getattr(self, 'user_context', None)

        with DatabaseManager() as db_manager:
            # Scope to THIS doctor's own patients first — resolves most name
            # ambiguity automatically (e.g. multiple "Vikas Reddy" in the
            # hospital, but only one assigned to this doctor).
            doctor_id = user_context.get('user_id') if user_context else None
            own_patients = db_manager.get_doctor_patients(doctor_user_id=doctor_id) if doctor_id else []
            own_matching = [
                p for p in own_patients
                if patient_name.lower() in f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".lower()
            ]

            if own_matching:
                if len(own_matching) > 1:
                    return None, json.dumps({
                        "status": "ambiguous_name",
                        "notice": f"Multiple patients match '{patient_name}'. Ask the user which one.",
                        "matching_patients": [
                            {
                                "id": p["patient_id"],
                                "name": f"{p.get('patient_first_name') or ''} {p.get('patient_last_name') or ''}".strip(),
                                "email": p.get("patient_email")
                            } for p in own_matching
                        ]
                    })
                return own_matching[0]["patient_id"], None

            # Fall back to searching all patients only if none of the doctor's
            # own patients match (e.g. covering-for-a-colleague scenarios).
            users = db_manager.get_users()
            matching_users = [
                u for u in users
                if patient_name.lower() in f"{u.first_name or ''} {u.last_name or ''}".lower()
                and u.role_id == 1
            ]

            if not matching_users:
                return None, json.dumps({
                    "status": "not_found",
                    "notice": f"No patient found with name containing '{patient_name}'."
                })

            if len(matching_users) > 1:
                return None, json.dumps({
                    "status": "ambiguous_name",
                    "notice": f"Multiple patients match '{patient_name}'. Ask the user which one.",
                    "matching_patients": [
                        {
                            "id": u.id,
                            "name": f"{u.first_name or ''} {u.last_name or ''}".strip(),
                            "email": u.email
                        } for u in matching_users
                    ]
                })

            return matching_users[0].id, None

    def _run(self, patient_id: Optional[int] = None, patient_name: Optional[str] = None,
             from_date: Optional[str] = None, to_date: Optional[str] = None,
             metric: Optional[str] = "glucose") -> str:
        if metric not in ("glucose", "tir", "sleep", "activity"):
            metric = "glucose"

        patient_id, resolution_error = self._resolve_patient_id(patient_id, patient_name)
        if resolution_error:
            object.__setattr__(self, 'last_chart_data', None)
            return resolution_error

        user_specified_range = bool(from_date or to_date)

        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        user_context = getattr(self, 'user_context', None)
        auth_token = user_context.get('auth_token') if user_context else None
        if not auth_token:
            return "Error: no authentication token available."

        try:
            data = self._fetch(patient_id, from_date, to_date, auth_token)
        except httpx.TimeoutException:
            return "The health data service timed out."
        except httpx.HTTPStatusError as e:
            return f"Could not retrieve health data (status {e.response.status_code})."
        except Exception as e:
            return f"Error retrieving health data: {str(e)}"

        glucose_daily = data.get("glucoseDailyAnalytics") or []
        status = "ok"
        notice = None

        if not glucose_daily:
            if user_specified_range:
                status = "no_data_in_range"
                notice = (
                    f"There are no glucose trends for patient {patient_id} "
                    f"between {from_date} and {to_date}."
                )
                object.__setattr__(self, 'last_chart_data', None)
                return json.dumps({
                    "patient_id": patient_id,
                    "date_range": f"{from_date} to {to_date}",
                    "status": status,
                    "notice": notice,
                    "glucose_daily": []
                })
            else:
                fallback_from = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
                try:
                    probe_data = self._fetch(patient_id, fallback_from, to_date, auth_token)
                except Exception:
                    probe_data = {}
                probe_glucose = probe_data.get("glucoseDailyAnalytics") or []
                if not probe_glucose:
                    status = "no_data_at_all"
                    notice = f"No glucose readings were found for patient {patient_id} in the last 180 days."
                    object.__setattr__(self, 'last_chart_data', None)
                    return json.dumps({
                        "patient_id": patient_id,
                        "status": status,
                        "notice": notice,
                        "glucose_daily": []
                    })

                most_recent_date = max(d["glucoseDate"] for d in probe_glucose)
                window_end_dt = datetime.strptime(most_recent_date, "%Y-%m-%d")
                window_start_dt = window_end_dt - timedelta(days=6)
                actual_from = window_start_dt.strftime("%Y-%m-%d")
                actual_to = window_end_dt.strftime("%Y-%m-%d")

                try:
                    data = self._fetch(patient_id, actual_from, actual_to, auth_token)
                except Exception:
                    data = probe_data

                glucose_daily = data.get("glucoseDailyAnalytics") or []
                status = "fallback_most_recent"
                notice = (
                    f"No glucose data in the last 7 days ({from_date} to {to_date}). "
                    f"Showing the most recent available week: {actual_from} to {actual_to}."
                )
                from_date, to_date = actual_from, actual_to

        program_summary = {}
        raw_program = data.get("patientProgramDetails")
        if raw_program:
            try:
                content = json.loads(raw_program).get("content", {})
                prog = content.get("patientProgram", {})
                details = content.get("patientDetails", {})
                program_summary = {
                    "doctor": prog.get("doctorName"),
                    "dha": prog.get("dhaName"),
                    "days_completed": prog.get("daysCompleted"),
                    "program_days": prog.get("programDays"),
                    "medications": prog.get("medications"),
                    "supplements": prog.get("supplements"),
                    "is_diabetic": details.get("isDiabetic"),
                }
            except Exception:
                pass

        metric_notice = None
        chart_data = None
        if len(glucose_daily) > 1:
            chart_data = self._build_chart(metric, glucose_daily, data, from_date, to_date)
            object.__setattr__(self, 'last_chart_data', chart_data)
            if chart_data is None and metric != "glucose":
                metric_notice = (
                    f"No {metric} data was recorded for patient {patient_id} "
                    f"during {from_date} to {to_date}."
                )
        else:
            object.__setattr__(self, 'last_chart_data', None)

        base_payload = {
            "patient_id": patient_id,
            "date_range": f"{from_date} to {to_date}",
            "status": status,
            "notice": notice,
            "metric_requested": metric,
            "metric_notice": metric_notice,
        }

        RAW_FALLBACK_CAP = 20  # avoid dumping large uncapped arrays into what the LLM
                                # has to read — an uncapped payload like this caused an
                                # "Agent stopped due to max iterations" failure elsewhere
                                # in this codebase (AGP chart's base64 image).

        if metric == "tir":
            if chart_data is not None:
                base_payload["summary"] = self._summarize(metric, glucose_daily, data)
            else:
                base_payload["glucose_daily"] = glucose_daily[:RAW_FALLBACK_CAP]
                if len(glucose_daily) > RAW_FALLBACK_CAP:
                    base_payload["glucose_daily_note"] = f"Showing {RAW_FALLBACK_CAP} of {len(glucose_daily)} entries."
        elif metric == "sleep":
            if chart_data is not None:
                base_payload["summary"] = self._summarize(metric, glucose_daily, data)
            else:
                raw_sleep = data.get("sleep") or []
                base_payload["sleep"] = raw_sleep[:RAW_FALLBACK_CAP]
                if len(raw_sleep) > RAW_FALLBACK_CAP:
                    base_payload["sleep_note"] = f"Showing {RAW_FALLBACK_CAP} of {len(raw_sleep)} entries."
        elif metric == "activity":
            if chart_data is not None:
                base_payload["summary"] = self._summarize(metric, glucose_daily, data)
            else:
                raw_activity = data.get("activity") or []
                base_payload["activity"] = raw_activity[:RAW_FALLBACK_CAP]
                if len(raw_activity) > RAW_FALLBACK_CAP:
                    base_payload["activity_note"] = f"Showing {RAW_FALLBACK_CAP} of {len(raw_activity)} entries."
        else:
            base_payload.update({
                "glucose_daily": glucose_daily[:RAW_FALLBACK_CAP],
                "blood_pressure": (data.get("bp") or [])[:RAW_FALLBACK_CAP],
                "heart_rate": (data.get("hr") or [])[:RAW_FALLBACK_CAP],
                "stress": (data.get("stress") or [])[:RAW_FALLBACK_CAP],
                "hrv": (data.get("hrv") or [])[:RAW_FALLBACK_CAP],
                "activity": (data.get("activity") or [])[:RAW_FALLBACK_CAP],
                "sleep": (data.get("sleep") or [])[:RAW_FALLBACK_CAP],
                "food_log_count": len(data.get("foodLogs") or []),
                "program_summary": program_summary
            })

        return json.dumps(base_payload)