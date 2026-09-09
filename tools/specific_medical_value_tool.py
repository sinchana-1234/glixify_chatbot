#!/usr/bin/env python3
"""
Specific Medical Value Tool (PostgreSQL Version - Clean)
"""

import logging
import json
from typing import Optional
from datetime import datetime
from langchain.tools import BaseTool
from sqlalchemy import text

from dal.postgres_db import SessionLocalPG

logger = logging.getLogger(__name__)


class SpecificMedicalValueTool(BaseTool):
    name: str = "get_specific_medical_value"
    description: str = "Get specific medical readings (glucose, BP, etc.) using PostgreSQL."

    def set_user_context(self, user_context):
        object.__setattr__(self, 'user_context', user_context)

    def _run(
        self,
        patient_id: Optional[int] = None,
        patient_name: Optional[str] = None,
        reading_type: str = "glucose",
        specific_time: Optional[str] = None,
        date_filter: Optional[str] = None,
        time_range: Optional[str] = None,
        analysis_type: str = "specific"
    ) -> str:

        try:
            # 🔐 ROLE CONTROL
            user_context = getattr(self, 'user_context', None)
            if user_context and user_context.get('role_id') == 1:
                patient_id = user_context.get('user_id')
                patient_name = None
                logger.info(f"Patient access → ID {patient_id}")

            if not patient_id:
                return "Patient ID is required."

            db = SessionLocalPG()

            try:
                # -------------------------
                # SLEEP BRANCH
                # -------------------------
                # Sleep doesn't fit the "average a numeric column" pattern of
                # the other reading types. It lives in sleep_readings_details
                # with one row per minute of sleep, tagged by `level`:
                #   0 = deep sleep   |   1 = light sleep   |   2 = REM sleep
                #   3 = awake (not counted in total sleep)
                # We use night-attribution (actual_time + 12h) so a sleep
                # session crossing midnight is attributed to the wake-up date.
                # This mirrors the logic in dal/postgres_queries.py (REST) and
                # dal/services/medical_readings_service.py (chat sleep handler).
                if reading_type == "sleep":
                    sleep_params = {"patient_id": patient_id}

                    if date_filter:
                        if len(date_filter) == 7:
                            # YYYY-MM (month query)
                            date_condition = (
                                "TO_CHAR(actual_time + INTERVAL '12 hours', 'YYYY-MM') = :date"
                            )
                            sleep_params["date"] = date_filter
                        else:
                            # YYYY-MM-DD (single date — the "night ending on" target)
                            date_condition = (
                                "DATE(actual_time + INTERVAL '12 hours') = :date"
                            )
                            sleep_params["date"] = date_filter
                    else:
                        date_condition = "1=1"

                    sleep_row = db.execute(
                        text(f"""
                            SELECT
                                COUNT(*) FILTER (WHERE level = 0) AS deep_minutes,
                                COUNT(*) FILTER (WHERE level = 1) AS light_minutes,
                                COUNT(*) FILTER (WHERE level = 2) AS rem_minutes,
                                COUNT(*) FILTER (WHERE level = 3) AS awake_minutes,
                                COUNT(*) FILTER (WHERE level IN (0, 1, 2)) AS total_sleep_minutes
                            FROM sleep_readings_details
                            WHERE patient_id = :patient_id
                              AND {date_condition}
                        """),
                        sleep_params,
                    ).fetchone()

                    deep_min  = int(sleep_row[0] or 0)
                    light_min = int(sleep_row[1] or 0)
                    rem_min   = int(sleep_row[2] or 0)
                    awake_min = int(sleep_row[3] or 0)
                    total_min = int(sleep_row[4] or 0)

                    if total_min == 0:
                        return json.dumps({
                            "reading_type": "sleep",
                            "patient_id": patient_id,
                            "date_filter": date_filter,
                            "total_sleep_duration": "0h 0m",
                            "total_sleep_minutes": 0,
                            "sleep_breakdown": {
                                "deep_sleep_minutes": 0,
                                "light_sleep_minutes": 0,
                                "rem_sleep_minutes": 0,
                                "awake_minutes": awake_min,
                            },
                            "message": "No sleep data found for the requested period.",
                        })

                    return json.dumps({
                        "reading_type": "sleep",
                        "patient_id": patient_id,
                        "date_filter": date_filter,
                        "total_sleep_duration": f"{total_min // 60}h {total_min % 60}m",
                        "total_sleep_minutes": total_min,
                        "sleep_breakdown": {
                            "deep_sleep_minutes": deep_min,
                            "light_sleep_minutes": light_min,
                            "rem_sleep_minutes": rem_min,
                            "awake_minutes": awake_min,
                        },
                    })

                # -------------------------
                # TABLE MAPPING
                # -------------------------
                mapping = {
                    "glucose": ("glucose_readings", "glucose_value", "event_time_utc"),
                    "blood_pressure": ("blood_pressure_readings", "systolic", "date_time_utc"),
                    "spo2": ("spo2_readings", "value", "date_time_utc"),
                    "body_temperature": ("body_temperature_readings", "temperature", "date_time_utc"),
                    "heart_rate": ("heart_rate_readings", "value", "date_time_utc"),
                    "hrv": ("hrv_readings", "value", "date_time_utc"),
                    "stress": ("stress_readings", "value", "date_time_utc"),
                }

                if reading_type not in mapping:
                    return f"Unsupported reading type: {reading_type}"

                table, column, time_col = mapping[reading_type]

                # -------------------------
                # DATE FILTER
                # -------------------------
                params = {"patient_id": patient_id}

                if date_filter:
                    if len(date_filter) == 7:
                        date_condition = f"TO_CHAR({time_col}, 'YYYY-MM') = :date"
                        params["date"] = date_filter
                    else:
                        date_condition = f"DATE({time_col}) = :date"
                        params["date"] = date_filter
                else:
                    date_condition = "1=1"

                # -------------------------
                # TIME RANGE FILTER
                # -------------------------
                time_condition = "1=1"
                if time_range:
                    if time_range == "morning":
                        time_condition = f"EXTRACT(HOUR FROM {time_col}) BETWEEN 6 AND 11"
                    elif time_range == "afternoon":
                        time_condition = f"EXTRACT(HOUR FROM {time_col}) BETWEEN 12 AND 16"
                    elif time_range == "evening":
                        time_condition = f"EXTRACT(HOUR FROM {time_col}) BETWEEN 17 AND 20"
                    elif time_range == "night":
                        time_condition = f"(EXTRACT(HOUR FROM {time_col}) >= 21 OR EXTRACT(HOUR FROM {time_col}) <= 5)"

                # -------------------------
                # ANALYSIS TYPE
                # -------------------------
                if analysis_type == "highest":
                    order = "DESC"
                elif analysis_type == "lowest":
                    order = "ASC"
                else:
                    order = "DESC"

                # -------------------------
                # QUERY
                # -------------------------
                query = f"""
                    SELECT {column}, {time_col}
                    FROM {table}
                    WHERE patient_id = :patient_id
                    AND {date_condition}
                    AND {time_condition}
                    ORDER BY {column} {order}
                    LIMIT 10
                """

                results = db.execute(text(query), params).fetchall()

                if not results:
                    return f"No {reading_type} readings found."

                # -------------------------
                # FORMAT RESPONSE
                # -------------------------
                formatted = [
                    {
                        "value": float(r[0]) if r[0] is not None else None,
                        "time": str(r[1])
                    }
                    for r in results
                ]

                # Specific time handling
                if analysis_type == "specific" and specific_time:
                    try:
                        target_time = datetime.fromisoformat(specific_time)
                        closest = min(
                            formatted,
                            key=lambda x: abs(datetime.fromisoformat(x["time"]) - target_time)
                        )
                        return json.dumps({
                            "type": "specific",
                            "reading": closest
                        }, indent=2)
                    except Exception:
                        pass

                return json.dumps({
                    "type": analysis_type,
                    "reading_type": reading_type,
                    "count": len(formatted),
                    "results": formatted
                }, indent=2)

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error in SpecificMedicalValueTool: {e}")
            return f"Error: {str(e)}"