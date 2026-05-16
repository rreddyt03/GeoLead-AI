import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from env_loader import load_env_files


load_env_files()

logger = logging.getLogger(__name__)


SALESFORCE_LOGIN_URL = os.environ.get("SALESFORCE_LOGIN_URL", "https://login.salesforce.com")
SALESFORCE_CLIENT_ID = os.environ.get("SALESFORCE_CLIENT_ID")
SALESFORCE_CLIENT_SECRET = os.environ.get("SALESFORCE_CLIENT_SECRET")
SALESFORCE_USERNAME = os.environ.get("SALESFORCE_USERNAME")
SALESFORCE_PASSWORD = os.environ.get("SALESFORCE_PASSWORD")
SALESFORCE_SECURITY_TOKEN = os.environ.get("SALESFORCE_SECURITY_TOKEN", "")
SALESFORCE_API_VERSION = os.environ.get("SALESFORCE_API_VERSION", "v61.0")
SALESFORCE_AUTH_FLOW = os.environ.get("SALESFORCE_AUTH_FLOW", "password")


def salesforce_is_configured():
    if SALESFORCE_AUTH_FLOW == "client_credentials":
        return all([SALESFORCE_CLIENT_ID, SALESFORCE_CLIENT_SECRET])
    return all(
        [
            SALESFORCE_CLIENT_ID,
            SALESFORCE_CLIENT_SECRET,
            SALESFORCE_USERNAME,
            SALESFORCE_PASSWORD,
        ]
    )


def get_salesforce_access_token():
    token_url = SALESFORCE_LOGIN_URL.rstrip("/") + "/services/oauth2/token"
    if SALESFORCE_AUTH_FLOW == "client_credentials":
        payload_dict = {
            "grant_type": "client_credentials",
            "client_id": SALESFORCE_CLIENT_ID,
            "client_secret": SALESFORCE_CLIENT_SECRET,
        }
    else:
        payload_dict = {
            "grant_type": "password",
            "client_id": SALESFORCE_CLIENT_ID,
            "client_secret": SALESFORCE_CLIENT_SECRET,
            "username": SALESFORCE_USERNAME,
            "password": f"{SALESFORCE_PASSWORD}{SALESFORCE_SECURITY_TOKEN}",
        }

    payload = urllib.parse.urlencode(payload_dict).encode()

    request_obj = urllib.request.Request(token_url, data=payload, method="POST")
    try:
        with urllib.request.urlopen(request_obj, timeout=20) as response:
            data = json.loads(response.read().decode())
            logger.info("Salesforce token acquired for instance %s", data.get("instance_url"))
            return data["access_token"], data["instance_url"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        logger.error("Salesforce token request failed (%s): %s", exc.code, body)
        raise RuntimeError(f"Salesforce token request failed ({exc.code}): {body}") from exc


def build_salesforce_lead_payload(
    data,
    recommendation_reason,
    summary_text,
    lead_score,
    lead_status,
    conversation_text,
):
    recommended_model = data.get("recommended_model")
    project_label = (data.get("project_intent") or "window").title()
    customer_name = data.get("customer_name") or "Website Lead"
    name_parts = customer_name.split()
    first_name = name_parts[0]
    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else "Lead"

    description = build_description(
        data,
        conversation_text,
        recommendation_reason,
        lead_score,
        lead_status,
        summary_text,
    )

    payload = {
        "FirstName": first_name,
        "LastName": last_name,
        "Email": data.get("customer_email"),
        "Phone": data.get("customer_phone"),
        "Company": "Website Lead",
        "LeadSource": "Website AI Chatbot",
        "Description": description,
        "Rating": lead_status,
        "Title": f"{project_label} Inquiry",
    }

    ai_status_field = os.environ.get("SALESFORCE_AI_STATUS_FIELD")
    if ai_status_field:
        payload[ai_status_field] = lead_status

    custom_field_map = {
        "WINDOW_SIZE_FIELD": ("size", str),
        "WINDOW_TYPE_FIELD": ("style", str),
        "WINDOW_MATERIAL_FIELD": ("material", str),
        "WINDOW_TIMELINE_FIELD": ("timeline", str),
        "WINDOW_BUDGET_FIELD": ("budget", str),
        "WINDOW_RECOMMENDED_MODEL_FIELD": ("recommended_model", str),
    }

    for env_key, (data_key, caster) in custom_field_map.items():
        sf_field_name = os.environ.get(env_key)
        if sf_field_name and data.get(data_key):
            payload[sf_field_name] = caster(data.get(data_key))

    return payload


def _stringify_action(action_value):
    if isinstance(action_value, dict):
        action = action_value.get("action") or "Unknown"
        priority = action_value.get("priority")
        if priority:
            return f"{action} ({priority})"
        return action
    return action_value or "Not set"


def _normalize_urgency(timeline_value):
    if not timeline_value:
        return "Unknown"
    lowered = str(timeline_value).lower()
    if lowered == "asap":
        return "High"
    if "week" in lowered:
        return "Medium"
    return "Low"


def _normalize_budget_sensitivity(budget_value):
    if not budget_value:
        return "Unknown"
    mapping = {
        "low": "High",
        "medium": "Medium",
        "high": "Low",
    }
    return mapping.get(str(budget_value).lower(), str(budget_value))


def _normalize_buying_intent(project_intent, timeline_value):
    if not project_intent and not timeline_value:
        return "Unknown"
    if str(timeline_value).lower() == "asap":
        return "High"
    if project_intent:
        return f"Active {str(project_intent).title()} Project"
    return "Researching"


def _normalize_key_interest(data):
    if data.get("room") and data.get("style"):
        return f"{data.get('room').title()} {data.get('style')} windows"
    if data.get("style"):
        return f"{data.get('style').title()} windows"
    if data.get("material"):
        return f"{data.get('material').title()} material"
    return "General window consultation"


def _display_ip(ip_value):
    if ip_value == "DEV_IP":
        return "Local Development Environment"
    return ip_value or "Not available"


def build_description(
    data,
    conversation_text,
    recommendation_reason,
    lead_score,
    lead_status,
    summary_text,
):
    recommended_product = data.get("recommended_product") or data.get("recommended_model") or "Not set"
    automation_action = _stringify_action(data.get("automation_action"))
    project_type = data.get("project_intent") or data.get("intent") or "Not set"
    window_count = data.get("window_count") or data.get("count") or "Not set"
    confidence_score = data.get("confidence_score") or "N/A"
    city = data.get("city") or data.get("geo_city") or "Unknown"
    region = data.get("region") or data.get("geo_region") or "Unknown"
    zip_code = data.get("zip") or data.get("geo_zip") or "Unknown"

    description = f"""
==============================
ADDISON WINDOWS AI LEAD REPORT
==============================

CUSTOMER DETAILS
----------------
Name: {data.get("customer_name") or "Not provided"}
Email: {data.get("customer_email") or "Not provided"}
Phone: {data.get("customer_phone") or "Not provided"}

LOCATION
--------
City: {city}
Region: {region}
ZIP: {zip_code}

LOCATION INTELLIGENCE
---------------------
IP Address: {_display_ip(data.get("customer_ip"))}
City: {data.get("geo_city") or "Not available"}
Region: {data.get("geo_region") or "Not available"}
ZIP: {data.get("geo_zip") or "Not available"}
Country: {data.get("geo_country") or "Not available"}
Coordinates: {data.get("geo_lat") or "Not available"}, {data.get("geo_lon") or "Not available"}
Approx Address: {data.get("geo_address") or "Not available"}
ISP: {data.get("geo_isp") or "Not available"}
Geo Confidence Score: {data.get("geo_confidence") or 0}/100

PROJECT DETAILS
---------------
Project Type: {project_type}
Window Count: {window_count}
Size: {data.get("size") or "Not set"}
Room: {data.get("room") or "Not set"}
Style: {data.get("style") or "Not set"}
Material: {data.get("material") or "Not set"}
Timeline: {data.get("timeline") or "Not set"}
Budget: {data.get("budget") or "Not set"}

AI QUALIFICATION
----------------
Lead Score: {data.get("lead_score_v2") or lead_score}
Lead Type: {data.get("lead_type") or lead_status}
Segment: {data.get("lead_segment") or "Not set"}
Priority Action: {automation_action}

RECOMMENDATION
--------------
Suggested Product: {recommended_product}
Reason: {data.get("recommendation_reason") or recommendation_reason or "Not set"}

SALES INSIGHTS
--------------
Buying Intent: {_normalize_buying_intent(project_type, data.get("timeline"))}
Urgency Level: {_normalize_urgency(data.get("timeline"))}
Budget Sensitivity: {_normalize_budget_sensitivity(data.get("budget"))}
Key Interest: {_normalize_key_interest(data)}
Confidence Score: {confidence_score}

NEXT BEST ACTION
----------------
{automation_action}

AI SUMMARY
----------
{summary_text or "Not available"}

FULL CONVERSATION
-----------------
{conversation_text or "Not available"}

==============================
END OF REPORT
==============================
"""

    return description.strip()


def sync_lead_to_salesforce(
    data,
    recommendation_reason,
    summary_text,
    lead_score,
    lead_status,
    conversation_text,
):
    if not salesforce_is_configured():
        logger.warning("Salesforce sync skipped: integration not configured")
        return None, "not_configured"

    if not data.get("customer_email"):
        logger.warning("Salesforce sync skipped: missing email")
        return None, "missing_email"

    payload = json.dumps(
        build_salesforce_lead_payload(
            data,
            recommendation_reason,
            summary_text,
            lead_score,
            lead_status,
            conversation_text,
        )
    ).encode()

    try:
        access_token, instance_url = get_salesforce_access_token()
        logger.info("Salesforce instance URL: %s", instance_url)
        lead_id = data.get("salesforce_lead_id")
        if lead_id:
            endpoint = f"{instance_url}/services/data/{SALESFORCE_API_VERSION}/sobjects/Lead/{lead_id}"
            method = "PATCH"
        else:
            endpoint = f"{instance_url}/services/data/{SALESFORCE_API_VERSION}/sobjects/Lead"
            method = "POST"

        request_obj = urllib.request.Request(
            endpoint,
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(request_obj, timeout=20) as response:
            if method == "POST":
                response_data = json.loads(response.read().decode())
                logger.info("Salesforce lead sync succeeded method=%s lead_id=%s", method, response_data.get("id"))
                return response_data.get("id"), "synced"
            logger.info("Salesforce lead sync succeeded method=%s lead_id=%s", method, lead_id)
            return lead_id, "synced"
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace")
        logger.error("Salesforce lead sync failed (%s): %s", exc.code, error_body)
        return data.get("salesforce_lead_id"), "error"
    except Exception as exc:
        logger.exception("Unexpected Salesforce sync failure: %s", exc)
        return data.get("salesforce_lead_id"), "error"
