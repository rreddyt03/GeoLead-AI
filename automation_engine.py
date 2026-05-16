def decide_action(lead_type):
    if lead_type == "HOT":
        return {
            "action": "CALL_SALES_TEAM",
            "priority": "HIGH",
        }
    if lead_type == "WARM":
        return {
            "action": "RUN_RETARGETING_ADS",
            "priority": "MEDIUM",
        }
    return {
        "action": "STORE_FOR_LATER",
        "priority": "LOW",
    }