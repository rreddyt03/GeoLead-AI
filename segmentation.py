def segment_lead(lead_score, city):
    if lead_score >= 80:
        return "HIGH_PRIORITY"
    if city in ["Dallas", "Austin"]:
        return "TARGET_CITY"
    return "LOW_PRIORITY"