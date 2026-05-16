HIGH_VALUE_CITIES = ["Dallas", "Austin", "New York", "San Francisco"]


def score_lead(lead_data):
    score = 0

    try:
        if lead_data.get("city") in HIGH_VALUE_CITIES:
            score += 30

        if int(lead_data.get("window_count", 0)) > 5:
            score += 25

        if lead_data.get("timeline") == "immediate":
            score += 20

        if lead_data.get("email"):
            score += 15

    except Exception as exc:
        print(f"[Lead Scoring Error]: {exc}")

    return score


def classify_lead(score):
    if score >= 80:
        return "HOT"
    if score >= 50:
        return "WARM"
    return "COLD"