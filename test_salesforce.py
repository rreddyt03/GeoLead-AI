import json

from salesforce import (
    build_salesforce_lead_payload,
    get_salesforce_access_token,
    salesforce_is_configured,
    sync_lead_to_salesforce,
)


def main():
    if not salesforce_is_configured():
        print("Salesforce is not configured. Set the SALESFORCE_* environment variables first.")
        return

    try:
        access_token, instance_url = get_salesforce_access_token()
    except Exception as exc:
        print("Connection failed:")
        print(exc)
        return

    print("Connected successfully.")
    print("Instance URL:", instance_url)
    print("Access token received:", bool(access_token))

    sample_data = {
        "project_intent": "replace",
        "count": 3,
        "size": "5x5",
        "style": "sliding",
        "material": "wood",
        "timeline": "1 week",
        "budget": "medium",
        "recommended_model": "400 Series",
        "customer_name": "Test Website Lead",
        "customer_email": "testlead@example.com",
        "customer_phone": "(214) 555-0101",
        "salesforce_lead_id": None,
    }
    summary = (
        "Addison Windows Consultation Summary\n\n"
        "This is a test lead created to verify the Salesforce integration."
    )
    recommendation_reason = "Balanced premium option with strong everyday performance and broad homeowner appeal."
    lead_status = "Warm"
    conversation_text = (
        "User: I need replacement windows for my home.\n"
        "Assistant: I recommend the 400 Series based on your project details."
    )

    print("\nLead payload preview:")
    print(
        json.dumps(
            build_salesforce_lead_payload(
                sample_data,
                recommendation_reason,
                summary,
                lead_score=4,
                lead_status=lead_status,
                conversation_text=conversation_text,
            ),
            indent=2,
        )
    )

    lead_id, status = sync_lead_to_salesforce(
        sample_data,
        recommendation_reason,
        summary,
        lead_score=4,
        lead_status=lead_status,
        conversation_text=conversation_text,
    )
    print("\nSync result:")
    print("Lead ID:", lead_id)
    print("Status:", status)


if __name__ == "__main__":
    main()
