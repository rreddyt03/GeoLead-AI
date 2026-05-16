# Addison Windows AI Sales Assistant - Sample Flow

This file shows the intended production-style behavior of the chatbot.

## Conversation Example

User: What kind of windows do you have?  
Assistant: We offer several popular types: sliding, casement, double-hung, picture, and bay/bow. Each fits different goals like airflow, light, and design. Are you planning replacement or a new installation?

User: Replacement for my bedroom  
Assistant: Great. How many windows are we planning for?

User: 3  
Assistant: Helpful. What size are the windows?

User: 5x6  
Assistant: For a bedroom, sliding is often a strong fit for easy operation and a clean look. Do you want sliding, or would you like a casement comparison first?

User: Sliding  
Assistant: Nice choice. Do you prefer vinyl, wood, or aluminum?

User: Wood, and what will pricing look like?  
Assistant: Wood usually sits in the balanced-to-premium range depending on size and install complexity. I can narrow this more after timeline and budget. When do you want to start?

User: 1 week  
Assistant: Perfect. Are you aiming for budget-friendly, balanced, or premium?

User: Balanced  
Assistant: Based on your details, 400 Series is the best fit right now. Would you like me to schedule a consultation or prepare a quote?

User: Yes  
Assistant: Great. What name should I use?

User: Rishi Reddy  
Assistant: Thanks, Rishi. What email should I use for your quote details?

User: rishi@example.com  
Assistant: Perfect. What phone number should our team use for follow-up?

User: 4695551212  
Assistant: Done. I’ve saved your consultation details and shared the request with our team for follow-up.

## Error Handling Examples

- Missing count: Assistant asks only for count, not repeating answered fields.
- Invalid email: Assistant asks for a valid email format.
- Invalid phone: Assistant asks for a valid phone with area code.
- Ambiguous message: Assistant answers what it can, then asks one next relevant question.
