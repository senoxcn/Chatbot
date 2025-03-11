intent_map = {
    "procurement_contact": {
        "keywords": [
            "contact procurement",
            "procurement contact",
            "procurement focal person",
            "contact person in procurement",
            "who handles procurement",
            "who to contact for procurement"
        ],
        "context": "The contact person for procurement inquiries is Dela Cruz, Juan. You can reach him at delacruz.juan@email.com."
    },
    "room_booking": {
        "keywords": [
            "how to book a room",
            "room reservation",
            "reserve meeting room"
        ],
        "context": "To book a room, visit room-reservation.com and follow the steps provided."
    },
    "admin_response_time": {
        "keywords": [
            "admin response time",
            "when will admin reply",
            "how long admin takes"
        ],
        "context": "Admin typically responds within 2-3 business days via email."
    }
}

#for intents
def detect_intent(user_query):
    for intent, details in intent_map.items():
        if any(keyword in user_query.lower() for keyword in details["keywords"]):
            return intent
    return None