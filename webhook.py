import os
import json
import stripe
from flask import Flask, request, jsonify
from supabase import create_client, Client

# Load your keys
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")  # From Stripe dashboard
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Needs service role access

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except ValueError as e:
        return f"Invalid payload: {e}", 400
    except stripe.error.SignatureVerificationError:
        return "Invalid signature", 400

    # ✅ Detect successful checkout or subscription creation
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        email = session["customer_email"]

        # 🔄 Update user in Supabase
        supabase.table("users").update({"paid": True}).eq("email", email).execute()

    return jsonify(success=True)

if __name__ == "__main__":
    app.run(port=4242)
