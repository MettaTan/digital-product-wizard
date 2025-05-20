// Supabase Edge Function: stripe-webhook
// Replace Flask completely

// Supabase Edge Function: stripe-webhook
// Full updated implementation with robust error handling

import { serve } from "https://deno.land/std@0.192.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const supabase = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
);

const stripeSecret = Deno.env.get("STRIPE_SECRET_KEY")!;
const endpointSecret = Deno.env.get("STRIPE_WEBHOOK_SECRET")!;

import Stripe from "https://esm.sh/stripe@12.14.0?target=deno";
const stripe = new Stripe(stripeSecret, {
  apiVersion: "2023-10-16",
  httpClient: Stripe.createFetchHttpClient(),
});

serve(async (req) => {
  console.log("⚡ WEBHOOK RECEIVED at " + new Date().toISOString());
  console.log("📨 Request method:", req.method);

  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  // Log environment details
  console.log("🌐 SUPABASE_URL:", Deno.env.get("SUPABASE_URL"));
  console.log(
    "🔑 SERVICE_ROLE_KEY (first 8 chars):",
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")?.slice(0, 8)
  );

  const sig = req.headers.get("stripe-signature");
  console.log("🔐 Stripe signature present:", !!sig);

  const body = await req.text();
  console.log("📦 Request body length:", body.length);

  let event;
  try {
    event = await stripe.webhooks.constructEventAsync(
      body,
      sig!,
      endpointSecret
    );
    console.log("✅ Webhook signature verified successfully");
    console.log("📨 Event type:", event.type);
  } catch (err) {
    console.error("❌ Webhook signature verification failed:", err);
    return new Response("Webhook Error", { status: 400 });
  }

  if (event.type === "checkout.session.completed") {
    const session = event.data.object;
    console.log("💳 Checkout session completed:", session.id);

    const email = session.customer_email;
    const customer_id = session.customer;

    console.log("📧 Customer email:", email);
    console.log("👤 Stripe customer ID:", customer_id);

    if (!email) {
      console.error("❌ Missing customer_email in session");
      return new Response(JSON.stringify({ error: "Missing customer email" }), {
        status: 200, // Still return 200 to avoid Stripe retries
        headers: { "Content-Type": "application/json" },
      });
    }

    if (!customer_id) {
      console.error("❌ Missing customer_id in session");
      console.error("📋 Full session data:", JSON.stringify(session));
      return new Response(JSON.stringify({ error: "Missing customer ID" }), {
        status: 200, // Still return 200 to avoid Stripe retries
        headers: { "Content-Type": "application/json" },
      });
    }

    try {
      // STEP 1: Update users.paid to true
      console.log("🔄 Updating paid status for email:", email);
      const { data: updateData, error: updateError } = await supabase
        .from("users")
        .update({ paid: true })
        .eq("email", email)
        .select();

      if (updateError) {
        console.error("❌ Error updating paid status:", updateError);
      } else {
        console.log("✅ Successfully updated paid status:", updateData);
      }

      // STEP 2: Get user by email to find their UUID
      console.log("🔍 Looking up user by email:", email);
      const { data: userData, error: userError } = await supabase
        .from("users")
        .select("id, email")
        .eq("email", email)
        .single();

      if (userError) {
        console.error("❌ Error finding user by email:", userError);
        return new Response(JSON.stringify({ error: "User lookup failed" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      if (!userData) {
        console.error("❌ No user found with email:", email);
        return new Response(JSON.stringify({ error: "User not found" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      console.log("✅ Found user:", userData);

      // STEP 3: Insert stripe customer record
      console.log("🔄 Saving stripe_customer record for user ID:", userData.id);
      console.log("🔄 Using stripe customer ID:", customer_id);

      // First check if a record already exists
      const { data: existingCustomer, error: checkError } = await supabase
        .from("stripe_customers")
        .select("*")
        .eq("id", userData.id)
        .maybeSingle();

      if (checkError) {
        console.error(
          "❌ Error checking for existing stripe_customer:",
          checkError
        );
      } else {
        console.log(
          "🔍 Existing stripe_customer check result:",
          existingCustomer
        );
      }

      // Based on whether record exists, update or insert
      let operation;
      if (existingCustomer) {
        console.log("🔄 Updating existing stripe_customer record");
        operation = supabase
          .from("stripe_customers")
          .update({ stripe_customer_id: customer_id })
          .eq("id", userData.id);
      } else {
        console.log("🔄 Inserting new stripe_customer record");
        operation = supabase.from("stripe_customers").insert([
          {
            id: userData.id,
            stripe_customer_id: customer_id,
          },
        ]);
      }

      const { data: customerData, error: customerError } = await operation;

      if (customerError) {
        console.error("❌ Error saving stripe_customer:", customerError);
        console.error("❌ Attempted data:", {
          id: userData.id,
          stripe_customer_id: customer_id,
        });
        console.error("❌ ID type:", typeof userData.id);
      } else {
        console.log("✅ Successfully saved stripe_customer:", customerData);
      }
    } catch (error) {
      console.error("❌ Unexpected error in webhook handler:", error);
      return new Response(JSON.stringify({ error: "Internal server error" }), {
        status: 200, // Still return 200 to avoid Stripe retries
        headers: { "Content-Type": "application/json" },
      });
    }
  } else {
    console.log("ℹ️ Ignoring non-checkout event:", event.type);
  }

  return new Response(JSON.stringify({ received: true }), {
    headers: { "Content-Type": "application/json" },
  });
});
