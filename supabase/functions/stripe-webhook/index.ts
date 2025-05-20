// Supabase Edge Function: stripe-webhook
// Replace Flask completely

import { serve } from "https://deno.land/std@0.192.0/http/server.ts";

// Import Supabase client
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
  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  console.log("🔍 SUPABASE_URL:", Deno.env.get("SUPABASE_URL"));
  console.log(
    "🔍 SERVICE_ROLE_KEY:",
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")?.slice(0, 8)
  ); // don't log full key

  const sig = req.headers.get("stripe-signature");
  const body = await req.text();

  let event;
  try {
    event = await stripe.webhooks.constructEventAsync(
      body,
      sig!,
      endpointSecret
    );
  } catch (err) {
    console.error("❌ Webhook signature verification failed:", err);
    return new Response("Webhook Error", { status: 400 });
  }

  if (event.type === "checkout.session.completed") {
    const session = event.data.object;
    const email = session.customer_email;
    const customer_id = session.customer;

    console.log(
      `✅ Processing checkout for email: ${email}, customer: ${customer_id}`
    );

    if (email) {
      try {
        // 1. First, get the user_id from the users table using email
        const { data: userData, error: userError } = await supabase
          .from("users")
          .select("id")
          .eq("email", email)
          .single();

        if (userError) {
          console.error("❌ Error finding user by email:", userError);
        } else if (!userData) {
          console.error("❌ No user found with email:", email);
        } else {
          console.log("✅ Found user with id:", userData.id);

          // 2. Update user's paid status
          const { error: updateError } = await supabase
            .from("users")
            .update({ paid: true })
            .eq("id", userData.id);

          if (updateError) {
            console.error("❌ Error updating paid status:", updateError);
          } else {
            console.log("✅ Updated paid status for user:", userData.id);
          }

          // 3. Upsert into stripe_customers using the user's UUID
          if (customer_id) {
            const { error: customerError } = await supabase
              .from("stripe_customers")
              .upsert({
                id: userData.id, // Use the UUID from users table
                stripe_customer_id: customer_id,
              });

            if (customerError) {
              console.error(
                "❌ Error upserting stripe_customer:",
                customerError
              );
            } else {
              console.log(
                "✅ Saved Stripe customer data for user:",
                userData.id
              );
            }
          } else {
            console.warn("⚠️ No customer_id found in session");
          }
        }
      } catch (error) {
        console.error("❌ Unexpected error processing checkout:", error);
      }
    } else {
      console.warn("❌ No customer_email found in session");
    }
  }

  return new Response(JSON.stringify({ received: true }), {
    headers: { "Content-Type": "application/json" },
  });
});
