// Simple diagnostic webhook with explicit error logging

import { serve } from "https://deno.land/std@0.192.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// For logging context
const WEBHOOK_VERSION = "diagnostic-v1";

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

// Test endpoint for direct insertion testing
serve(async (req) => {
  console.log(
    `🚀 [${WEBHOOK_VERSION}] Request received: ${new Date().toISOString()}`
  );

  // Handle test endpoint
  if (req.method === "GET") {
    const url = new URL(req.url);

    if (url.pathname.includes("/test-insert")) {
      const userId = url.searchParams.get("user_id");
      const customerId =
        url.searchParams.get("customer_id") || `cus_test_${Date.now()}`;

      if (!userId) {
        return new Response(
          JSON.stringify({ error: "Missing user_id parameter" }),
          {
            status: 400,
            headers: { "Content-Type": "application/json" },
          }
        );
      }

      console.log(`🧪 Test insertion request for user ${userId}`);

      try {
        // First, check if RLS is enabled on the table
        const { data: rlsData, error: rlsError } = await supabase.rpc(
          "check_rls_enabled",
          {
            table_name: "stripe_customers",
          }
        );

        console.log(`🔒 RLS check result:`, rlsData, rlsError);

        // Try direct insert with detailed logging
        console.log(
          `📥 Attempting direct insert: user=${userId}, customer=${customerId}`
        );

        const insertResult = await supabase.from("stripe_customers").insert([
          {
            id: userId,
            stripe_customer_id: customerId,
          },
        ]);

        console.log(`📄 Insert result:`, {
          data: insertResult.data,
          error: insertResult.error
            ? {
                message: insertResult.error.message,
                details: insertResult.error.details,
                hint: insertResult.error.hint,
                code: insertResult.error.code,
              }
            : null,
          status: insertResult.status,
          statusText: insertResult.statusText,
        });

        // Try raw SQL insert as a fallback
        if (insertResult.error) {
          console.log(`🔄 Trying SQL insert fallback`);

          const sqlResult = await supabase.rpc("execute_direct_insert", {
            user_id: userId,
            customer_id: customerId,
          });

          console.log(`📄 SQL insert result:`, {
            data: sqlResult.data,
            error: sqlResult.error
              ? {
                  message: sqlResult.error.message,
                  details: sqlResult.error.details,
                  hint: sqlResult.error.hint,
                  code: sqlResult.error.code,
                }
              : null,
          });

          if (sqlResult.error) {
            return new Response(
              JSON.stringify({
                test: "direct-insert",
                success: false,
                standard_insert: { success: false, error: insertResult.error },
                sql_insert: { success: false, error: sqlResult.error },
              }),
              {
                headers: { "Content-Type": "application/json" },
              }
            );
          } else {
            return new Response(
              JSON.stringify({
                test: "direct-insert",
                success: true,
                message: "SQL insert succeeded but standard insert failed",
                standard_insert: { success: false, error: insertResult.error },
                sql_insert: { success: true },
              }),
              {
                headers: { "Content-Type": "application/json" },
              }
            );
          }
        } else {
          return new Response(
            JSON.stringify({
              test: "direct-insert",
              success: true,
              message: "Standard insert succeeded",
              standard_insert: { success: true },
            }),
            {
              headers: { "Content-Type": "application/json" },
            }
          );
        }
      } catch (e) {
        console.error(`❌ Unexpected error during test:`, e);
        return new Response(
          JSON.stringify({
            test: "direct-insert",
            success: false,
            error: e.toString(),
          }),
          {
            status: 500,
            headers: { "Content-Type": "application/json" },
          }
        );
      }
    }

    // Generic info for any other GET request
    return new Response(
      JSON.stringify({
        message:
          "This is a webhook endpoint. Use POST for webhook events or GET /test-insert for testing.",
        version: WEBHOOK_VERSION,
      }),
      {
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  const sig = req.headers.get("stripe-signature");
  const body = await req.text();

  console.log(
    `📦 [${WEBHOOK_VERSION}] Request body length: ${body.length} chars`
  );

  let event;
  try {
    event = await stripe.webhooks.constructEventAsync(
      body,
      sig!,
      endpointSecret
    );
    console.log(
      `✅ [${WEBHOOK_VERSION}] Webhook verified. Event type: ${event.type}`
    );
  } catch (err) {
    console.error(`❌ [${WEBHOOK_VERSION}] Webhook verification failed:`, err);
    return new Response("Webhook Error: Invalid signature", { status: 400 });
  }

  if (event.type === "checkout.session.completed") {
    const session = event.data.object;
    const email = session.customer_email;
    const customer_id = session.customer;

    console.log(
      `💳 [${WEBHOOK_VERSION}] Checkout completed for ${email} with customer ID ${customer_id}`
    );

    try {
      // STEP 1: Update user paid status - This part works
      console.log(`💰 Updating paid status for email: ${email}`);
      const updateResult = await supabase
        .from("users")
        .update({ paid: true })
        .eq("email", email);

      console.log(`📄 Update result:`, {
        data: updateResult.data,
        error: updateResult.error
          ? {
              message: updateResult.error.message,
              details: updateResult.error.details,
              hint: updateResult.error.hint,
              code: updateResult.error.code,
            }
          : null,
        status: updateResult.status,
        statusText: updateResult.statusText,
      });

      // STEP 2: Get user ID from email
      console.log(`🔍 Looking up user ID for email: ${email}`);
      const userResult = await supabase
        .from("users")
        .select("id")
        .eq("email", email)
        .maybeSingle();

      console.log(`📄 User lookup result:`, {
        data: userResult.data,
        error: userResult.error
          ? {
              message: userResult.error.message,
              details: userResult.error.details,
              hint: userResult.error.hint,
              code: userResult.error.code,
            }
          : null,
        status: userResult.status,
        statusText: userResult.statusText,
      });

      if (userResult.error || !userResult.data) {
        console.error(`❌ Failed to find user for email: ${email}`);
        return new Response(
          JSON.stringify({
            received: true,
            error: "User lookup failed",
          }),
          {
            headers: { "Content-Type": "application/json" },
          }
        );
      }

      const userId = userResult.data.id;
      console.log(`🆔 Found user ID: ${userId}`);

      // STEP 3: Insert stripe customer record
      console.log(`💳 Inserting stripe customer record for user ID: ${userId}`);

      // Try standard insert
      const insertResult = await supabase.from("stripe_customers").insert([
        {
          id: userId,
          stripe_customer_id: customer_id,
        },
      ]);

      console.log(`📄 Insert result:`, {
        data: insertResult.data,
        error: insertResult.error
          ? {
              message: insertResult.error.message,
              details: insertResult.error.details,
              hint: insertResult.error.hint,
              code: insertResult.error.code,
            }
          : null,
        status: insertResult.status,
        statusText: insertResult.statusText,
      });

      // If standard insert fails, try SQL fallback
      if (insertResult.error) {
        console.log(`🔄 Standard insert failed, trying SQL fallback`);

        const sqlResult = await supabase.rpc("execute_direct_insert", {
          user_id: userId,
          customer_id: customer_id,
        });

        console.log(`📄 SQL insert result:`, {
          data: sqlResult.data,
          error: sqlResult.error
            ? {
                message: sqlResult.error.message,
                details: sqlResult.error.details,
                hint: sqlResult.error.hint,
                code: sqlResult.error.code,
              }
            : null,
        });

        if (sqlResult.error) {
          console.error(`❌ Both insert methods failed`);
          return new Response(
            JSON.stringify({
              received: true,
              success: false,
              update_status: !updateResult.error,
              insert_status: false,
              sql_status: false,
            }),
            {
              headers: { "Content-Type": "application/json" },
            }
          );
        } else {
          console.log(`✅ SQL insert succeeded`);
          return new Response(
            JSON.stringify({
              received: true,
              success: true,
              message: "Used SQL fallback",
              update_status: !updateResult.error,
              insert_status: false,
              sql_status: true,
            }),
            {
              headers: { "Content-Type": "application/json" },
            }
          );
        }
      } else {
        console.log(`✅ Standard insert succeeded`);
        return new Response(
          JSON.stringify({
            received: true,
            success: true,
            message: "Standard insert worked",
            update_status: !updateResult.error,
            insert_status: true,
          }),
          {
            headers: { "Content-Type": "application/json" },
          }
        );
      }
    } catch (error) {
      console.error(`❌ [${WEBHOOK_VERSION}] Unexpected error:`, error);
      return new Response(
        JSON.stringify({
          received: true,
          error: "Internal server error",
          details: error.toString(),
        }),
        {
          headers: { "Content-Type": "application/json" },
        }
      );
    }
  } else {
    console.log(
      `ℹ️ [${WEBHOOK_VERSION}] Ignoring non-checkout event: ${event.type}`
    );
  }

  return new Response(JSON.stringify({ received: true }), {
    headers: { "Content-Type": "application/json" },
  });
});
