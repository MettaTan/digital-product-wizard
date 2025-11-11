import Stripe from "stripe";
import { ENV } from "./_core/env";

if (!ENV.stripeSecretKey) {
  throw new Error("STRIPE_SECRET_KEY is not configured");
}

export const stripe = new Stripe(ENV.stripeSecretKey, {
  apiVersion: "2025-10-29.clover",
});

/**
 * Create a Stripe checkout session for product purchase
 */
export async function createCheckoutSession(params: {
  productId: number;
  productTitle: string;
  tier: "monthly" | "yearly" | "lifetime";
  priceInCents: number;
  userId: number;
  userEmail: string;
  userName?: string;
  origin: string;
}): Promise<string> {
  const { productId, productTitle, tier, priceInCents, userId, userEmail, userName, origin } = params;

  // Determine mode and line items based on tier
  const mode: Stripe.Checkout.SessionCreateParams.Mode =
    tier === "lifetime" ? "payment" : "subscription";

  const session = await stripe.checkout.sessions.create({
    mode,
    payment_method_types: ["card"],
    customer_email: userEmail,
    client_reference_id: userId.toString(),
    metadata: {
      user_id: userId.toString(),
      product_id: productId.toString(),
      tier,
      customer_email: userEmail,
      customer_name: userName || "",
    },
    line_items: [
      {
        price_data: {
          currency: "usd",
          product_data: {
            name: productTitle,
            description: `${tier.charAt(0).toUpperCase() + tier.slice(1)} access`,
          },
          unit_amount: priceInCents,
          ...(mode === "subscription" && {
            recurring: {
              interval: tier === "monthly" ? "month" : "year",
            },
          }),
        },
        quantity: 1,
      },
    ],
    success_url: `${origin}/purchase-success?session_id={CHECKOUT_SESSION_ID}`,
    cancel_url: `${origin}/dashboard`,
    allow_promotion_codes: true,
  });

  if (!session.url) {
    throw new Error("Failed to create checkout session");
  }

  return session.url;
}

/**
 * Retrieve checkout session details
 */
export async function getCheckoutSession(sessionId: string) {
  return await stripe.checkout.sessions.retrieve(sessionId);
}

/**
 * Retrieve subscription details
 */
export async function getSubscription(subscriptionId: string) {
  return await stripe.subscriptions.retrieve(subscriptionId);
}

/**
 * Cancel a subscription
 */
export async function cancelSubscription(subscriptionId: string) {
  return await stripe.subscriptions.cancel(subscriptionId);
}

/**
 * Create a customer portal session for subscription management
 */
export async function createCustomerPortalSession(params: {
  customerId: string;
  origin: string;
}): Promise<string> {
  const { customerId, origin } = params;

  const session = await stripe.billingPortal.sessions.create({
    customer: customerId,
    return_url: `${origin}/dashboard`,
  });

  return session.url;
}
