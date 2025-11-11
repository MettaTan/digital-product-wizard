import { Request, Response } from "express";
import Stripe from "stripe";
import { stripe } from "../stripe";
import { ENV } from "../_core/env";
import { createPurchase, updatePurchase, getPurchaseByUserAndProduct } from "../db";

export async function handleStripeWebhook(req: Request, res: Response) {
  const sig = req.headers["stripe-signature"];

  if (!sig) {
    console.error("[Stripe Webhook] No signature found");
    return res.status(400).send("No signature");
  }

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(
      req.body,
      sig,
      ENV.stripeWebhookSecret
    );
  } catch (err: any) {
    console.error(`[Stripe Webhook] Signature verification failed: ${err.message}`);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  // Handle test events
  if (event.id.startsWith("evt_test_")) {
    console.log("[Stripe Webhook] Test event detected, returning verification response");
    return res.json({ verified: true });
  }

  console.log(`[Stripe Webhook] Received event: ${event.type}`);

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        const session = event.data.object as Stripe.Checkout.Session;
        await handleCheckoutCompleted(session);
        break;
      }

      case "invoice.paid": {
        const invoice = event.data.object as Stripe.Invoice;
        await handleInvoicePaid(invoice);
        break;
      }

      case "customer.subscription.updated": {
        const subscription = event.data.object as Stripe.Subscription;
        await handleSubscriptionUpdated(subscription);
        break;
      }

      case "customer.subscription.deleted": {
        const subscription = event.data.object as Stripe.Subscription;
        await handleSubscriptionDeleted(subscription);
        break;
      }

      default:
        console.log(`[Stripe Webhook] Unhandled event type: ${event.type}`);
    }

    res.json({ received: true });
  } catch (error: any) {
    console.error(`[Stripe Webhook] Error processing event: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
}

async function handleCheckoutCompleted(session: Stripe.Checkout.Session) {
  console.log("[Stripe Webhook] Processing checkout.session.completed");

  const userId = session.metadata?.user_id;
  const productId = session.metadata?.product_id;
  const tier = session.metadata?.tier as "monthly" | "yearly" | "lifetime";

  if (!userId || !productId || !tier) {
    console.error("[Stripe Webhook] Missing metadata in checkout session");
    return;
  }

  const userIdNum = parseInt(userId);
  const productIdNum = parseInt(productId);

  // Check if purchase already exists
  const existingPurchase = await getPurchaseByUserAndProduct(userIdNum, productIdNum);

  if (existingPurchase) {
    console.log("[Stripe Webhook] Purchase already exists, updating...");
    await updatePurchase(existingPurchase.id, {
      status: "active",
      stripeCustomerId: session.customer as string || undefined,
      stripeSubscriptionId: session.subscription as string || undefined,
      stripePaymentIntentId: session.payment_intent as string || undefined,
    });
  } else {
    console.log("[Stripe Webhook] Creating new purchase...");
    await createPurchase({
      userId: userIdNum,
      productId: productIdNum,
      tier,
      status: "active",
      stripeCustomerId: session.customer as string || undefined,
      stripeSubscriptionId: session.subscription as string || undefined,
      stripePaymentIntentId: session.payment_intent as string || undefined,
    });
  }

  console.log("[Stripe Webhook] Purchase created/updated successfully");
}

async function handleInvoicePaid(invoice: Stripe.Invoice) {
  console.log("[Stripe Webhook] Processing invoice.paid");

  // Subscription invoices are handled by subscription.updated events
  console.log(`[Stripe Webhook] Invoice ${invoice.id} paid`);
}

async function handleSubscriptionUpdated(subscription: Stripe.Subscription) {
  console.log("[Stripe Webhook] Processing customer.subscription.updated");

  // Find purchase by subscription ID
  // For now, we'll just log it - you can add more logic here
  console.log(`[Stripe Webhook] Subscription ${subscription.id} status: ${subscription.status}`);
}

async function handleSubscriptionDeleted(subscription: Stripe.Subscription) {
  console.log("[Stripe Webhook] Processing customer.subscription.deleted");

  // Mark purchase as canceled
  // You would need to add a helper function to find purchase by subscription ID
  console.log(`[Stripe Webhook] Subscription ${subscription.id} deleted`);
}
