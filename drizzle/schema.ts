import { int, mysqlEnum, mysqlTable, text, timestamp, varchar } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** Manus OAuth identifier (openId) returned from the OAuth callback. Unique per user. */
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Digital products created by users to sell
 */
export const products = mysqlTable("products", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull().references(() => users.id, { onDelete: "cascade" }),
  
  // Product details
  title: varchar("title", { length: 255 }).notNull(),
  description: text("description"),
  niche: varchar("niche", { length: 255 }),
  targetAudience: text("targetAudience"),
  
  // Course outline (JSON array of module titles/descriptions)
  courseOutline: text("courseOutline"),
  
  // Community settings
  communityPlatform: varchar("communityPlatform", { length: 100 }), // discord, circle, slack
  communityLink: text("communityLink"),
  communityInstructions: text("communityInstructions"),
  
  // Pricing (in cents)
  priceMonthly: int("priceMonthly"),
  priceYearly: int("priceYearly"),
  priceLifetime: int("priceLifetime"),
  
  // Stripe product/price IDs
  stripeProductId: varchar("stripeProductId", { length: 255 }),
  stripePriceIdMonthly: varchar("stripePriceIdMonthly", { length: 255 }),
  stripePriceIdYearly: varchar("stripePriceIdYearly", { length: 255 }),
  stripePriceIdLifetime: varchar("stripePriceIdLifetime", { length: 255 }),
  
  // Status
  status: mysqlEnum("status", ["draft", "published", "archived"]).default("draft").notNull(),
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Product = typeof products.$inferSelect;
export type InsertProduct = typeof products.$inferInsert;

/**
 * Course modules - each module has on-screen document and narration script
 */
export const modules = mysqlTable("modules", {
  id: int("id").autoincrement().primaryKey(),
  productId: int("productId").notNull().references(() => products.id, { onDelete: "cascade" }),
  
  // Module info
  order: int("order").notNull(), // Display order
  title: varchar("title", { length: 255 }).notNull(),
  description: text("description"),
  
  // Content - the key deliverables for video creation
  onScreenDoc: text("onScreenDoc"), // Markdown content to display on screen
  script: text("script"), // Narration script with hooks, cues, anecdotes
  
  // Metadata
  estimatedDuration: int("estimatedDuration"), // Minutes
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Module = typeof modules.$inferSelect;
export type InsertModule = typeof modules.$inferInsert;

/**
 * Assets and frameworks (worksheets, templates, checklists, etc.)
 */
export const assets = mysqlTable("assets", {
  id: int("id").autoincrement().primaryKey(),
  productId: int("productId").notNull().references(() => products.id, { onDelete: "cascade" }),
  
  title: varchar("title", { length: 255 }).notNull(),
  type: mysqlEnum("type", ["worksheet", "template", "checklist", "framework", "guide", "other"]).notNull(),
  description: text("description"),
  
  // Content stored as markdown/text
  content: text("content"),
  
  // Optional file URL if exported to PDF/etc
  fileUrl: text("fileUrl"),
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Asset = typeof assets.$inferSelect;
export type InsertAsset = typeof assets.$inferInsert;

/**
 * Customer purchases - tracks who bought what and their access level
 */
export const purchases = mysqlTable("purchases", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull().references(() => users.id, { onDelete: "cascade" }),
  productId: int("productId").notNull().references(() => products.id, { onDelete: "cascade" }),
  
  // Purchase details
  tier: mysqlEnum("tier", ["monthly", "yearly", "lifetime"]).notNull(),
  status: mysqlEnum("status", ["active", "canceled", "expired", "past_due"]).default("active").notNull(),
  
  // Stripe info
  stripeCustomerId: varchar("stripeCustomerId", { length: 255 }),
  stripeSubscriptionId: varchar("stripeSubscriptionId", { length: 255 }),
  stripePaymentIntentId: varchar("stripePaymentIntentId", { length: 255 }),
  
  // Subscription dates
  currentPeriodStart: timestamp("currentPeriodStart"),
  currentPeriodEnd: timestamp("currentPeriodEnd"),
  
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Purchase = typeof purchases.$inferSelect;
export type InsertPurchase = typeof purchases.$inferInsert;