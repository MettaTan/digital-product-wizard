import { eq, and } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { InsertUser, users, products, InsertProduct, modules, InsertModule, assets, InsertAsset, purchases, InsertPurchase } from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

// ===== PRODUCTS =====

export async function createProduct(product: InsertProduct) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const [result] = await db.insert(products).values(product);
  return Number(result.insertId);
}

export async function getProductById(id: number) {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(products).where(eq(products.id, id)).limit(1);
  return result.length > 0 ? result[0] : undefined;
}

export async function getProductsByUserId(userId: number) {
  const db = await getDb();
  if (!db) return [];
  
  return await db.select().from(products).where(eq(products.userId, userId)).orderBy(products.createdAt);
}

export async function updateProduct(id: number, updates: Partial<InsertProduct>) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  // Skip update if no fields to update
  if (Object.keys(updates).length === 0) {
    return;
  }
  
  await db.update(products).set(updates).where(eq(products.id, id));
}

export async function deleteProduct(id: number) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  await db.delete(products).where(eq(products.id, id));
}

// ===== MODULES =====

export async function createModule(module: InsertModule) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const [result] = await db.insert(modules).values(module);
  return Number(result.insertId);
}

export async function getModulesByProductId(productId: number) {
  const db = await getDb();
  if (!db) return [];
  
  return await db.select().from(modules).where(eq(modules.productId, productId)).orderBy(modules.order);
}

export async function updateModule(id: number, updates: Partial<InsertModule>) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  // Skip update if no fields to update
  if (Object.keys(updates).length === 0) {
    return;
  }
  
  await db.update(modules).set(updates).where(eq(modules.id, id));
}

export async function deleteModule(id: number) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  await db.delete(modules).where(eq(modules.id, id));
}

// ===== ASSETS =====

export async function createAsset(asset: InsertAsset) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const [result] = await db.insert(assets).values(asset);
  return Number(result.insertId);
}

export async function getAssetsByProductId(productId: number) {
  const db = await getDb();
  if (!db) return [];
  
  return await db.select().from(assets).where(eq(assets.productId, productId));
}

export async function updateAsset(id: number, updates: Partial<InsertAsset>) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  await db.update(assets).set(updates).where(eq(assets.id, id));
}

export async function deleteAsset(id: number) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  await db.delete(assets).where(eq(assets.id, id));
}

// ===== PURCHASES =====

export async function createPurchase(purchase: InsertPurchase) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const [result] = await db.insert(purchases).values(purchase);
  return Number(result.insertId);
}

export async function getPurchasesByUserId(userId: number) {
  const db = await getDb();
  if (!db) return [];
  
  return await db.select().from(purchases).where(eq(purchases.userId, userId));
}

export async function getPurchaseByUserAndProduct(userId: number, productId: number) {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(purchases)
    .where(and(
      eq(purchases.userId, userId),
      eq(purchases.productId, productId)
    ))
    .limit(1);
  
  return result.length > 0 ? result[0] : undefined;
}

export async function updatePurchase(id: number, updates: Partial<InsertPurchase>) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  await db.update(purchases).set(updates).where(eq(purchases.id, id));
}
