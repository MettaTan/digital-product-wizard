import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { TRPCError } from "@trpc/server";
import { z } from "zod";
import {
  getProductsByUserId,
  getProductById,
  createProduct,
  updateProduct,
  deleteProduct,
  getModulesByProductId,
  createModule,
  getAssetsByProductId,
  createAsset,
  getPurchasesByUserId,
} from "./db";
import { createCheckoutSession } from "./stripe";
import {
  generateCourseOutline,
  generateModuleContent,
  generateAssets,
  type CourseOutlineModule,
} from "./aiGenerator";

export const appRouter = router({
    // if you need to use socket.io, read and register route in server/_core/index.ts, all api should start with '/api/' so that the gateway can route correctly
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  // Product management and AI generation
  products: router({
    // List user's products
    list: protectedProcedure.query(async ({ ctx }) => {
      return await getProductsByUserId(ctx.user.id);
    }),

    // Get single product with modules and assets
    getById: protectedProcedure
      .input(z.object({ id: z.number() }))
      .query(async ({ input }) => {
        const product = await getProductById(input.id);
        if (!product) throw new TRPCError({ code: 'NOT_FOUND' });
        
        const modules = await getModulesByProductId(input.id);
        const assets = await getAssetsByProductId(input.id);
        
        return { product, modules, assets };
      }),

    // Create new product (draft)
    create: protectedProcedure
      .input(z.object({
        title: z.string().min(1),
        description: z.string().optional(),
        niche: z.string().min(1),
        targetAudience: z.string().min(1),
      }))
      .mutation(async ({ ctx, input }) => {
        const productId = await createProduct({
          userId: ctx.user.id,
          title: input.title,
          description: input.description,
          niche: input.niche,
          targetAudience: input.targetAudience,
          status: 'draft',
        });
        
        return { productId };
      }),

    // Generate course outline
    generateOutline: protectedProcedure
      .input(z.object({
        productId: z.number(),
      }))
      .mutation(async ({ input }) => {
        const product = await getProductById(input.productId);
        if (!product) throw new TRPCError({ code: 'NOT_FOUND' });
        
        const outline = await generateCourseOutline({
          productTitle: product.title,
          niche: product.niche || '',
          targetAudience: product.targetAudience || '',
          description: product.description || undefined,
        });
        
        // Save outline to product
        await updateProduct(input.productId, {
          courseOutline: JSON.stringify(outline.modules),
        });
        
        return outline;
      }),

    // Generate all module content
    generateModules: protectedProcedure
      .input(z.object({
        productId: z.number(),
      }))
      .mutation(async ({ input }) => {
        const product = await getProductById(input.productId);
        if (!product || !product.courseOutline) {
          throw new TRPCError({ code: 'BAD_REQUEST', message: 'Product outline not found' });
        }
        
        const outline: CourseOutlineModule[] = JSON.parse(product.courseOutline);
        
        // Generate content for each module
        const moduleIds: number[] = [];
        for (let i = 0; i < outline.length; i++) {
          const moduleOutline = outline[i];
          
          const content = await generateModuleContent({
            productTitle: product.title,
            niche: product.niche || '',
            moduleTitle: moduleOutline.title,
            moduleDescription: moduleOutline.description,
            learningObjectives: moduleOutline.learningObjectives,
            moduleNumber: i + 1,
            totalModules: outline.length,
          });
          
          const moduleId = await createModule({
            productId: input.productId,
            order: i,
            title: moduleOutline.title,
            description: moduleOutline.description,
            onScreenDoc: content.onScreenDoc,
            script: content.script,
            estimatedDuration: content.estimatedDuration,
          });
          
          moduleIds.push(moduleId);
        }
        
        return { moduleIds, count: moduleIds.length };
      }),

    // Generate assets
    generateAssets: protectedProcedure
      .input(z.object({
        productId: z.number(),
      }))
      .mutation(async ({ input }) => {
        const product = await getProductById(input.productId);
        if (!product || !product.courseOutline) {
          throw new TRPCError({ code: 'BAD_REQUEST', message: 'Product outline not found' });
        }
        
        const outline: CourseOutlineModule[] = JSON.parse(product.courseOutline);
        
        const generatedAssets = await generateAssets({
          productTitle: product.title,
          niche: product.niche || '',
          targetAudience: product.targetAudience || '',
          courseOutline: outline,
        });
        
        // Save assets to database
        const assetIds: number[] = [];
        for (const asset of generatedAssets) {
          const assetId = await createAsset({
            productId: input.productId,
            title: asset.title,
            type: asset.type,
            content: asset.content,
          });
          assetIds.push(assetId);
        }
        
        return { assetIds, count: assetIds.length };
      }),

    // Update product
    update: protectedProcedure
      .input(z.object({
        id: z.number(),
        title: z.string().optional(),
        description: z.string().optional(),
        status: z.enum(['draft', 'published', 'archived']).optional(),
        communityPlatform: z.string().optional(),
        communityLink: z.string().optional(),
        priceMonthly: z.number().optional(),
        priceYearly: z.number().optional(),
        priceLifetime: z.number().optional(),
      }))
      .mutation(async ({ input }) => {
        const { id, ...updates } = input;
        await updateProduct(id, updates);
        return { success: true };
      }),

    // Delete product
    delete: protectedProcedure
      .input(z.object({ id: z.number() }))
      .mutation(async ({ input }) => {
        await deleteProduct(input.id);
        return { success: true };
      }),

    // Create checkout session
    createCheckout: protectedProcedure
      .input(z.object({
        productId: z.number(),
        tier: z.enum(["monthly", "yearly", "lifetime"]),
      }))
      .mutation(async ({ ctx, input }) => {
        const product = await getProductById(input.productId);
        if (!product) throw new TRPCError({ code: 'NOT_FOUND' });
        
        // Get price based on tier
        let priceInCents: number;
        if (input.tier === "monthly" && product.priceMonthly) {
          priceInCents = product.priceMonthly;
        } else if (input.tier === "yearly" && product.priceYearly) {
          priceInCents = product.priceYearly;
        } else if (input.tier === "lifetime" && product.priceLifetime) {
          priceInCents = product.priceLifetime;
        } else {
          throw new TRPCError({ code: 'BAD_REQUEST', message: 'Price not set for this tier' });
        }
        
        const checkoutUrl = await createCheckoutSession({
          productId: product.id,
          productTitle: product.title,
          tier: input.tier,
          priceInCents,
          userId: ctx.user.id,
          userEmail: ctx.user.email || '',
          userName: ctx.user.name || undefined,
          origin: ctx.req.headers.origin || 'http://localhost:3000',
        });
        
        return { checkoutUrl };
      }),

    // Get user's purchases
    myPurchases: protectedProcedure.query(async ({ ctx }) => {
      return await getPurchasesByUserId(ctx.user.id);
    }),
  }),

  // TODO: add more feature routers here, e.g.
  // todo: router({
  //   list: protectedProcedure.query(({ ctx }) =>
  //     db.getUserTodos(ctx.user.id)
  //   ),
  // }),
});

export type AppRouter = typeof appRouter;
