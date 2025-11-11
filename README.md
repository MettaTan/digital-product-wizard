# Digital Product Wizard

Your AI-powered assistant for turning ideas into income. Create and sell complete digital product packages in minutes — no tech skills needed.

## Features

### 🎯 AI-Powered Product Creation
- **Smart Course Outline Generation**: AI creates comprehensive course structures based on your niche and target audience
- **Module Content Generation**: Each module includes:
  - **On-Screen Document**: Formatted content ready for video recording (like Loom-style presentations)
  - **Narration Script**: Complete script with hooks, cues, and prompts for anecdotes
  - **Estimated Duration**: Time estimates for each module
- **Asset & Framework Generation**: Automatically creates worksheets, templates, checklists, and guides
- **Community Setup Helper**: Guidance for setting up paid communities

### 💰 Flexible Pricing & Payments
- **Multiple Pricing Tiers**: Offer monthly, yearly, or lifetime access
- **Stripe Integration**: Secure payment processing with automatic access provisioning
- **Subscription Management**: Handle recurring payments and cancellations
- **Promotional Codes**: Built-in support for discount codes

### 🛍️ Marketplace & Customer Portal
- **Public Marketplace**: Customers can browse and purchase published products
- **Secure Access Control**: Only paying customers can access content
- **Content Delivery**: Clean interface for viewing modules and downloading assets
- **Purchase Management**: Track customer purchases and access

### 👨‍💼 Creator Dashboard
- **Product Management**: Create, edit, and publish digital products
- **Draft System**: Save work-in-progress products
- **Export Functionality**: Download modules and assets for external use
- **Status Tracking**: Monitor draft vs. published products

## Getting Started

### For Creators

1. **Sign In**: Click "Get Started Free" on the homepage
2. **Create Product**: 
   - Go to Dashboard → "Create New Product"
   - Enter product details (title, niche, target audience)
   - AI generates course outline automatically
3. **Generate Content**:
   - Review and approve the AI-generated outline
   - Generate all module content (docs + scripts)
   - Generate complementary assets and frameworks
4. **Set Pricing**: Configure monthly, yearly, or lifetime pricing
5. **Publish**: Review and publish your product to the marketplace

### For Customers

1. **Browse Marketplace**: View all published digital products
2. **Purchase**: Choose your preferred pricing tier (monthly/yearly/lifetime)
3. **Access Content**: After payment, access your purchased products
4. **Learn**: View course modules, download assets, and track progress

## Tech Stack

- **Frontend**: React 19 + TypeScript + Tailwind CSS 4
- **Backend**: Express + tRPC + Node.js
- **Database**: MySQL (via Drizzle ORM)
- **AI**: OpenAI GPT-4 for content generation
- **Payments**: Stripe for checkout and subscriptions
- **Authentication**: Manus OAuth

## Key Workflows

### Product Creation Flow
1. Enter product details → 2. AI generates outline → 3. Generate modules → 4. Generate assets → 5. Set pricing → 6. Publish

### Purchase Flow
1. Browse marketplace → 2. Select product & tier → 3. Stripe checkout → 4. Automatic access provisioning → 5. Access content

### Content Structure

Each course module includes:
- **Title & Description**: Clear learning objectives
- **On-Screen Document**: Markdown-formatted content designed for screen recording
- **Narration Script**: Detailed script with:
  - Opening hooks
  - Key talking points
  - Prompts for personal anecdotes
  - Transition cues
  - Closing statements

## Stripe Setup

### Test Mode
- Use test card: `4242 4242 4242 4242`
- Any future expiry date and CVC
- Webhook endpoint: `/api/stripe/webhook`

### Going Live
1. Complete Stripe KYC verification
2. Update keys in Settings → Payment
3. Test with 99% discount promo code (minimum $0.50 order)

## Database Schema

### Products
- Basic info (title, description, niche, target audience)
- Pricing tiers (monthly, yearly, lifetime in cents)
- Status (draft, published, archived)

### Modules
- Linked to products
- On-screen document content
- Narration script
- Order and duration

### Assets
- Linked to products
- Type (worksheet, template, checklist, guide)
- Markdown content

### Purchases
- User-product relationship
- Tier and status
- Stripe IDs for payment tracking

## API Endpoints

### tRPC Procedures
- `products.create`: Create new product
- `products.list`: Get user's products
- `products.getById`: Get product with modules and assets
- `products.generateOutline`: AI course outline generation
- `products.generateModules`: AI module content generation
- `products.generateAssets`: AI asset generation
- `products.createCheckout`: Create Stripe checkout session
- `products.myPurchases`: Get user's purchases

### Webhooks
- `/api/stripe/webhook`: Handle Stripe payment events

## Environment Variables

All required environment variables are automatically configured:
- `STRIPE_SECRET_KEY`: Stripe API key
- `STRIPE_WEBHOOK_SECRET`: Webhook signing secret
- `VITE_STRIPE_PUBLISHABLE_KEY`: Frontend Stripe key
- `DATABASE_URL`: Database connection
- `BUILT_IN_FORGE_API_KEY`: AI service key

## Development

```bash
# Install dependencies
pnpm install

# Run dev server
pnpm dev

# Push database schema
pnpm db:push
```

## Deployment

1. Create a checkpoint in the dashboard
2. Click "Publish" button in the management UI
3. Configure custom domain (optional)
4. Update Stripe webhook URL to production endpoint

## Next Steps

See `todo.md` for planned features and improvements.

## Support

For issues with Stripe integration, check Settings → Payment in the management UI.
