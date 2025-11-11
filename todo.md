# Digital Product Wizard - TODO

## Phase 1: Database Schema & Backend
- [x] Design and implement products table with pricing tiers
- [x] Design and implement course modules table with on-screen docs and scripts
- [x] Design and implement assets table for frameworks and downloadable resources
- [x] Design and implement purchases table for customer access tracking
- [x] Create database helper functions for product CRUD operations
- [x] Create database helper functions for module management
- [x] Create database helper functions for asset management
- [x] Create database helper functions for purchase tracking

## Phase 2: AI Content Generation Engine
- [x] Setup OpenAI API integration
- [x] Create course outline generation prompt and function
- [x] Create module content generator (on-screen document)
- [x] Create narration script generator with hooks and cues
- [x] Create asset/framework generator (worksheets, templates)
- [x] Create community setup helper
- [x] Implement batch generation for all modules
- [x] Add content regeneration capability

## Phase 3: Product Creation Wizard
- [x] Design wizard UI with multi-step form
- [x] Step 1: Product details (title, niche, target audience)
- [x] Step 2: Course outline generation and review
- [x] Step 3: Module generation with preview
- [x] Step 4: Assets and frameworks generation
- [ ] Step 5: Community setup configuration
- [x] Step 6: Pricing configuration (monthly/yearly/lifetime)
- [x] Step 7: Review and publish
- [x] Add save draft functionality
- [x] Add edit existing product capability

## Phase 4: Customer Portal
- [x] Build customer dashboard for purchased products
- [x] Create product access page with module viewer
- [x] Implement module content display (on-screen doc + script)
- [x] Create asset download center
- [ ] Add community access link display
- [ ] Implement subscription status display
- [ ] Add Stripe customer portal integration

## Phase 5: Stripe Payment Integration
- [x] Setup Stripe API integration
- [x] Create product pricing in Stripe
- [x] Implement checkout flow for monthly subscription
- [x] Implement checkout flow for yearly subscription
- [x] Implement checkout flow for lifetime access
- [x] Create webhook handler for payment success
- [x] Implement automatic access provisioning
- [x] Add subscription management endpoints
- [ ] Handle subscription cancellation

## Phase 6: Seller Dashboard
- [x] Create seller product list page
- [ ] Add product analytics (views, purchases, revenue)
- [ ] Implement customer management view
- [x] Add product editing capability
- [ ] Create sales reporting

## Phase 7: Polish & Testing
- [x] Test complete product creation flow
- [x] Test payment flows for all pricing tiers
- [x] Test customer access and portal
- [x] Add loading states and error handling
- [x] Implement responsive design
- [x] Add user feedback and notifications
- [x] Create user documentation

## Bug Fixes
- [x] Fix "Failed to fetch" error on /create page when creating products
- [x] Fix wizard stuck on "Generate Assets" step with no content showing
- [x] Fix "No values to set" error during product creation/update
