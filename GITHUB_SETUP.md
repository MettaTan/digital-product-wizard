# GitHub Setup Instructions

## Quick Upload to GitHub

### Method 1: Using GitHub CLI (if authenticated)

```bash
cd /home/ubuntu/digital-product-wizard
gh repo create digital-product-wizard-v2 --public --source=. --remote=origin --push
```

### Method 2: Manual Upload via GitHub Website

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `digital-product-wizard-v2`
   - Description: "AI-powered platform to create and sell digital product packages with course modules, scripts, and assets"
   - Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license
   - Click "Create repository"

2. **Push your code:**
   ```bash
   cd /home/ubuntu/digital-product-wizard
   git remote add origin https://github.com/YOUR_USERNAME/digital-product-wizard-v2.git
   git branch -M main
   git push -u origin main
   ```

### Method 3: Download and Upload

1. **Download the code:**
   - Use the Manus management UI to download all project files
   - Or use the Code panel to download as ZIP

2. **Upload to GitHub:**
   - Create new repository on GitHub
   - Upload files via web interface or use GitHub Desktop

## Repository Details

**Project Name:** Digital Product Wizard  
**Tech Stack:** React 19, TypeScript, Express, tRPC, MySQL, Stripe, OpenAI  
**Features:**
- AI-powered course outline and module generation
- On-screen documents + narration scripts for video creation
- Asset and framework generation
- Stripe payment integration (monthly/yearly/lifetime)
- Customer marketplace and access portal
- Seller dashboard

## Environment Variables

Remember to set up these secrets in your deployment platform:
- `DATABASE_URL` - MySQL connection string
- `STRIPE_SECRET_KEY` - Stripe API key
- `STRIPE_WEBHOOK_SECRET` - Stripe webhook signing secret
- `VITE_STRIPE_PUBLISHABLE_KEY` - Stripe publishable key
- Built-in Manus OAuth and AI keys (auto-configured in Manus platform)

## Next Steps After Upload

1. Update the repository URL in README.md
2. Add GitHub Actions for CI/CD (optional)
3. Set up branch protection rules
4. Configure GitHub Issues for bug tracking
5. Add contributing guidelines if open source
