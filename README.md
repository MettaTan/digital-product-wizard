# 🧠 Digital Product Wizard

An AI-powered tool to help you create, remix, and launch digital products faster.

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/MettaTan/digital-product-wizard.git
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

#### Main App Environment Variables

You need 3 keys for the main app:

- OPENAI_API_KEY
- JAMAI_API_KEY
- JAMAI_PROJECT_ID

Create your main .env file:

```bash
cp .env.example .env
```

Then edit .env and paste in your real keys.

#### Stripe & Supabase Environment Variables

You also need 6 keys for Stripe payments and Supabase database:

- SUPABASE_URL
- SUPABASE_KEY (anon public key)
- SUPABASE_SERVICE_ROLE_KEY
- STRIPE_PUBLISHABLE_KEY
- STRIPE_SECRET_KEY
- STRIPE_WEBHOOK_SECRET

Create your Stripe/Supabase .env file:

```bash
cp supabase/functions/stripe-webhook/.env.example supabase/functions/stripe-webhook/.env
```

Then edit `supabase/functions/stripe-webhook/.env` and paste in your real keys.

**How to get these keys:**

1. **Supabase Keys**: Go to [Supabase Dashboard](https://supabase.com/dashboard) → Your Project → Settings → API
   - Copy "Project URL" as `SUPABASE_URL`
   - Copy "anon public" key as `SUPABASE_KEY`
   - Copy "service_role" key as `SUPABASE_SERVICE_ROLE_KEY`

2. **Stripe Keys**: Go to [Stripe Dashboard](https://dashboard.stripe.com/) → Developers → API Keys
   - Copy "Publishable key" as `STRIPE_PUBLISHABLE_KEY`
   - Copy "Secret key" as `STRIPE_SECRET_KEY`
   - For webhook secret: Go to Developers → Webhooks → Create/Edit webhook → Copy "Signing secret" as `STRIPE_WEBHOOK_SECRET`

> 📝 Both .env files are already in .gitignore and won't be committed to Git.

### 4. Install Required System Dependencies

#### 🪟 Windows (FFmpeg)

Whisper (used for transcription) depends on `ffmpeg`.

1. Download the prebuilt binary from: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
2. Click on `ffmpeg-release-full.7z` under "Release builds"
3. Extract the archive using [7-Zip](https://www.7-zip.org/)
4. Locate the `bin/` folder inside the extracted directory (e.g., `C:\ffmpeg\bin`)
5. Add that folder to your System `PATH`:
   - Search "Environment Variables" in Windows
   - Edit the `Path` system variable
   - Click **New** and paste the path to the `bin` folder
6. Restart your terminal/editor and run:

```bash
ffmpeg -version
```

You should see version info if installed correctly.

#### 🪟 Windows (Tesseract-OCR)

To extract text from video frames, `pytesseract` depends on [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract).

1. Download the installer from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install Tesseract (default location: `C:\Program Files\Tesseract-OCR`)
3. Add that folder to your System `PATH`:
   - Search "Environment Variables" in Windows
   - Edit the `Path` system variable
   - Click **New** and add: `C:\Program Files\Tesseract-OCR`
4. Restart your terminal/editor and test with:

```bash
tesseract --version
```

You should see version info if installed correctly.

#### 🌞 macOS

```bash
brew install ffmpeg tesseract
```

#### 🐑 Linux (Debian/Ubuntu)

```bash
sudo apt update && sudo apt install ffmpeg tesseract-ocr
```

### 4.5 Set Up Supabase CLI and Docker (for Stripe Webhook)

To deploy the Stripe webhook to Supabase Edge Functions, you'll need:

- ✅ [Supabase CLI](https://github.com/supabase/cli)
- ✅ [Docker Desktop](https://www.docker.com/products/docker-desktop) (for function bundling)

#### 🪟 Windows

1. **Download Supabase CLI binary**:
   - Go to: [https://github.com/supabase/cli/releases/latest](https://github.com/supabase/cli/releases/latest)
   - Download `supabase-windows-x64.exe`
   - Rename it to `supabase.exe`
   - Move it to a folder in your `PATH` (e.g., `C:\Program Files\Supabase\`)
   - Add that folder to your System `PATH`:
     - Search "Environment Variables" → Edit `Path` → Add: `C:\Program Files\Supabase\`

2. **Install Docker Desktop**:
   - Download: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   - Install with WSL2 backend if prompted
   - Start Docker Desktop (wait for the whale icon to appear)

3. **Verify setup**:
   - Open a new terminal
   - Run: `supabase --version` (should return CLI version)
   - Run: `docker --version` (should return Docker version)

#### 🌞 macOS

```bash
brew install deno
brew install supabase/tap/supabase
brew install --cask docker
```

> Open Docker Desktop after install and ensure it's running.

### 5. Run the Script

```
python script.py
```

### 6. Deploy the Stripe Webhook

In your project folder:

```bash
supabase login
supabase functions deploy stripe-webhook
```

> ✅ Once deployed, your webhook will live on Supabase's servers and run 24/7 — even if your laptop is offline.

### 7. Copy Environment Variables for Local Testing

For local development, copy the Stripe/Supabase environment variables to your main directory:

```bash
cp supabase/functions/stripe-webhook/.env .env.stripe
```

Then load both environment files in your app or merge them into your main `.env` file.

### 8. Run the App

```
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).