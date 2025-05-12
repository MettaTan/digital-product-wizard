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

You need 3 keys:

- OPENAI_API_KEY
- JAMAI_API_KEY
- JAMAI_PROJECT_ID

Create your .env:

```
cp .env.example .env
```

Then edit .env and paste in your real keys. (.env is already in .gitignore.)

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

### 5. Run the Script

```
python script.py
```

### 6. Run the App

```
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).
