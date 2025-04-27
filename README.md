# 🧠 Digital Product Wizard

An AI-powered tool to help you create, remix, and launch digital products faster.

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/MettaTan/digital-product-wizard.git
cd digital-product-wizard
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
bash
Copy code
cp .env.example .env
Then edit .env and paste in your real keys.
```

(.env is already in .gitignore.)

### 4. Run the App

```
streamlit run app.py
```

The app will open at http://localhost:8501.
