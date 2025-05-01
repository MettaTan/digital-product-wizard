# auth_app.py
import streamlit as st
import streamlit_authenticator as stauth

# --- Simulated User Database ---
hashed_pw = stauth.Hasher().generate(["password1", "password2"])

users = {
    "user1@gmail.com": {"password": hashed_pw[0], "paid": False},
    "user2@gmail.com": {"password": hashed_pw[1], "paid": True},
}

names = ["User One", "User Two"]
usernames = ["user1", "user2"]
emails = ["user1@gmail.com", "user2@gmail.com"]
hashed_passwords = [users[email]["password"] for email in emails]

credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "email": emails[i],
            "password": hashed_passwords[i],
        } for i in range(len(usernames))
    }
}

# --- Main Authentication function ---
def authenticate_user():
    authenticator = stauth.Authenticate(credentials, "some_cookie_name", "some_signature_key", cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login("Login", "main")
    
    if authentication_status:
        email = credentials["usernames"][username]["email"]
        return name, email, users[email]["paid"], True
    elif authentication_status == False:
        st.error("❌ Incorrect username or password.")
        return None, None, None, False
    else:
        st.warning("Please enter your email and password.")
        return None, None, None, False
