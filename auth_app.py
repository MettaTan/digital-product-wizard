# auth_app.py
import streamlit as st
import streamlit_authenticator as stauth

# --- User Authentication ---
# Hash passwords individually
hasher = stauth.Hasher()
# Your version of streamlit-authenticator doesn't support list input
# Hash each password separately
hashed_password1 = hasher.hash("password1")
hashed_password2 = hasher.hash("password2")
hashed_passwords = [hashed_password1, hashed_password2]

# Set up user database
users = {
    "user1@gmail.com": {"password": hashed_passwords[0], "paid": False},
    "user2@gmail.com": {"password": hashed_passwords[1], "paid": True},
}

names = ["User One", "User Two"]
usernames = ["user1", "user2"]
emails = ["user1@gmail.com", "user2@gmail.com"]

# Create credentials dictionary for Streamlit Authenticator
credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "email": emails[i],
            "password": users[emails[i]]["password"],
        } for i in range(len(usernames))
    }
}


# --- Main Authentication function ---
def authenticate_user():
    authenticator = stauth.Authenticate(credentials, "cookie_name", "signature_key", cookie_expiry_days=30)
    result = authenticator.login('main', 'Login')

    if result is None:
        return None, None, None, False

    name, auth_status, username = result

    if auth_status:
        email = credentials["usernames"][username]["email"]
        return name, email, users[email]["paid"], True
    elif auth_status == False:
        st.error("❌ Incorrect username or password.")
        return None, None, None, False
    else:
        st.warning("Please enter your username and password.")
        return None, None, None, False

