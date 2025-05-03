import streamlit as st
import streamlit_authenticator as stauth

# Basic user info
names = ["User One", "User Two"]
usernames = ["user1", "user2"]
emails = ["user1@gmail.com", "user2@gmail.com"]
passwords = ["password1", "password2"]

# Hash passwords (individually if needed)
hasher = stauth.Hasher()
hashed_passwords = [hasher.hash(pw) for pw in passwords]

# Credential structure for streamlit-authenticator
credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "email": emails[i],
            "password": hashed_passwords[i],
        } for i in range(len(usernames))
    }
}

# Paid status separated
users = {
    "user1@gmail.com": {"paid": False},
    "user2@gmail.com": {"paid": True},
}

# Authenticator object
authenticator = stauth.Authenticate(
    credentials, "cookie_name", "signature_key", cookie_expiry_days=30
)
