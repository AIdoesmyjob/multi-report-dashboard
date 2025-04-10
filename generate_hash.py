import streamlit_authenticator as stauth
import sys

import bcrypt
import sys

if len(sys.argv) > 1:
    password_to_hash = sys.argv[1]
    # Use bcrypt directly, which is what streamlit-authenticator uses internally
    hashed_password = bcrypt.hashpw(password_to_hash.encode(), bcrypt.gensalt())
    # Print the hashed password as a string
    print(hashed_password.decode())
else:
    print("Usage: python generate_hash.py <password_to_hash>")
