# auth_kite.py
from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv
load_dotenv()

# Replace with your credentials
api_key = os.getenv("KITE_API_KEY")
api_secret = os.getenv("KITE_API_SECRET")

# Initialize KiteConnect
kite = KiteConnect(api_key=api_key)

# Step 1: Generate login URL
print("Please visit this URL to authorize:")
print(kite.login_url())
print("\nAfter authorization, you'll be redirected to your redirect URL")
print("Copy the 'request_token' from the URL")

# Step 2: After you get the request_token from the redirect URL
request_token = input("\nEnter the request_token: ")

# Step 3: Generate session (access token)
try:
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    
    print(f"\n✅ Access Token: {access_token}")
    print("\nSave this access token to your .env file as KITE_ACCESS_TOKEN")
    print("This token is valid until the end of the trading day")
    
except Exception as e:
    print(f"Error: {e}")