import os
import random

DATA_DIR = "data"
OUTPUT_CSV = os.path.join(DATA_DIR, "phishing.csv")

def generate_synthetic_dataset():
    """Generates a highly realistic synthetic phishing dataset."""
    import pandas as pd
    print("Generating a locally-sourced synthetic dataset to guarantee out-of-the-box execution...")
    
    safe_templates = [
        "Hi {}, just following up on our meeting yesterday. Let me know when you have time to chat.",
        "Here is the report you requested. Please review it before the end of the week.",
        "Don't forget we have a team lunch at 12 PM tomorrow!",
        "Thanks for your purchase! Your order will be shipped soon. You can track it here: http://amazon-tracking.com/123",
        "Hello, your monthly statement is attached. Thank you for banking with us.",
        "Can we reschedule our 3 PM? I have a conflict. How about 4 PM?",
        "Hey, are we still on for dinner tonight?",
        "Attached is the signed contract. Excellent doing business with you.",
        "The project deployment was successful. Check the logs at https://github.com/org/repo/actions",
        "Reminder: Please submit your timesheets by Friday."
    ]

    phishing_templates = [
        "URGENT: Your account has been compromised. Verify your identity immediately at http://secure-login-update-xyz.com",
        "You have won a $1000 Walmart gift card! Click here to claim your prize: http://free-gift-cards-claim.info",
        "Action Required: Update your bank password to avoid account suspension. http://chase-bank-verify-security.org",
        "Dear customer, your PayPal account is restricted. Re-activate it now: http://paypal.com.secure-login-update.com/verify",
        "FINAL NOTICE: Pay your outstanding invoice immediately to avoid legal action. Click to view invoice: http://invoices-urgent.net",
        "We noticed unusual login activity. Please confirm your login details here: http://verify-apple-id.net",
        "Your Netflix subscription has expired. Renew now by clicking this link: http://netflix-billing-update.com",
        "Congratulations! You've been selected for an exclusive offer. Claim here: http://exclusive-rewards-today.com",
        "Security Alert: Someone tried to access your account. Secure it now: http://security-alert-verify.com",
        "Urgent request from the CEO: I need you to purchase 5 Apple gift cards and send me the codes."
    ]

    names = ['John', 'Sarah', 'Mike', 'Emily', 'David', 'Jessica']
    
    data = []
    # Generate 1500 Safe Emails
    for _ in range(1500):
        template = random.choice(safe_templates)
        if "{}" in template:
            text = template.format(random.choice(names))
        else:
            text = template
        data.append({'Email Text': text, 'Email Type': 'Safe Email'})

    # Generate 500 Phishing Emails
    for _ in range(500):
        data.append({'Email Text': random.choice(phishing_templates), 'Email Type': 'Phishing Email'})

    # Shuffle
    random.shuffle(data)

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated {len(df)} emails (Safe and Phishing) and saved to {OUTPUT_CSV}")

def download_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print("Attempting to download dataset from Kaggle API...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files("nabeelsajid917/predict-detect-phishing-emails", path=DATA_DIR, unzip=True)
        print("Successfully downloaded dataset from Kaggle.")
        for f in os.listdir(DATA_DIR):
            if f.endswith('.csv') and f != "phishing.csv":
                os.rename(os.path.join(DATA_DIR, f), OUTPUT_CSV)
                break
    except Exception as e:
        print(f"Kaggle API failed or not configured (Error: {e})")
        generate_synthetic_dataset()

if __name__ == "__main__":
    download_dataset()
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)
    print(f"\nDataset loaded. Shape: {df.shape}")
    print(df.head())
