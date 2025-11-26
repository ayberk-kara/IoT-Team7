def check(passkey):
    try:
        with open("auth.txt", "r") as file:
            valid_passkeys = [line.strip() for line in file]
        return passkey in valid_passkeys
    except FileNotFoundError:
        print("Error: auth.txt file not found.")
        return False