import os

def generate_aes_key():
    # Generate a random 32-byte key
    key = os.urandom(32)
    print(key.hex())

if __name__ == "__main__":
    generate_aes_key()