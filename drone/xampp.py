import subprocess


def start_xampp_windows():
    # Path to your XAMPP installation directory
    xampp_path = r"C:\xampp\xampp_start.exe"

    try:
        subprocess.run([xampp_path], check=True)
        print("XAMPP started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start XAMPP: {e}")

if __name__ == "__main__":
    start_xampp_windows()
