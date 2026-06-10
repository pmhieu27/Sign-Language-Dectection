import os
import sys

# Tự động sửa lỗi OpenCV (libgthread-2.0.so.0) trên Streamlit Cloud nếu có
try:
    import cv2
except ImportError:
    import subprocess
    print("Detected missing OpenCV dependencies. Reinstalling opencv-python-headless...", file=sys.stderr)
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.10.0.84"], check=True)
        import cv2
        print("Successfully reinstalled opencv-python-headless!", file=sys.stderr)
    except Exception as e:
        print(f"Failed to reinstall opencv-python-headless: {e}", file=sys.stderr)

# Thiết lập encoding UTF-8 cho stdout/stderr để tránh lỗi hiển thị ký tự Unicode trên Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def main():
    from src.apps.streamlit_app import main as streamlit_main

    streamlit_main()


if __name__ == "__main__":
    main()
