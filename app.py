import os
import sys

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
