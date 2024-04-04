# from streamlit.testing.v1 import AppTest
import sys

sys.path.insert(0, "src/frontend")
from app import main


def test_app_main():
    assert main() is None
