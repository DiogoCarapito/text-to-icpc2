from streamlit.testing.v1 import AppTest


def test_app():
    at = AppTest.from_file("etl/etl.py", default_timeout=100).run()
    assert not at.exception
