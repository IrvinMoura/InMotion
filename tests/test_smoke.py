import kinect
import kinect.input


def test_package_imports() -> None:
    assert kinect.__doc__
    assert kinect.input.__doc__
