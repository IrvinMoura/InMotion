import inmotion
import inmotion.input


def test_package_imports() -> None:
    assert inmotion.__doc__
    assert inmotion.input.__doc__
