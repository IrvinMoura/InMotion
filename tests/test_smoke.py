import inmotion
import inmotion.capture
import inmotion.gestures
import inmotion.input
import inmotion.vision


def test_package_imports() -> None:
    assert inmotion.__doc__
    assert inmotion.input.__doc__


def test_pipeline_packages_import() -> None:
    assert inmotion.vision.__doc__
    assert inmotion.gestures.__doc__
    assert inmotion.capture.__doc__


def test_pyyaml_available() -> None:
    import yaml

    assert yaml.safe_load("a: [1, 2]") == {"a": [1, 2]}
