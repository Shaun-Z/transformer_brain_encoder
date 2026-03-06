def test_src_package_imports():
    import src
    import src.cli
    import src.data
    import src.models
    import src.train

    assert src is not None
