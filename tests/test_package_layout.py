from pe_research import __version__


def test_package_is_importable() -> None:
    assert __version__


def test_research_domains_are_importable() -> None:
    import pe_research.data
    import pe_research.experiment
    import pe_research.model

    assert pe_research.data.__doc__
    assert pe_research.model.__doc__
    assert pe_research.experiment.__doc__
