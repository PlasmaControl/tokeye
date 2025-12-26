def test_import_tokeye():
    """Test that the tokeye package can be imported."""
    import tokeye

    assert tokeye is not None


def test_basic_math():
    """Test basic arithmetic to verify pytest is working."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6

