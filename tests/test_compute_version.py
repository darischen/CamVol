from installer.compute_version import next_version


def test_bare_base_tag_counts_as_patch_zero():
    # The repo's existing `v1.0` tag is treated as patch 0, so the first
    # auto-release is 1.0.1 (matches the product owner's expectation).
    assert next_version("1.0", ["v1.0"]) == "1.0.1"


def test_increments_above_highest_patch():
    assert next_version("1.0", ["v1.0", "v1.0.1", "v1.0.2"]) == "1.0.3"


def test_patch_compared_numerically_not_lexically():
    assert next_version("1.0", ["v1.0.9", "v1.0.10"]) == "1.0.11"


def test_no_matching_tags_starts_at_zero():
    assert next_version("1.0", []) == "1.0.0"


def test_fresh_minor_base_starts_at_zero():
    # After bumping VERSION to 1.1, with no v1.1.* tags yet, start at .0.
    assert next_version("1.1", ["v1.0", "v1.0.5"]) == "1.1.0"


def test_other_minor_and_major_tags_are_ignored():
    assert next_version("1.0", ["v1.0.3", "v2.0.0", "v1.1.7"]) == "1.0.4"


def test_handles_whitespace_and_blank_entries():
    assert next_version("1.0", [" v1.0.1 ", "", "v1.0"]) == "1.0.2"
