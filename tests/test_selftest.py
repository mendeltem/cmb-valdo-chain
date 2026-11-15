"""cmb selftest passes in a fresh environment without the original MicrobleedNet package (found 15.11.: the FRST
self-check imported microbleednet for a printed comparison and failed in a fresh install)."""

from conftest import cmb


def test_selftest_ok():
    r = cmb("selftest")
    assert "SELFTEST OK" in r.stdout, r.stdout[-2000:]
