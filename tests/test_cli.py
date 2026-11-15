"""Every command answers --help, and the option letters are those of cmb_3d_unet_experiments / MicrobleedNet."""
import pytest

from conftest import cmb

COMMANDS = ["predict", "preprocess", "train", "fine_tune", "evaluate", "cross_validate", "get-model", "selftest", "paths"]


@pytest.mark.parametrize("command", COMMANDS)
def test_help(command):
    r = cmb(command, "--help")
    assert "usage" in r.stdout.lower()


def test_same_letters_as_cmbx():
    from cmb.cli import build_parser
    sub = build_parser()._subparsers._group_actions[0].choices
    flags = lambda c: {s for a in sub[c]._actions for s in a.option_strings}
    assert {"-i", "-l", "-m", "-ep", "-ilr", "-bs"} <= flags("train")
    assert {"-i", "-l", "-m", "-o", "--ft_layers", "--calibrate_only"} <= flags("fine_tune")
    assert {"-i", "-m", "-o"} <= flags("evaluate")
    assert {"-i", "-l", "-o", "-fold", "--stage2"} <= flags("cross_validate")
    assert {"-i", "-ir", "-f", "-b", "-br", "-l", "-lr", "-o"} <= flags("preprocess")
    assert {"-i", "-o", "-m", "--frst", "-frst", "--brain-mask"} <= flags("predict")
    assert not any("swi" in s for s in flags("predict")), "the public model is T2* only: no sequence switch"
