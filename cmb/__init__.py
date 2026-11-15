"""cmb -- cerebral microbleed detection on T2*-weighted MRI.

Clean re-implementation of the experiment scripts in ``../code`` (German, grown during
exploration). Every module here must reproduce the numbers of its predecessor exactly;
the pipeline is deterministic, so "same seed -> same F1" is the acceptance test.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # what the lazy names below resolve to, for type checkers and editors
    from cmb.api import cross_validate as cross_validate
    from cmb.api import evaluate as evaluate
    from cmb.api import fine_tune as fine_tune
    from cmb.api import get_model as get_model
    from cmb.api import predict as predict
    from cmb.api import preprocess as preprocess
    from cmb.api import train as train
    from cmb.envfile import env as env


def __getattr__(name):  # the Python API (cmb.api) without importing torch at "import cmb"
    if name == "env":
        from cmb.envfile import env

        return env
    if name in ("get_model", "predict", "preprocess", "train", "fine_tune", "evaluate", "cross_validate"):
        from cmb import api

        return getattr(api, name)
    raise AttributeError(name)
