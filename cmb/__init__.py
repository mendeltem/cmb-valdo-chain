"""cmb -- cerebral microbleed detection on T2*-weighted MRI.

Clean re-implementation of the experiment scripts in ``../code`` (German, grown during
exploration). Every module here must reproduce the numbers of its predecessor exactly;
the pipeline is deterministic, so "same seed -> same F1" is the acceptance test.
"""


def __getattr__(name):            # the Python API (cmb.api) without importing torch at "import cmb"
    if name == "env":
        from cmb.envfile import env
        return env
    if name in ("get_model", "predict", "preprocess", "train", "fine_tune", "evaluate", "cross_validate"):
        from cmb import api
        return getattr(api, name)
    raise AttributeError(name)
