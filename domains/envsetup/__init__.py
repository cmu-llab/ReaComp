"""Environment-setup domain (SetupBench, EnvBench-Python) for ReaComp.

The domain class lives in ``domains.envsetup.domain`` and is registered lazily by
``domains.__init__``; this package intentionally imports nothing so that
``python -m domains.envsetup.tasks`` and friends run without import cycles.
"""
