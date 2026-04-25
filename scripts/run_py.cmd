@echo off
REM Wrapper to run Python with UTF-8 and unbuffered mode on Windows.
REM Usage: scripts\run_py.cmd scripts\validate_release_readiness.py [--args]
py -3 -X utf8 %*
