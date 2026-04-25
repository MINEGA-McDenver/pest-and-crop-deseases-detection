param([Parameter(ValueFromRemainingArguments=$true)]$Args)
# PowerShell wrapper: run via `.
un_py.ps1 scripts\validate_release_readiness.py`
& py -3 -X utf8 @Args
