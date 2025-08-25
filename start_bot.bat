@echo off
REM Fill your secrets in .env (copy .env.example -> .env) or set them below.
setlocal
if exist .env (
  for /f "usebackq tokens=1,2 delims== " %%a in (`type .env`) do (
    if not "%%a"=="" set %%a=%%b
  )
)
python main.py
pause
