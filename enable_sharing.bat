@echo off
REM ====================================================================
REM  Audio Research Assistant -- enable LAN sharing (run ONCE)
REM
REM  HOW TO USE:  right-click this file  ->  "Run as administrator"
REM
REM  It opens Windows Firewall for TCP port 8600 on ALL network
REM  profiles (Private + Public + Domain) so teammates on the same
REM  Wi-Fi/LAN can reach the app at  http://<your-ip>:8600
REM
REM  To undo later, run:
REM     netsh advfirewall firewall delete rule name="Audio Research Assistant 8600"
REM ====================================================================

net session >nul 2>&1
if %errorlevel% neq 0 (
  echo.
  echo  [!] This must be run as Administrator.
  echo      Right-click enable_sharing.bat  ->  "Run as administrator".
  echo.
  pause
  exit /b 1
)

echo Removing any old rule...
netsh advfirewall firewall delete rule name="Audio Research Assistant 8600" >nul 2>&1

echo Adding firewall rule: allow inbound TCP 8600 (all profiles)...
netsh advfirewall firewall add rule name="Audio Research Assistant 8600" dir=in action=allow protocol=TCP localport=8600 profile=any

if %errorlevel%==0 (
  echo.
  echo  [OK] Port 8600 is now allowed through Windows Firewall.
  echo       Next: run  python run.py --share
  echo       Then share  http://192.168.1.8:8600  with your teammate.
) else (
  echo.
  echo  [FAILED] Could not add the rule. Make sure you ran as Administrator.
)
echo.
pause
