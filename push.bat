@echo off
set git_bash="C:\Program Files\Git\bin\sh.exe"

call cleanup.bat

%git_bash% -c "git add CdS_Base FunctionDrawer Dogodo CdS_Data/Dogodo/Data iPlug2-master/36Common iPlug2-master/36Effects push.bat pull.bat cleanup.bat"
%git_bash% -c "git commit"
%git_bash% -c "git push -u origin master"

pause