@echo off

call git.bat

call cleanup.bat

%git_bash% -c "git add CdS_Base FunctionDrawer Dogodo CdS_Data/Dogodo/Data iPlug2-master/36Common iPlug2-master/36Effects VstRelease iPlug2-master/36Instruments 36Shapes push.bat pull.bat cleanup.bat"

%git_bash% -c "git commit"
%git_bash% -c "git push -u origin master"

pause
