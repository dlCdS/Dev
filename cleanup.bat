@echo off

set loc_dir="E:\Programmes\VS2017"

set folder_to_find=x64,ipch,.vs
set folder_to_find2=Debug,Release
set files_to_find= *.ipch,*.hint,*.aps,*.lib

echo Deleting common compilation folders:
for /d /r "%loc_dir%" %%i in (%folder_to_find%) do (
	if exist "%%i" (
		echo %%i
		rmdir /Q /S %%i
	)
)

echo Deleting specific compilation folders:
for /d /r "%loc_dir%" %%i in (%folder_to_find2%) do (
	if exist "%%i" (
		echo %%i
		rmdir /Q /S %%i
	)
)

echo Deleting files
for /r "%loc_dir%" %%i in (%files_to_find%) do (
	if exist "%%i" (
		echo %%i
		del /q %%i
	)
)

pause