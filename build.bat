@echo off
:restart
title AL-TECHNOLOGIES
setx OPENCV_VIDEOIO_PRIORITY_MSMF 0
echo Building application...
pyuic5 form_RegisterIdentity.ui -o form_RegisterIdentity.py
pyuic5 form_TrainIdentity.ui -o form_TrainIdentity.py
pyuic5 form_EraseIdentity.ui -o form_EraseIdentity.py
pyuic5 altech_gui.ui -o altech_gui.py
pyrcc5 -o resources_rc.py resources.qrc
echo Running...
python cdsone.py %*
if errorlevel 1 (goto restart)
pause