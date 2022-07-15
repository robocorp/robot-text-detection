*** Settings ***
Library  OpenRPA.py

*** Tasks ***
Detect text
    Load image  screenshot.png
    Click Word  TWTallySCPConsole.exe  distance=1
