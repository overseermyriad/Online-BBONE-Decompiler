# Online-BBONE-Decompiler
This is a python file that decompiles the animation formats of the popular game Plants vs. Zombies Online (Chinese name: 植物大战僵尸Online) into Adobe Animate XFL format.

The decompiler currently supports all BBONE versions publically available. If a version is not supported, please tell me.

The output XFL is in the same format as PvZ2 XFLs to help animators and modders easily read, edit and implement these animations.

Example usage:
BBONE_to_XFL.py input.bbone

Supported arguments:
--merge-similar: For some reason, there are buggy animations that have 100s of the same sprite that are slightly different. This causes the output XFL to be insanely large in size. Thus, this merges similar sprites to avoid lag.
--separate-layers: This separates the timeline of animations into multiple layers. It is recommended to export with this option enabled.

It comes with 4 batch scripts that process all of the BBONEs in the folder where the batch file is located. The batch file with arguments in their name have said arguments enabled by default.

Required Libraries:

Python

pip install Pillow

pip install numpy


This tool is not perfect, so if there are any bugs, please tell me or someone who can contact me. Good luck.

Special thanks goes to [SproutNan's BBONE Repository](https://github.com/SproutNan/BBone_Decom) as I used code from their repository.
