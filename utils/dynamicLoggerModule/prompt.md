I want to specify The behavior of `w` and `a` mode for writing log modes.
`a` mode is just the regular append mode.

but `w` mode Due to other differences in this project. should behave a below:
- the mode on common `logging` package, even with `w` mode of DynamicLogger is `append` (`a`) mode. note modes of writing to a file from DynamicLogger and common `logging` python package is different.
- default is `w` mode
- there should be an internal flag(`_wMode_remove_mainLogFolder`). if one of logCalls has been called, to remove `mainLogFolder` if we have `w` mode, and turn that flag True.
- another variable named `_wMode_remove_logFilesAtBeginning`
    - in the log funcs if mode is `w`, after that what file to logTo has been determined, if that file is not in `_wMode_remove_logFilesAtBeginning` check if that file exists, remove that file, and add the path of that file to `_wMode_remove_logFilesAtBeginning`. note this is intended to apply `w` mode for files which are not in `mainLogFolder` as just removing `mainLogFolder` does not apply `w` mode for these files


side notes:
1. we know the project locates the baseFolder(root folder) of main project and puts `logs/` folder their. from the codes of the project try to understand what folder I am referring to. I call it `mainLogFolder`
2. add `fileWriteMode` to `All Main Log Args`

do u have any questions?