; Inno Setup script — compiled by scripts\build_exe.ps1 after PyInstaller.
; Installs the one-folder build to Program Files and adds Start Menu /
; desktop shortcuts. Job data lives in %LOCALAPPDATA%\Upscaler (see
; backend\paths.py), so uninstalling never touches renders.

#define AppName "Upscaler"
#define AppVersion "0.2.0"
#define AppExe "Upscaler.exe"

[Setup]
AppId={{8C4B0B60-7C1E-4E4B-9C4D-2A1F9E0D5A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppName}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist
OutputBaseFilename=UpscalerSetup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#AppExe}
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\Upscaler\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
