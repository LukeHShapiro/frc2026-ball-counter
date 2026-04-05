$ws = New-Object -ComObject WScript.Shell
$desktop = [System.Environment]::GetFolderPath('Desktop')
$sc = $ws.CreateShortcut("$desktop\OmniScouter.lnk")
$sc.TargetPath = 'C:\Users\scout\AppData\Local\Programs\Python\Python313\python.exe'
$sc.Arguments = '"C:\Users\scout\OneDrive\Desktop\ClaudeProd\frc2026-ball-counter\desktop_app.py"'
$sc.WorkingDirectory = 'C:\Users\scout\OneDrive\Desktop\ClaudeProd\frc2026-ball-counter'
$sc.IconLocation = 'C:\Users\scout\OneDrive\Desktop\ClaudeProd\frc2026-ball-counter\assets\icon.ico,0'
$sc.Description = 'OmniScouter - FRC 2026 Match Analysis'
$sc.WindowStyle = 1
$sc.Save()
Write-Host "Shortcut created at $desktop\OmniScouter.lnk"
