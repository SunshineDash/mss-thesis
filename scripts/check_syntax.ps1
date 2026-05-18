Set-Location 'c:\Users\Sunshine Dash\Desktop\AudioSepAblationStudy'

foreach ($f in Get-ChildItem -Recurse src -Filter *.py) {
    $out = python -m py_compile $f.FullName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host ("OK: " + $f.FullName)
    } else {
        Write-Host ("FAIL: " + $f.FullName)
        Write-Host $out
    }
}
