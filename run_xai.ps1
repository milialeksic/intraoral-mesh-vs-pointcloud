$scans = (Get-Content data\processed\split.json -Raw | ConvertFrom-Json).test
New-Item -ItemType Directory -Force -Path logs | Out-Null

foreach ($scan in $scans) {
    foreach ($graph in "mesh", "knn") {
        foreach ($target in "region", "boundary") {
            Write-Host "=== $scan  $graph  $target ==="
            python -u src\explain.py --run runs\$graph --scan $scan --target $target --steps 256 |
                Tee-Object -FilePath logs\explain_all.log -Append
        }
    }
}