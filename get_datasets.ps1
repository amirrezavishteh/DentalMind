$datasetsDir = "$(Get-Location)\datasets"
New-Item -ItemType Directory -Path $datasetsDir -Force -ErrorAction SilentlyContinue | Out-Null

Write-Host "Dental Datasets Downloader" -ForegroundColor Cyan

$downloads = @(
    @{ Name="DENTEX Training"; Url="https://zenodo.org/records/7812323/files/training_data.zip"; File="DENTEX_training.zip" },
    @{ Name="DENTEX Test"; Url="https://zenodo.org/records/7812323/files/test_data.zip"; File="DENTEX_test.zip" },
    @{ Name="Dental Caries"; Url="https://zenodo.org/record/4907880/files/dataset.zip"; File="Caries_dataset.zip" }
)

$done = 0
$failed = 0
$skip = 0

foreach ($item in $downloads) {
    $fpath = Join-Path $datasetsDir $item.File
    
    if (Test-Path $fpath) {
        Write-Host "SKIP: $($item.Name)" -ForegroundColor Green
        $skip++
        continue
    }
    
    Write-Host "DOWNLOAD: $($item.Name)..." -ForegroundColor Cyan
    
    if (wget -O $fpath $item.Url 2>&1) {
        if (Test-Path $fpath) {
            $mb = [math]::Round((Get-Item $fpath).Length / 1MB, 1)
            Write-Host "  OK ($mb MB)" -ForegroundColor Green
            $done++
        } else {
            Write-Host "  FAIL" -ForegroundColor Red
            $failed++
        }
    } else {
        Write-Host "  ERROR" -ForegroundColor Red
        $failed++
    }
}

Write-Host ""
Write-Host "Done: $done | Failed: $failed | Skipped: $skip"
Write-Host ""
Write-Host "Extracting..." -ForegroundColor Yellow

Get-ChildItem -Path $datasetsDir -Filter "*.zip" -ErrorAction SilentlyContinue | ForEach-Object {
    $extract = Join-Path $datasetsDir $_.BaseName
    if (-not (Test-Path $extract)) {
        Write-Host "  $($_.BaseName)..." -ForegroundColor Cyan
        Expand-Archive -Path $_.FullName -DestinationPath $extract -ErrorAction SilentlyContinue
        Write-Host "    Done" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Datasets ready at: $datasetsDir" -ForegroundColor Green
Write-Host ""
Write-Host "Manual access needed for:" -ForegroundColor Yellow
Write-Host "  - Tufts: https://tdd.ece.tufts.edu/"
Write-Host "  - Roboflow: https://roboflow.com"
