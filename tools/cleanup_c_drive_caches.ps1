$ErrorActionPreference = 'Continue'

$freeBefore = (Get-PSDrive -Name C).Free
$targets = @(
    'C:\Users\narak\AppData\Local\NVIDIA\DXCache',
    'C:\Users\narak\AppData\Local\NVIDIA\GLCache',
    'C:\Users\narak\AppData\Local\Temp',
    'C:\Windows\Temp',
    'C:\Users\narak\AppData\Local\npm-cache',
    'C:\Users\narak\AppData\Local\pip\Cache',
    'C:\Users\narak\AppData\Local\Google\Chrome\User Data\Default\Cache',
    'C:\Users\narak\AppData\Local\Google\Chrome\User Data\Default\Code Cache',
    'C:\Users\narak\AppData\Local\Microsoft\Edge\User Data\Default\Cache',
    'C:\Users\narak\AppData\Local\Microsoft\Edge\User Data\Default\Code Cache',
    'C:\Users\narak\AppData\Local\CrashDumps',
    'C:\Users\narak\.cache\huggingface',
    'C:\Users\narak\.cache\torch'
)

$rows = @()
foreach ($target in $targets) {
    if (-not (Test-Path -LiteralPath $target)) {
        continue
    }

    $resolved = [System.IO.Path]::GetFullPath($target)
    if ($resolved -ne $target) {
        throw "Unexpected target resolution: $target -> $resolved"
    }

    $sizeBefore = (Get-ChildItem -LiteralPath $resolved -Force -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum

    Get-ChildItem -LiteralPath $resolved -Force -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    $sizeAfter = (Get-ChildItem -LiteralPath $resolved -Force -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum

    $rows += [pscustomobject]@{
        Path = $resolved
        RemovedGB = [math]::Round(($sizeBefore - $sizeAfter) / 1GB, 3)
        RemainingGB = [math]::Round($sizeAfter / 1GB, 3)
    }
}

$freeAfter = (Get-PSDrive -Name C).Free
$rows | Sort-Object RemovedGB -Descending | Format-Table -AutoSize
[pscustomobject]@{
    FreeBeforeGB = [math]::Round($freeBefore / 1GB, 3)
    FreeAfterGB = [math]::Round($freeAfter / 1GB, 3)
    NetFreedGB = [math]::Round(($freeAfter - $freeBefore) / 1GB, 3)
} | Format-Table -AutoSize
