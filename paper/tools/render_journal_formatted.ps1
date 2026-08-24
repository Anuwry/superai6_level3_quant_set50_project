param([string]$Version = 'v1')

$ErrorActionPreference = 'Stop'
$qaDir = "D:\SET50_direction_prediction_paper\paper\qa\journal_formatted_$Version"
$source = "D:\SET50_direction_prediction_paper\paper\SET_direction_manuscript_journal_formatted_$Version.docx"
$renderInput = "$qaDir\render_input.docx"
$pdf = "$qaDir\journal_formatted_$Version.pdf"

New-Item -ItemType Directory -Path $qaDir -Force | Out-Null
Copy-Item -LiteralPath $source -Destination $renderInput -Force

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($renderInput, $false, $true)
    $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 1, 100, 0, $true, $true, 1, $true, $true, $false)
    Write-Output "PDF=$pdf"
    Write-Output "Pages=$($doc.ComputeStatistics(2))"
}
finally {
    if ($doc -ne $null) { $doc.Close($false) }
    if ($word -ne $null) { $word.Quit() }
    if ($doc -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($word -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
