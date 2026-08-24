$ErrorActionPreference = 'Stop'
$source = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_llm_benchmark_scope_corrected.docx'
$copy = 'D:\SET50_direction_prediction_paper\paper\qa\render_input.docx'
$pdf = 'D:\SET50_direction_prediction_paper\paper\qa\newest_original_manuscript_llm_benchmark_scope_corrected.pdf'
Copy-Item -LiteralPath $source -Destination $copy -Force
if (Test-Path -LiteralPath $pdf) { Remove-Item -LiteralPath $pdf -Force }
$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($copy, $false, $true)
    $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 1, 50, 0, $true, $true, 1, $true, $true, $false)
    Write-Output "PDF=$pdf"
}
finally {
    if ($doc -ne $null) { $doc.Close($false) }
    if ($word -ne $null) { $word.Quit() }
    if ($doc -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($word -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
