$ErrorActionPreference = 'Stop'
$outputPath = 'D:\SET50_direction_prediction_paper\paper\qa\eq_collection_test.docx'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $document = $word.Documents.Add()
    $range = $document.Range(0, 0)
    $range.Text = "`ty_t=C_(t+1), d_t=I(C_(t+1)>C_t)`t(1)`r"
    $equationRange = $document.Range(1, 36)
    [void]$document.OMaths.Add($equationRange)
    $document.OMaths.BuildUp()
    $document.SaveAs2($outputPath, 16)
    Write-Output "OMaths=$($document.OMaths.Count)"
}
finally {
    if ($document -ne $null) { $document.Close($false) }
    if ($word -ne $null) { $word.Quit() }
    if ($document -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($word -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
