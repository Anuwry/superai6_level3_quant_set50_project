$ErrorActionPreference = 'Stop'

$source = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_revised.docx'
$working = 'D:\SET50_direction_prediction_paper\paper\qa\figure5_clean_working.docx'
$output = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_revised_figure5_fixed.docx'
$figure = 'D:\SET50_direction_prediction_paper\paper\assets\figure5_separated_audits.png'

foreach ($required in @($source, $figure)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

Copy-Item -LiteralPath $source -Destination $working -Force
if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Force
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($working, $false, $false)

    $captionRange = $document.Content.Duplicate
    $find = $captionRange.Find
    $find.ClearFormatting()
    $find.Text = 'Figure 5.'
    $find.Forward = $true
    $find.Wrap = 0
    if (-not $find.Execute()) {
        throw 'Figure 5 caption was not found.'
    }

    $captionParagraph = $captionRange.Paragraphs.Item(1)
    $captionStart = $captionParagraph.Range.Start
    $captionIndex = 0
    for ($i = 1; $i -le $document.Paragraphs.Count; $i++) {
        if ($document.Paragraphs.Item($i).Range.Start -eq $captionStart) {
            $captionIndex = $i
            break
        }
    }
    if ($captionIndex -eq 0) {
        throw 'Figure 5 caption paragraph index was not found.'
    }

    $imageParagraph = $null
    for ($i = $captionIndex - 1; $i -ge [Math]::Max(1, $captionIndex - 5); $i--) {
        $candidate = $document.Paragraphs.Item($i)
        if ($candidate.Range.InlineShapes.Count -gt 0) {
            $imageParagraph = $candidate
            break
        }
    }
    if ($null -eq $imageParagraph) {
        throw 'Figure 5 inline image was not found.'
    }

    while ($imageParagraph.Range.InlineShapes.Count -gt 0) {
        $imageParagraph.Range.InlineShapes.Item(1).Delete()
    }
    $content = $imageParagraph.Range.Duplicate
    $content.End = $content.End - 1
    $content.Text = ''

    $insert = $imageParagraph.Range.Duplicate
    $insert.Collapse(1)
    $shape = $document.InlineShapes.AddPicture($figure, $false, $true, $insert)
    $shape.LockAspectRatio = -1
    $shape.Width = 15.0 * 28.3464567
    $imageParagraph.Alignment = 1
    $imageParagraph.Format.LineSpacingRule = 0
    $imageParagraph.Format.SpaceBefore = 6
    $imageParagraph.Format.SpaceAfter = 3
    $imageParagraph.Format.KeepWithNext = -1

    $document.Repaginate()
    $document.SaveAs2($output, 16)
    Write-Output "Output=$output"
    Write-Output "Pages=$($document.ComputeStatistics(2))"
    Write-Output "Tables=$($document.Tables.Count)"
    Write-Output "InlineFigures=$($document.InlineShapes.Count)"
    Write-Output "Fields=$($document.Fields.Count)"
    Write-Output "Equations=$($document.OMaths.Count)"
}
finally {
    if ($null -ne $document) {
        $document.Close(0)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    }
    if ($null -ne $word) {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
