$ErrorActionPreference = 'Stop'
$source = 'D:\SET50_direction_prediction_paper\paper\newest_academic_journal_style.docx'
$copy = 'D:\SET50_direction_prediction_paper\paper\qa\journal_pagination_input.docx'
$output = 'D:\SET50_direction_prediction_paper\paper\newest_academic_journal_style_final.docx'
$wdSectionBreakContinuous = 3
$wdAlignParagraphCenter = 1
$wdFormatDocumentDefault = 16
$msoFalse = 0
Copy-Item -LiteralPath $source -Destination $copy -Force
if (Test-Path -LiteralPath $output) { Remove-Item -LiteralPath $output -Force }
$word=$null;$doc=$null
try {
    $word=New-Object -ComObject Word.Application
    $word.Visible=$false
    $word.DisplayAlerts=0
    $doc=$word.Documents.Open($copy,$false,$false)

    # A continuous break at the document end balances the final two-column reference page.
    $endPosition=$doc.Content.End-1
    $endRange=$doc.Range($endPosition,$endPosition)
    $endRange.InsertBreak($wdSectionBreakContinuous)
    $last=$doc.Sections.Item($doc.Sections.Count)
    $last.PageSetup.TextColumns.SetCount(1)
    try {$last.PageSetup.LineNumbering.Active=$false}catch{}

    # Compact the three mid-paper pipeline diagrams so full-width display sections
    # can occupy the remaining page space instead of forcing large blank areas.
    $targetWidthsCm=@{4=13.0;5=10.5;6=12.5}
    foreach($shapeIndex in $targetWidthsCm.Keys){
        if($doc.InlineShapes.Count -ge $shapeIndex){
            $shape=$doc.InlineShapes.Item([int]$shapeIndex)
            $shape.LockAspectRatio=-1
            $shape.Width=$doc.Application.CentimetersToPoints([double]$targetWidthsCm[$shapeIndex])
        }
    }

    # Remove all inherited page-number fields before rebuilding one linked sequence.
    for($s=1;$s -le $doc.Sections.Count;$s++){
        $sec=$doc.Sections.Item($s)
        $sec.PageSetup.DifferentFirstPageHeaderFooter=$false
        $sec.Headers.Item(1).LinkToPrevious=$false
        $sec.Footers.Item(1).LinkToPrevious=$false
        $sec.Headers.Item(1).Range.Text=''
        $sec.Footers.Item(1).Range.Text=''
    }
    $first=$doc.Sections.Item(1)
    [void]$first.Footers.Item(1).PageNumbers.Add($wdAlignParagraphCenter,$true)
    $first.Footers.Item(1).PageNumbers.RestartNumberingAtSection=$false
    $first.Footers.Item(1).Range.Font.Name='Times New Roman'
    $first.Footers.Item(1).Range.Font.Size=8
    for($s=2;$s -le $doc.Sections.Count;$s++){
        $sec=$doc.Sections.Item($s)
        $sec.Headers.Item(1).LinkToPrevious=$true
        $sec.Footers.Item(1).LinkToPrevious=$true
        $sec.Footers.Item(1).PageNumbers.RestartNumberingAtSection=$false
    }

    $doc.Repaginate()
    $doc.SaveAs2($output,$wdFormatDocumentDefault)
    Write-Output "OUTPUT=$output"
    Write-Output "PAGES=$($doc.ComputeStatistics(2))"
    Write-Output "SECTIONS=$($doc.Sections.Count)"
    Write-Output "FIELDS=$($doc.Fields.Count)"
    Write-Output "OMATHS=$($doc.OMaths.Count)"
}
finally{
    if($doc-ne$null){$doc.Close($false)}
    if($word-ne$null){$word.Quit()}
    if($doc-ne$null){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc)}
    if($word-ne$null){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)}
    [GC]::Collect();[GC]::WaitForPendingFinalizers()
}
