$ErrorActionPreference = 'Stop'

$source = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_revised_figure5_fixed.docx'
$output = 'D:\SET50_direction_prediction_paper\paper\qa\zotero_field_insert_test.docx'
Copy-Item -LiteralPath $source -Destination $output -Force

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$document = $word.Documents.Open($output, $false, $false)
try {
    $sourceField = $null
    foreach ($field in $document.Fields) {
        if ($field.Code.Text -match 'ZOTERO_ITEM' -and $field.Result.Text -eq '(Uthayopas et al., 2025)') {
            $sourceField = $field
            break
        }
    }
    if ($null -eq $sourceField) { throw 'Source Zotero field not found.' }

    $jsonStart = $sourceField.Code.Text.IndexOf('{')
    $payload = $sourceField.Code.Text.Substring($jsonStart) | ConvertFrom-Json
    $payload.citationID = [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $formatted = '(Uthayopas et al., 2025)'
    $payload.properties.formattedCitation = $formatted
    $payload.properties.plainCitation = $formatted
    $json = $payload | ConvertTo-Json -Depth 100 -Compress

    $targetParagraph = $document.Paragraphs.Item(45)
    $target = $targetParagraph.Range.Duplicate
    $target.End = $target.End - 1
    $target.Collapse(0)
    $target.InsertAfter(' ')
    $target.Collapse(0)
    $wholeField = $document.Range($sourceField.Code.Start - 1, $sourceField.Result.End + 1)
    $wholeField.Copy() | Out-Null
    $target.Paste()
    $newField = $targetParagraph.Range.Fields.Item($targetParagraph.Range.Fields.Count)
    $newField.Code.Text = " ADDIN ZOTERO_ITEM CSL_CITATION $json "
    $newField.Result.Text = $formatted

    $document.Save()
    Write-Output "fields=$($document.Fields.Count)"
    Write-Output "new_code=$($newField.Code.Text.Substring(0, [Math]::Min(100, $newField.Code.Text.Length)))"
    Write-Output "new_result=$($newField.Result.Text)"
}
finally {
    $document.Close(0)
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
