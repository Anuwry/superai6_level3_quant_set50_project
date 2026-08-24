$ErrorActionPreference = 'Stop'

$source = 'D:\SET50_direction_prediction_paper\paper\qa\llm_scope_source_copy.docx'
$working = 'D:\SET50_direction_prediction_paper\paper\qa\llm_scope_working.docx'
$output = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_llm_benchmark_scope_corrected.docx'

if (-not (Test-Path -LiteralPath $source)) {
    throw "Source manuscript not found: $source"
}
Copy-Item -LiteralPath $source -Destination $working -Force
if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Force
}

function Find-TextRange {
    param($Document, [string]$Text)
    $range = $Document.Content.Duplicate
    $find = $range.Find
    $find.ClearFormatting()
    $find.Text = $Text
    $find.Forward = $true
    $find.Wrap = 0
    $find.Format = $false
    $find.MatchCase = $false
    $find.MatchWildcards = $false
    if (-not $find.Execute()) {
        throw "Text not found: $Text"
    }
    return $range
}

function Replace-Span {
    param(
        $Document,
        [string]$StartText,
        [string]$EndText,
        [string]$Replacement
    )
    $startRange = Find-TextRange -Document $Document -Text $StartText
    $searchRange = $Document.Range($startRange.Start, $Document.Content.End)
    $find = $searchRange.Find
    $find.ClearFormatting()
    $find.Text = $EndText
    $find.Forward = $true
    $find.Wrap = 0
    $find.Format = $false
    $find.MatchCase = $false
    $find.MatchWildcards = $false
    if (-not $find.Execute()) {
        throw "End text not found: $EndText"
    }
    $target = $Document.Range($startRange.Start, $searchRange.End)
    $target.Text = $Replacement
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($working, $false, $false)

    # Section 4.3: remove the out-of-scope Local-NLP-versus-Leader comparison and
    # its untested causal explanation. The registered benchmark remains Debate
    # versus equal-call and near-cost LLM controls.
    Replace-Span -Document $document `
        -StartText 'However, the 2023 local classifier still exceeded the Leader' `
        -EndText 'does not replace temporally supervised domain calibration.' `
        -Replacement ''

    $spacingRange = Find-TextRange -Document $document -Text 'inference calls.  Recent finance-agent'
    $spacingRange.Text = 'inference calls. Recent finance-agent'

    # Section 5: retain only the boundary between intrinsic LLM evaluation and
    # unavailable Leader-derived downstream forecasting evidence.
    Replace-Span -Document $document `
        -StartText 'However, the local supervised classifier remained stronger,' `
        -EndText 'locked common cohort.' `
        -Replacement 'An authoritative Leader-derived downstream forecasting arm was not available on the locked common cohort.'

    $document.Repaginate()
    $document.SaveAs2($output, 16)

    $citationCount = 0
    $emptyCitationCount = 0
    $bibliographyCount = 0
    foreach ($field in $document.Fields) {
        if ($field.Code.Text -match 'ZOTERO_ITEM') {
            $citationCount++
            if ([string]::IsNullOrWhiteSpace($field.Result.Text)) {
                $emptyCitationCount++
            }
        }
        if ($field.Code.Text -match 'ZOTERO_BIBL') {
            $bibliographyCount++
        }
    }

    Write-Output "Output=$output"
    Write-Output "Pages=$($document.ComputeStatistics(2))"
    Write-Output "Tables=$($document.Tables.Count)"
    Write-Output "InlineFigures=$($document.InlineShapes.Count)"
    Write-Output "Equations=$($document.OMaths.Count)"
    Write-Output "CitationFields=$citationCount"
    Write-Output "EmptyCitationFields=$emptyCitationCount"
    Write-Output "BibliographyFields=$bibliographyCount"
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
