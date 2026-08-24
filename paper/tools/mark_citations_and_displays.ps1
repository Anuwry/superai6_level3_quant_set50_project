param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$wdFindStop = 0
$wdColorRed = 255
$wdYellow = 7
$wdNoHighlight = 0

function Find-AllAndFormat {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Text,
        [int]$FontColor = -1,
        [int]$Highlight = -1
    )

    $search = $Document.Content.Duplicate
    $count = 0
    while ($search.Start -lt $Document.Content.End) {
        $find = $search.Find
        $find.ClearFormatting()
        $find.Text = $Text
        $find.Forward = $true
        $find.Wrap = $wdFindStop
        $find.Format = $false
        if (-not $find.Execute()) { break }

        if ($FontColor -ge 0) { $search.Font.Color = $FontColor }
        if ($Highlight -ge 0) { $search.HighlightColorIndex = $Highlight }
        $count++
        $next = $search.End
        $search.SetRange($next, $Document.Content.End)
    }
    return $count
}

$inputResolved = (Resolve-Path -LiteralPath $InputPath).Path
$outputFull = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $outputFull
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}
if (Test-Path -LiteralPath $outputFull) {
    throw "Output already exists; refusing to overwrite: $outputFull"
}
Copy-Item -LiteralPath $inputResolved -Destination $outputFull

$redClaims = @(
    'Date-only observations were interpreted as Stock Exchange of Thailand sessions in Asia/Bangkok, with features available at 17:00 after the normal close.',
    'For 2024-2025, 69,824 deduplicated official SET headlines yielded 4,619 point-in-time article-symbol pairs under six non-overlapping SET50 membership versions, including the March-April 2025 transition.',
    'The operational sentiment model is balanced logistic regression over character TF-IDF features with the target ticker prepended.',
    'Causal rolling VMD adds six variables.',
    'All fits used Adam, mean squared error, 20 epochs, batch size 32, and shuffle = false.',
    'SHAP rankings used Gradient Explainer with 100 training-only background sequences, at most 128 evenly spaced training ranking sequences, 200 samples, deterministic cell seeds, and float32 tensors.',
    'LIME was an explanation-fidelity diagnostic only:',
    'Balanced accuracy is the primary forecasting endpoint.',
    'Holm adjustment controls the registered familywise error rate.',
    'The registered Yahoo Finance source failed its overlap and availability gate, after which a dated deviation accepted the Investing.com historical interface; the 138-row sample from 5 January to 30 July 2026 is therefore source-contingent.'
)

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($outputFull, $false, $false)
    $doc.TrackRevisions = $false

    $redCount = 0
    foreach ($claim in $redClaims) {
        $matches = Find-AllAndFormat -Document $doc -Text $claim -FontColor $wdColorRed
        if ($matches -ne 1) {
            throw "Expected exactly one citation-review match but found ${matches}: $claim"
        }
        $redCount += $matches
    }

    # The latest draft still lacks the actual causal regime-labelling equation
    # and supporting method reference. Mark the section title red so that the
    # omission is visible without inserting new scientific prose into the paper.
    $regimeHeading = '3.7 Causal market regimes and SHAP selection'
    $matches = Find-AllAndFormat -Document $doc -Text $regimeHeading -FontColor $wdColorRed
    if ($matches -ne 1) {
        throw "Expected one regime-method heading match; found $matches"
    }
    $redCount += $matches

    $yellowCount = 0
    foreach ($paragraph in $doc.Paragraphs) {
        $text = $paragraph.Range.Text.Trim()
        if ($text -match '(?i)(Insert\s+Figure|Table\s+Here\s+Later|Figure\s+Here\s+Later|Table\s+for\s+news|Plot\s+Graph\s+data)') {
            $range = $paragraph.Range.Duplicate
            if ($range.End -gt $range.Start) { $range.End = $range.End - 1 }
            $range.HighlightColorIndex = $wdYellow
            $yellowCount++
        }
    }

    $doc.Save()
    $doc.Close()
    $doc = $null
    $word.Quit()
    $word = $null

    [pscustomobject]@{
        Output = $outputFull
        RedCitationReviewMarks = $redCount
        YellowDisplayPlaceholders = $yellowCount
    }
}
finally {
    if ($null -ne $doc) {
        try { $doc.Close($false) } catch {}
    }
    if ($null -ne $word) {
        try { $word.Quit() } catch {}
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
