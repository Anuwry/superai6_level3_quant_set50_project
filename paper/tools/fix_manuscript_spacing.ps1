param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$wdFindStop = 0
$wdAlignParagraphLeft = 0
$wdAlignParagraphCenter = 1
$wdAlignParagraphJustify = 3
$wdLineSpaceSingle = 0
$wdLineSpace1pt5 = 1
$wdYellow = 7

function Find-TextRange {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $range = $Document.Content.Duplicate
    $find = $range.Find
    $find.ClearFormatting()
    $find.Text = $Text
    $find.Forward = $true
    $find.Wrap = $wdFindStop
    $find.Format = $false
    if (-not $find.Execute()) { return $null }
    return $range.Duplicate
}

function Replace-AllExact {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Old,
        [Parameter(Mandatory = $true)][string]$New
    )
    $search = $Document.Content.Duplicate
    $count = 0
    while ($search.Start -lt $Document.Content.End) {
        $find = $search.Find
        $find.ClearFormatting()
        $find.Text = $Old
        $find.Forward = $true
        $find.Wrap = $wdFindStop
        $find.Format = $false
        if (-not $find.Execute()) { break }
        $start = $search.Start
        $search.Text = $New
        $count++
        $next = $start + $New.Length
        $search.SetRange($next, $Document.Content.End)
    }
    return $count
}

function Insert-PeriodAtCitationTransition {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Transition
    )
    $search = $Document.Content.Duplicate
    $count = 0
    while ($search.Start -lt $Document.Content.End) {
        $find = $search.Find
        $find.ClearFormatting()
        $find.Text = $Transition
        $find.Forward = $true
        $find.Wrap = $wdFindStop
        $find.Format = $false
        if (-not $find.Execute()) { break }
        $citationBoundary = $Transition.LastIndexOf(') ')
        if ($citationBoundary -lt 0) { throw "Citation transition has no closing boundary: $Transition" }
        $insertAt = $search.Start + $citationBoundary + 1
        $Document.Range($insertAt, $insertAt).Text = '.'
        $count++
        $search.SetRange($insertAt + 1, $Document.Content.End)
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

$citationTransitions = @(
    @('(Fama, 1970; Olorunnimbe & Viktor, 2023) Deep', '(Fama, 1970; Olorunnimbe & Viktor, 2023). Deep'),
    @('(Hochreiter & Schmidhuber, 1997; Qin et al., 2017; Vaswani et al., 2017) Representative', '(Hochreiter & Schmidhuber, 1997; Qin et al., 2017; Vaswani et al., 2017). Representative'),
    @('(Fischer & Krauss, 2018; Hoseinzade & Haratizadeh, 2019) That', '(Fischer & Krauss, 2018; Hoseinzade & Haratizadeh, 2019). That'),
    @('(Olorunnimbe & Viktor, 2023; Sezer et al., 2020) Earlier', '(Olorunnimbe & Viktor, 2023; Sezer et al., 2020). Earlier'),
    @('(Luo et al., 2023; Sawhney et al., 2020; H. Wang et al., 2020) More', '(Luo et al., 2023; Sawhney et al., 2020; H. Wang et al., 2020). More'),
    @('(W.-J. Liu et al., 2024; M. Wang et al., 2024) Signal', '(W.-J. Liu et al., 2024; M. Wang et al., 2024). Signal'),
    @('(Hochreiter & Schmidhuber, 1997) Their', '(Hochreiter & Schmidhuber, 1997). Their'),
    @('(Qin et al., 2017; Vaswani et al., 2017) Large', '(Qin et al., 2017; Vaswani et al., 2017). Large'),
    @('(Hoseinzade & Haratizadeh, 2019) However', '(Hoseinzade & Haratizadeh, 2019). However'),
    @('(Dragomiretskiy & Zosso, 2014) In', '(Dragomiretskiy & Zosso, 2014). In'),
    @('(T. Liu et al., 2022; Y. Liu et al., 2024) The', '(T. Liu et al., 2022; Y. Liu et al., 2024). The'),
    @('(T. Liu et al., 2022) We', '(T. Liu et al., 2022). We'),
    @('(Luo et al., 2023; Sawhney et al., 2020; H. Wang et al., 2020) Recent', '(Luo et al., 2023; Sawhney et al., 2020; H. Wang et al., 2020). Recent'),
    @('(W.-J. Liu et al., 2024; M. Wang et al., 2024) An', '(W.-J. Liu et al., 2024; M. Wang et al., 2024). An'),
    @('(Du et al., 2024; Liang et al., 2024) Self', '(Du et al., 2024; Liang et al., 2024). Self'),
    @('(X. Wang et al., 2023) Comparing', '(X. Wang et al., 2023). Comparing'),
    @('(Pagan & Sossounov, 2003) Trend', '(Pagan & Sossounov, 2003). Trend'),
    @('(Ribeiro et al., 2016) Neither', '(Ribeiro et al., 2016). Neither'),
    @('(Adebayo et al., 2018) Finance', '(Adebayo et al., 2018). Finance'),
    @('(Yeo et al., 2025) We', '(Yeo et al., 2025). We'),
    @('(Bergmeir et al., 2018) Financial', '(Bergmeir et al., 2018). Financial'),
    @('(Arnott et al., 2019; Olorunnimbe & Viktor, 2023) We', '(Arnott et al., 2019; Olorunnimbe & Viktor, 2023). We'),
    @('(Brodersen et al., 2010) Families', '(Brodersen et al., 2010). Families'),
    @('(Holm, 1979) Economic', '(Holm, 1979). Economic'),
    @('(Uthayopas et al., 2025) Exclude', '(Uthayopas et al., 2025). Exclude'),
    @('(Investing.com, n.d.-a, n.d.-b) The', '(Investing.com, n.d.-a, n.d.-b). The')
)

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($outputFull, $false, $false)
    $doc.TrackRevisions = $false

    # The 2.2 heading was joined to the previous paragraph with a manual line
    # break. Convert only that boundary into a true paragraph boundary.
    $heading22 = Find-TextRange -Document $doc -Text '2.2 Denoising and multimodal financial information'
    if ($null -eq $heading22) { throw 'Could not find the embedded 2.2 heading' }
    $previousCharacter = $doc.Range($heading22.Start - 1, $heading22.Start)
    if ($previousCharacter.Text -eq [string][char]11) {
        $previousCharacter.Text = [string][char]13
    } elseif ($previousCharacter.Text -ne [string][char]13) {
        $doc.Range($heading22.Start, $heading22.Start).InsertBefore([string][char]13)
    }

    # Remaining manual line breaks are layout artifacts, not semantic breaks.
    $manualBreaksRemoved = Replace-AllExact -Document $doc -Old '^l' -New ' '

    # Safe textual spacing and punctuation corrections.
    $textFixCount = 0
    foreach ($pair in @(
        @('pre 2022 model selection', 'pre-2022 model selection'),
        @('Casual rolling variational mode decomposition', 'Causal rolling variational mode decomposition'),
        @('without Holm adjusted significance', 'without Holm-adjusted significance'),
        @('Balanced accuracy the mean of class-wise recalls is', 'Balanced accuracy, the mean of class-wise recalls, is'),
        @('3 Materials and Methods', '3. Materials and Methods'),
        @('4 Results and Discussion', '4. Results and Discussion'),
        @('5 Conclusion', '5. Conclusion')
    )) {
        $textFixCount += Replace-AllExact -Document $doc -Old $pair[0] -New $pair[1]
    }

    # Normalize repeated spaces before matching citation-to-sentence
    # transitions; the source contains one such double-space boundary.
    for ($pass = 0; $pass -lt 5; $pass++) {
        $replaced = Replace-AllExact -Document $doc -Old '  ' -New ' '
        if ($replaced -eq 0) { break }
        $textFixCount += $replaced
    }

    $citationFixCount = 0
    foreach ($pair in $citationTransitions) {
        $matches = Replace-AllExact -Document $doc -Old $pair[0] -New $pair[1]
        if ($matches -ne 1) {
            throw "Expected one citation-transition match but found ${matches}: $($pair[0])"
        }
        # Replacing a mixed-format Word range can inherit formatting from its
        # first run. Force only the citation itself bold and the following
        # sentence word back to regular weight.
        $transitionRange = Find-TextRange -Document $doc -Text $pair[1]
        if ($null -eq $transitionRange) { throw "Could not re-open corrected transition: $($pair[1])" }
        $closeIndex = $pair[1].LastIndexOf(')')
        $doc.Range($transitionRange.Start, $transitionRange.Start + $closeIndex + 1).Font.Bold = -1
        $doc.Range($transitionRange.Start + $closeIndex + 1, $transitionRange.End).Font.Bold = 0
        $citationFixCount += $matches
    }

    # Add terminal punctuation when a prose paragraph ends directly with an
    # author-year citation. This also covers citations followed by a paragraph
    # mark rather than by the next sentence in the same Word paragraph.
    $terminalCitationFixCount = 0
    foreach ($paragraph in $doc.Paragraphs) {
        $paragraphText = $paragraph.Range.Text.TrimEnd([char]13, [char]7)
        if (
            $paragraphText.EndsWith(')') -and
            ($paragraphText -match '\((?:[^()]|\([^()]*\))*(?:(?:19|20)\d{2}|n\.d\.)[^()]*\)$')
        ) {
            $insertAt = $paragraph.Range.End - 1
            if ($paragraph.Range.Text.EndsWith([string][char]7)) { $insertAt-- }
            $doc.Range($insertAt, $insertAt).Text = '.'
            $terminalCitationFixCount++
        }
    }
    $citationFixCount += $terminalCitationFixCount

    # Collapse accidental repeated spaces created by removing manual breaks.
    for ($pass = 0; $pass -lt 5; $pass++) {
        $replaced = Replace-AllExact -Document $doc -Old '  ' -New ' '
        if ($replaced -eq 0) { break }
        $textFixCount += $replaced
    }

    # Locate Abstract after text cleanup, then remove empty spacer paragraphs in
    # the scholarly body. The author/title block above Abstract is preserved.
    $abstractRange = Find-TextRange -Document $doc -Text 'Abstract'
    if ($null -eq $abstractRange) { throw 'Could not find Abstract heading' }
    $abstractStart = $abstractRange.Paragraphs.Item(1).Range.Start
    $blankRemoved = 0
    for ($i = $doc.Paragraphs.Count; $i -ge 1; $i--) {
        $paragraph = $doc.Paragraphs.Item($i)
        if ($paragraph.Range.Start -le $abstractStart) { continue }
        $text = $paragraph.Range.Text.Trim([char]13, [char]7, [char]32, [char]9)
        if ([string]::IsNullOrWhiteSpace($text)) {
            $paragraph.Range.Delete()
            $blankRemoved++
        }
    }

    $mainHeadings = @(
        'Abstract',
        '1. Introduction',
        '2. Related works',
        '3. Materials and Methods',
        '4. Results and Discussion',
        '5. Conclusion',
        'Acknowledgements',
        'Reproducibility and data availability',
        'References'
    )

    $bodyStarted = $false
    $mainCount = 0
    $subCount = 0
    $bodyCount = 0
    foreach ($paragraph in $doc.Paragraphs) {
        $raw = $paragraph.Range.Text.Trim([char]13, [char]7)
        $text = ($raw -replace [string][char]11, ' ').Trim()
        if ($text -eq 'Abstract') { $bodyStarted = $true }
        if (-not $bodyStarted -or [string]::IsNullOrWhiteSpace($text)) { continue }

        $paragraph.Format.WidowControl = -1

        if ($mainHeadings -contains $text) {
            $paragraph.Format.SpaceBefore = 12
            $paragraph.Format.SpaceAfter = 6
            $paragraph.Format.LineSpacingRule = $wdLineSpaceSingle
            $paragraph.Format.Alignment = $wdAlignParagraphLeft
            $paragraph.Format.KeepWithNext = -1
            $headingFontRange = $paragraph.Range.Duplicate
            $headingFontRange.MoveEnd(1, -1) | Out-Null
            $headingFontRange.Font.Bold = -1
            $mainCount++
        }
        elseif ($text -match '^\d+\.\d+\s+') {
            $paragraph.Format.SpaceBefore = 10
            $paragraph.Format.SpaceAfter = 4
            $paragraph.Format.LineSpacingRule = $wdLineSpaceSingle
            $paragraph.Format.Alignment = $wdAlignParagraphLeft
            $paragraph.Format.KeepWithNext = -1
            $headingFontRange = $paragraph.Range.Duplicate
            $headingFontRange.MoveEnd(1, -1) | Out-Null
            $headingFontRange.Font.Bold = -1
            $subCount++
        }
        elseif ($text -match '^Keywords:') {
            $paragraph.Format.SpaceBefore = 6
            $paragraph.Format.SpaceAfter = 10
            $paragraph.Format.LineSpacingRule = $wdLineSpaceSingle
            $paragraph.Format.Alignment = $wdAlignParagraphLeft
            $paragraph.Format.KeepWithNext = 0
            $bodyCount++
        }
        elseif ($text -match '(?i)(Insert\s+Figure|Table\s+Here\s+Later|Figure\s+Here\s+Later)') {
            $paragraph.Format.SpaceBefore = 6
            $paragraph.Format.SpaceAfter = 6
            $paragraph.Format.LineSpacingRule = $wdLineSpaceSingle
            $paragraph.Format.Alignment = $wdAlignParagraphCenter
            $paragraph.Format.KeepWithNext = -1
            $paragraph.Range.HighlightColorIndex = $wdYellow
            $bodyCount++
        }
        elseif ($text -match '^Figure\s+\d+[\.,]' -or $text -match '^Note:') {
            $paragraph.Format.SpaceBefore = 4
            $paragraph.Format.SpaceAfter = 6
            $paragraph.Format.LineSpacingRule = $wdLineSpaceSingle
            $paragraph.Format.Alignment = $wdAlignParagraphLeft
            $paragraph.Format.KeepWithNext = 0
            $bodyCount++
        }
        else {
            $paragraph.Format.SpaceBefore = 0
            if ($paragraph.Range.ListFormat.ListType -ne 0) {
                $paragraph.Format.SpaceAfter = 3
            } else {
                $paragraph.Format.SpaceAfter = 6
            }
            $paragraph.Format.LineSpacingRule = $wdLineSpace1pt5
            $paragraph.Format.Alignment = $wdAlignParagraphJustify
            $paragraph.Format.KeepWithNext = 0
            $bodyCount++
        }
    }

    # Review aid requested by the author: make every existing parenthetical
    # author-year citation bold so it remains easy to locate during final
    # citation-manager insertion. Numeric-only parentheses are excluded.
    $boldCitationCount = 0
    $citationPattern = '\([A-Z][^()\r]*(?:(?:19|20)\d{2}|n\.d\.)[^()\r]*\)'
    for ($paragraphIndex = 1; $paragraphIndex -le $doc.Paragraphs.Count; $paragraphIndex++) {
        $paragraph = $doc.Paragraphs.Item($paragraphIndex)
        $paragraphText = $paragraph.Range.Text
        foreach ($match in [regex]::Matches($paragraphText, $citationPattern)) {
            $citationRange = $paragraph.Range.Duplicate
            $citationFind = $citationRange.Find
            $citationFind.ClearFormatting()
            $citationFind.Text = $match.Value
            $citationFind.Forward = $true
            $citationFind.Wrap = $wdFindStop
            $citationFind.Format = $false
            if (-not $citationFind.Execute()) {
                throw "Could not format citation for review: $($match.Value)"
            }
            $citationRange.Font.Bold = -1
            $boldCitationCount++
        }
    }

    $doc.Save()
    $doc.Close()
    $doc = $null
    $word.Quit()
    $word = $null

    [pscustomobject]@{
        Output = $outputFull
        ManualLineBreaksRemoved = $manualBreaksRemoved
        CitationTransitionsFixed = $citationFixCount
        OtherTextSpacingFixes = $textFixCount
        EmptyBodyParagraphsRemoved = $blankRemoved
        MainHeadingsFormatted = $mainCount
        SubheadingsFormatted = $subCount
        BodyParagraphsFormatted = $bodyCount
        CitationsBoldedForReview = $boldCitationCount
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
