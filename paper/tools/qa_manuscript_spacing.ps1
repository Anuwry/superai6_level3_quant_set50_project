param(
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression

function Get-DocxXmlStats {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = New-Object System.IO.FileStream(
        (Resolve-Path -LiteralPath $Path).Path,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite
    )
    $archive = New-Object System.IO.Compression.ZipArchive(
        $stream,
        [System.IO.Compression.ZipArchiveMode]::Read,
        $false
    )
    try {
        $entry = $archive.GetEntry('word/document.xml')
        $reader = New-Object System.IO.StreamReader($entry.Open())
        try { $xml = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $xmlDocument = New-Object System.Xml.XmlDocument
        $xmlDocument.PreserveWhitespace = $true
        $xmlDocument.LoadXml($xml)
        $namespace = New-Object System.Xml.XmlNamespaceManager($xmlDocument.NameTable)
        $namespace.AddNamespace('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
        $yellowParts = foreach ($run in $xmlDocument.SelectNodes('//w:r[w:rPr/w:highlight[@w:val="yellow"]]', $namespace)) {
            foreach ($node in $run.SelectNodes('.//w:t|.//w:tab|.//w:br', $namespace)) {
                if ($node.LocalName -eq 't') { $node.InnerText }
                elseif ($node.LocalName -eq 'tab') { "`t" }
                else { "`n" }
            }
        }
        $yellowNormalizedText = (($yellowParts -join '') -replace '\s+', ' ').Trim()

        [pscustomobject]@{
            RedRuns = ([regex]::Matches($xml, '<w:color[^>]+w:val="(?:FF0000|ff0000)"')).Count
            YellowHighlights = ([regex]::Matches($xml, '<w:highlight[^>]+w:val="yellow"')).Count
            YellowNormalizedText = $yellowNormalizedText
        }
    }
    finally {
        $archive.Dispose()
        $stream.Dispose()
    }
}

function Get-WordAudit {
    param([Parameter(Mandatory = $true)][string]$Path)
    $word = $null
    $doc = $null
    try {
        $word = New-Object -ComObject Word.Application
        $word.Visible = $false
        $word.DisplayAlerts = 0
        $doc = $word.Documents.Open((Resolve-Path -LiteralPath $Path).Path, $false, $true)
        $allText = $doc.Content.Text
        $manualBreaks = ([regex]::Matches($allText, [string][char]11)).Count
        $blankParagraphs = 0
        $abstractSeen = $false
        $mainHeadings = @()
        $subheadings = @()
        $missingTerminalCitationPeriods = @()
        $methodsSubheadings = @()
        $placeholderCount = 0
        $authorYearCitationCount = 0
        $nonBoldAuthorYearCitationCount = 0

        foreach ($paragraph in $doc.Paragraphs) {
            $text = $paragraph.Range.Text.Trim([char]13, [char]7, [char]32, [char]9)
            if ($text -eq 'Abstract') { $abstractSeen = $true }
            if ($abstractSeen -and [string]::IsNullOrWhiteSpace($text)) { $blankParagraphs++ }

            if ($text -in @('Abstract','1. Introduction','2. Related works','3. Materials and Methods','4. Results and Discussion','5. Conclusion','Acknowledgements','Reproducibility and data availability','References')) {
                $mainHeadings += [pscustomobject]@{
                    Text = $text
                    Before = [math]::Round($paragraph.Format.SpaceBefore, 2)
                    After = [math]::Round($paragraph.Format.SpaceAfter, 2)
                    LineRule = $paragraph.Format.LineSpacingRule
                    KeepNext = $paragraph.Format.KeepWithNext
                }
            }
            elseif ($text -match '^\d+\.\d+\s+') {
                $subheadings += [pscustomobject]@{
                    Text = $text
                    Before = [math]::Round($paragraph.Format.SpaceBefore, 2)
                    After = [math]::Round($paragraph.Format.SpaceAfter, 2)
                    LineRule = $paragraph.Format.LineSpacingRule
                    KeepNext = $paragraph.Format.KeepWithNext
                }
                if ($text -match '^3\.') { $methodsSubheadings += $text }
            }

            if ($text -match '(?i)(Insert\s+Figure|Table\s+Here\s+Later|Figure\s+Here\s+Later)') { $placeholderCount++ }
            if ($text.EndsWith(')') -and $text -match '\((?:[^()]|\([^()]*\))*(?:(?:19|20)\d{2}|n\.d\.)[^()]*\)$') {
                $missingTerminalCitationPeriods += $text.Substring([math]::Max(0, $text.Length - 120))
            }

            $citationPattern = '\([A-Z][^()\r]*(?:(?:19|20)\d{2}|n\.d\.)[^()\r]*\)'
            foreach ($match in [regex]::Matches($paragraph.Range.Text, $citationPattern)) {
                $authorYearCitationCount++
                $citationRange = $paragraph.Range.Duplicate
                $citationFind = $citationRange.Find
                $citationFind.ClearFormatting()
                $citationFind.Text = $match.Value
                $citationFind.Forward = $true
                $citationFind.Wrap = 0
                $citationFind.Format = $false
                if (-not $citationFind.Execute() -or $citationRange.Font.Bold -ne -1) {
                    $nonBoldAuthorYearCitationCount++
                }
            }
        }

        [pscustomobject]@{
            Text = $allText
            Paragraphs = $doc.Paragraphs.Count
            ManualBreaks = $manualBreaks
            BlankParagraphsAfterAbstract = $blankParagraphs
            MainHeadings = $mainHeadings
            Subheadings = $subheadings
            MethodsSubheadings = $methodsSubheadings
            MissingTerminalCitationPeriods = $missingTerminalCitationPeriods
            PlaceholderCount = $placeholderCount
            AuthorYearCitationCount = $authorYearCitationCount
            NonBoldAuthorYearCitationCount = $nonBoldAuthorYearCitationCount
        }
    }
    finally {
        if ($null -ne $doc) { try { $doc.Close($false) } catch {} }
        if ($null -ne $word) { try { $word.Quit() } catch {} }
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
    }
}

$inputAudit = Get-WordAudit -Path $InputPath
$outputAudit = Get-WordAudit -Path $OutputPath
$inputXml = Get-DocxXmlStats -Path $InputPath
$outputXml = Get-DocxXmlStats -Path $OutputPath

$digitPattern = '(?<!\p{L})\d+(?:[.,]\d+)*'
$inputNumbers = [regex]::Matches($inputAudit.Text, $digitPattern) | ForEach-Object Value
$outputNumbers = [regex]::Matches($outputAudit.Text, $digitPattern) | ForEach-Object Value
$numericTokensEqual = (($inputNumbers -join '|') -ceq ($outputNumbers -join '|'))

$mainFormattingValid = @($outputAudit.MainHeadings | Where-Object {
    $_.Before -ne 12 -or $_.After -ne 6 -or $_.LineRule -ne 0 -or $_.KeepNext -ne -1
}).Count -eq 0
$subFormattingValid = @($outputAudit.Subheadings | Where-Object {
    $_.Before -ne 10 -or $_.After -ne 4 -or $_.LineRule -ne 0 -or $_.KeepNext -ne -1
}).Count -eq 0

[pscustomobject]@{
    InputParagraphs = $inputAudit.Paragraphs
    OutputParagraphs = $outputAudit.Paragraphs
    OutputManualBreaks = $outputAudit.ManualBreaks
    OutputBlankParagraphsAfterAbstract = $outputAudit.BlankParagraphsAfterAbstract
    MainHeadingCount = $outputAudit.MainHeadings.Count
    SubheadingCount = $outputAudit.Subheadings.Count
    MethodsSubheadings = ($outputAudit.MethodsSubheadings -join '; ')
    MainHeadingFormattingValid = $mainFormattingValid
    SubheadingFormattingValid = $subFormattingValid
    MissingTerminalCitationPeriods = $outputAudit.MissingTerminalCitationPeriods.Count
    InputRedRuns = $inputXml.RedRuns
    OutputRedRuns = $outputXml.RedRuns
    InputYellowHighlights = $inputXml.YellowHighlights
    OutputYellowHighlights = $outputXml.YellowHighlights
    HighlightedTextPreserved = ($inputXml.YellowNormalizedText -ceq $outputXml.YellowNormalizedText)
    InputPlaceholders = $inputAudit.PlaceholderCount
    OutputPlaceholders = $outputAudit.PlaceholderCount
    AuthorYearCitations = $outputAudit.AuthorYearCitationCount
    NonBoldAuthorYearCitations = $outputAudit.NonBoldAuthorYearCitationCount
    NumericTokensEqual = $numericTokensEqual
    InputNumericTokenCount = $inputNumbers.Count
    OutputNumericTokenCount = $outputNumbers.Count
    SHA256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash
}

if ($outputAudit.ManualBreaks -ne 0) { throw 'Manual line breaks remain in output.' }
if ($outputAudit.BlankParagraphsAfterAbstract -ne 0) { throw 'Blank body paragraphs remain in output.' }
if (-not $mainFormattingValid) { throw 'Main heading spacing validation failed.' }
if (-not $subFormattingValid) { throw 'Subheading spacing validation failed.' }
if ($outputAudit.MethodsSubheadings.Count -ne 8) { throw 'Methods must contain exactly 8 numbered subsections.' }
if ($outputAudit.MissingTerminalCitationPeriods.Count -ne 0) { throw 'A paragraph-ending citation still lacks terminal punctuation.' }
if ($inputXml.RedRuns -ne $outputXml.RedRuns) { throw 'Red review formatting count changed.' }
if ($inputXml.YellowNormalizedText -cne $outputXml.YellowNormalizedText) { throw 'Yellow-highlighted visible text changed.' }
if ($inputAudit.PlaceholderCount -ne $outputAudit.PlaceholderCount) { throw 'Display placeholder count changed.' }
if ($outputAudit.NonBoldAuthorYearCitationCount -ne 0) { throw 'One or more author-year citations are not bold.' }
if (-not $numericTokensEqual) { throw 'Numeric token sequence changed.' }
