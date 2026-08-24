$ErrorActionPreference = 'Stop'

$source = 'D:\SET50_direction_prediction_paper\paper\qa\section_split_source_copy.docx'
$working = 'D:\SET50_direction_prediction_paper\paper\qa\results_discussion_separated_working.docx'
$output = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_results_discussion_separated.docx'

if (-not (Test-Path -LiteralPath $source)) {
    throw "Source manuscript not found: $source"
}
Copy-Item -LiteralPath $source -Destination $working -Force
if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Force
}

$oacute = [char]0x00F3
$lopez = 'L' + $oacute + 'pez'

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

function Find-ParagraphStarting {
    param($Document, [string]$Prefix)
    $range = Find-TextRange -Document $Document -Text $Prefix
    return $range.Paragraphs.Item(1)
}

function Format-BodyParagraph {
    param($Paragraph)
    $Paragraph.Range.Font.Name = 'Times New Roman'
    $Paragraph.Range.Font.Size = 12
    $Paragraph.Range.Font.Bold = 0
    $Paragraph.Range.Font.Italic = 0
    $Paragraph.Range.Font.Color = 0
    $Paragraph.Format.Alignment = 3
    $Paragraph.Format.LineSpacingRule = 1
    $Paragraph.Format.SpaceBefore = 0
    $Paragraph.Format.SpaceAfter = 6
    $Paragraph.Format.FirstLineIndent = 0
    $Paragraph.Format.KeepWithNext = 0
}

function Format-SectionHeading {
    param($Paragraph)
    $Paragraph.Range.Font.Name = 'Times New Roman'
    $Paragraph.Range.Font.Size = 12
    $Paragraph.Range.Font.Bold = 1
    $Paragraph.Range.Font.Italic = 0
    $Paragraph.Range.Font.Color = 0
    $Paragraph.Format.Alignment = 0
    $Paragraph.Format.LineSpacingRule = 1
    $Paragraph.Format.SpaceBefore = 12
    $Paragraph.Format.SpaceAfter = 6
    $Paragraph.Format.FirstLineIndent = 0
    $Paragraph.Format.KeepWithNext = -1
    $Paragraph.Format.PageBreakBefore = 0
}

function Set-ParagraphText {
    param($Document, [string]$Prefix, [string]$NewText, [switch]$Heading)
    $paragraph = Find-ParagraphStarting -Document $Document -Prefix $Prefix
    $content = $paragraph.Range.Duplicate
    $content.End = $content.End - 1
    $content.Text = $NewText
    if ($Heading) {
        Format-SectionHeading -Paragraph $paragraph
    }
    else {
        Format-BodyParagraph -Paragraph $paragraph
    }
}

function Get-ZoteroFieldByPattern {
    param($Document, [string]$Pattern)
    foreach ($field in $Document.Fields) {
        if ($field.Code.Text -match 'ZOTERO_ITEM' -and $field.Result.Text -like "*$Pattern*") {
            return $field
        }
    }
    throw "Zotero citation source not found for pattern: $Pattern"
}

function Insert-CombinedZoteroCitation {
    param(
        $Document,
        [string]$Placeholder,
        [string[]]$SourcePatterns,
        [string]$FormattedCitation
    )

    $sourceFields = @()
    $citationItems = New-Object System.Collections.ArrayList
    $seen = @{}
    $schema = 'https://github.com/citation-style-language/schema/raw/master/csl-citation.json'

    foreach ($pattern in $SourcePatterns) {
        $sourceField = Get-ZoteroFieldByPattern -Document $Document -Pattern $pattern
        $sourceFields += $sourceField
        $jsonStart = $sourceField.Code.Text.IndexOf('{')
        if ($jsonStart -lt 0) {
            throw "Invalid Zotero field for pattern: $pattern"
        }
        $payload = $sourceField.Code.Text.Substring($jsonStart) | ConvertFrom-Json
        if ($null -ne $payload.schema) {
            $schema = $payload.schema
        }
        foreach ($item in $payload.citationItems) {
            if ($null -ne $item.uris -and $item.uris.Count -gt 0) {
                $key = [string]$item.uris[0]
            }
            elseif ($null -ne $item.itemData.DOI) {
                $key = [string]$item.itemData.DOI
            }
            else {
                $key = [string]$item.itemData.title
            }
            if (-not $seen.ContainsKey($key)) {
                [void]$citationItems.Add($item)
                $seen[$key] = $true
            }
        }
    }

    $newPayload = [ordered]@{
        citationID = [Guid]::NewGuid().ToString('N').Substring(0, 8)
        properties = [ordered]@{
            unsorted = $false
            formattedCitation = $FormattedCitation
            plainCitation = $FormattedCitation
            noteIndex = 0
        }
        citationItems = @($citationItems)
        schema = $schema
    }
    $json = $newPayload | ConvertTo-Json -Depth 100 -Compress

    $target = Find-TextRange -Document $Document -Text $Placeholder
    $targetParagraph = $target.Paragraphs.Item(1)
    $fieldsBefore = $targetParagraph.Range.Fields.Count
    $templateField = $sourceFields[0]
    $wholeField = $Document.Range($templateField.Code.Start - 1, $templateField.Result.End + 1)
    $wholeField.Copy() | Out-Null
    $target.Text = ''
    $target.Collapse(1)
    $target.Paste()

    $newField = $targetParagraph.Range.Fields.Item($fieldsBefore + 1)
    $newField.Code.Text = " ADDIN ZOTERO_ITEM CSL_CITATION $json "
    $newField.Result.Text = $FormattedCitation
    $newField.Result.Font.Name = 'Times New Roman'
    $newField.Result.Font.Size = 12
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($working, $false, $false)

    Set-ParagraphText -Document $document -Prefix '4. Results and Discussion' -NewText '4. Results' -Heading
    Set-ParagraphText -Document $document -Prefix '5. Conclusion' -NewText '6. Conclusion' -Heading

    $discussionHeading = '5. Discussion'
    $discussion1 = 'Taken together, the results indicate that reliability depended more on the temporal contract and the fit between representation and architecture than on cumulative model complexity. VMD, predicted news and regime-SHAP produced effects with different signs across the five architectures, while the frozen SET100 transfer weakened every model. This architecture- and setting-sensitivity is consistent with financial deep-learning reviews that identify data representation, validation design and market context as major sources of apparent performance variation [[CITE_SYNTH_DL]]. The paper therefore does not infer a universal hierarchy from isolated winning cells; it treats those cells as hypotheses that must survive matched budgets, falsification and temporal transfer.'
    $discussion2 = 'Across the component ablations, a common compression-capacity trade-off provides the most coherent explanation. VMD removes high-frequency variation, daily sentiment compresses multiple articles into a scalar and regime-SHAP restricts the input set within smaller conditional samples. Each operation can improve signal-to-noise alignment for one architecture while removing weak complementary information required by another. Consequently, benefits reported by continuous-price VMD systems and richer news-interaction systems do not directly contradict the mixed next-day directional effects observed here [[CITE_SYNTH_COMPONENTS]]. The positive LSTM-CNN VMD cell and CNN regime-SHAP cell should therefore be interpreted as architecture-specific interactions, not evidence that either enhancement is generally beneficial.'
    $discussion3 = 'The intrinsic LLM experiment addresses a related but distinct level of reliability. Bull/Bear role separation improved sentiment accuracy relative to equal-call and near-cost controls, supporting the possibility that structured argument diversity contributes information beyond repeated sampling alone [[CITE_SYNTH_LLM]]. However, the local supervised classifier remained stronger, and an authoritative Leader-derived downstream forecasting arm was not available on the locked common cohort. The defensible inference is therefore limited to intrinsic sentiment classification: better role-structured reasoning does not, by itself, demonstrate incremental next-day market predictability. Keeping those endpoints separate prevents an upstream benchmark gain from being misrepresented as a trading or forecasting gain.'
    $discussion4 = 'The inferential results further distinguish absence of conclusive evidence from evidence of no effect. Four outer years impose a minimum exact two-sided sign-flip p-value of 0.125, whereas Holm adjustment protects the registered family against selective emphasis. Balanced accuracy, shuffled and lagged controls, the partial-2026 stress test and frozen same-exchange transfer expose failure modes that raw accuracy or a single holdout would miss. This interpretation follows established cautions concerning class-sensitive evaluation, temporal cross-validation and backtest selection [[CITE_SYNTH_INFERENCE]]. The practical implication is that the framework is most useful as a reliability audit: it identifies which apparent gains remain plausible, which are architecture contingent and which require genuinely prospective confirmation before deployment claims are warranted.'

    $anchor = Find-ParagraphStarting -Document $document -Prefix '6. Conclusion'
    $start = $anchor.Range.Start
    $block = $discussionHeading + "`r" + $discussion1 + "`r" + $discussion2 + "`r" + $discussion3 + "`r" + $discussion4 + "`r"
    $insert = $document.Range($start, $start)
    $insert.InsertBefore($block)

    Format-SectionHeading -Paragraph (Find-ParagraphStarting -Document $document -Prefix $discussionHeading)
    Format-BodyParagraph -Paragraph (Find-ParagraphStarting -Document $document -Prefix 'Taken together, the results indicate')
    Format-BodyParagraph -Paragraph (Find-ParagraphStarting -Document $document -Prefix 'Across the component ablations')
    Format-BodyParagraph -Paragraph (Find-ParagraphStarting -Document $document -Prefix 'The intrinsic LLM experiment addresses')
    Format-BodyParagraph -Paragraph (Find-ParagraphStarting -Document $document -Prefix 'The inferential results further distinguish')

    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_SYNTH_DL]]' -SourcePatterns @('Olorunnimbe & Viktor, 2023; Sezer') -FormattedCitation '(Olorunnimbe & Viktor, 2023; Sezer et al., 2020)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_SYNTH_COMPONENTS]]' -SourcePatterns @('T. Liu et al., 2022; Y. Liu', 'W.-J. Liu et al.') -FormattedCitation '(T. Liu et al., 2022; Y. Liu et al., 2024; W.-J. Liu et al., 2024; M. Wang et al., 2024)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_SYNTH_LLM]]' -SourcePatterns @('Du et al.') -FormattedCitation '(Du et al., 2024; Liang et al., 2024; X. Wang et al., 2023; Dong et al., 2025)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_SYNTH_INFERENCE]]' -SourcePatterns @('Brodersen', 'Bergmeir', 'Arnott') -FormattedCitation ('(Brodersen et al., 2010; Bergmeir et al., 2018; Arnott et al., 2019; Bailey & ' + $lopez + ' de Prado, 2014)')

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
