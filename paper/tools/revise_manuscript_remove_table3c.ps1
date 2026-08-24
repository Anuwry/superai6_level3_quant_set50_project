$ErrorActionPreference = 'Stop'

$source = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_spacing_corrected.docx'
$working = 'D:\SET50_direction_prediction_paper\paper\qa\table3c_revision_working.docx'
$output = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_revised.docx'
$figure5 = 'D:\SET50_direction_prediction_paper\paper\assets\figure5_separated_audits.png'
$figure7 = 'D:\SET50_direction_prediction_paper\paper\assets\figure7_heatmap_summary.png'

$ndash = [char]0x2013
$times = [char]0x00D7

foreach ($required in @($source, $figure5, $figure7)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

Copy-Item -LiteralPath $source -Destination $working -Force
if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Force
}

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
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Prefix
    )
    $range = Find-TextRange -Document $Document -Text $Prefix
    return $range.Paragraphs.Item(1)
}

function Set-ParagraphText {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Prefix,
        [Parameter(Mandatory = $true)][string]$NewText
    )
    $paragraph = Find-ParagraphStarting -Document $Document -Prefix $Prefix
    $content = $paragraph.Range.Duplicate
    $content.End = $content.End - 1
    $content.Text = $NewText
}

function Delete-ParagraphStarting {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Prefix
    )
    $paragraph = Find-ParagraphStarting -Document $Document -Prefix $Prefix
    $paragraph.Range.Delete() | Out-Null
}

function Replace-InlineFigureBeforeCaption {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$CaptionPrefix,
        [Parameter(Mandatory = $true)][string]$ImagePath,
        [Parameter(Mandatory = $true)][double]$WidthCm
    )

    $caption = Find-ParagraphStarting -Document $Document -Prefix $CaptionPrefix
    $captionStart = $caption.Range.Start
    $paragraphIndex = 0
    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        if ($Document.Paragraphs.Item($i).Range.Start -eq $captionStart) {
            $paragraphIndex = $i
            break
        }
    }
    if ($paragraphIndex -eq 0) {
        throw "Caption paragraph index not found: $CaptionPrefix"
    }

    $imageParagraph = $null
    for ($i = $paragraphIndex - 1; $i -ge [Math]::Max(1, $paragraphIndex - 5); $i--) {
        $candidate = $Document.Paragraphs.Item($i)
        if ($candidate.Range.InlineShapes.Count -gt 0) {
            $imageParagraph = $candidate
            break
        }
    }
    if ($null -eq $imageParagraph) {
        throw "Inline figure not found before caption: $CaptionPrefix"
    }

    while ($imageParagraph.Range.InlineShapes.Count -gt 0) {
        $imageParagraph.Range.InlineShapes.Item(1).Delete()
    }
    $content = $imageParagraph.Range.Duplicate
    $content.End = $content.End - 1
    $content.Text = ''

    $insert = $imageParagraph.Range.Duplicate
    $insert.Collapse(1)
    $shape = $Document.InlineShapes.AddPicture($ImagePath, $false, $true, $insert)
    $shape.LockAspectRatio = -1
    $shape.Width = $WidthCm * 28.3464567

    $imageParagraph.Alignment = 1
    $imageParagraph.Format.LineSpacingRule = 0
    $imageParagraph.Format.SpaceBefore = 6
    $imageParagraph.Format.SpaceAfter = 3
    $imageParagraph.Format.KeepWithNext = -1
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($working, $false, $false)

    $methodText = 'The audited downstream forecasting route uses expanding out-of-sample Local-NLP daily features together with the registered Market-Only, News-Only, shuffled-news, lagged-news and random-feature controls. The Bull/Bear/Leader system is evaluated separately on the locked 2023 intrinsic sentiment cohort, and its sentiment accuracy is not interpreted as evidence of downstream forecasting value. Figure 5 summarizes these two distinct audit components.'
    Set-ParagraphText -Document $document -Prefix 'The audited forecasting route used to expand Local-NLP daily features.' -NewText $methodText

    $figure5Caption = 'Figure 5. Separate multimodal forecasting and role-structured sentiment audits. The forecasting experiment compares Market-Only with point-in-time Local-NLP predicted news and its falsification controls. The locked 2023 Bull/Bear/Leader benchmark is evaluated only against compute-matched sentiment controls and is not used as a downstream forecasting arm.'
    Set-ParagraphText -Document $document -Prefix 'Figure 5.' -NewText $figure5Caption

    Set-ParagraphText -Document $document -Prefix '4.3 The LLM role system improved intrinsic sentiment' -NewText '4.3 The LLM role system improved intrinsic sentiment'

    $table3BNote = 'Note: Table 3B reports intrinsic sentiment performance on the locked 2023 cohort. These results must not be interpreted as evidence of downstream SET50 forecasting value.'
    Set-ParagraphText -Document $document -Prefix 'Note: Table 3B is the intrinsic sentiment benchmark.' -NewText $table3BNote

    # Preserve the completed Local-NLP x Regime-SHAP factorial, but move it to the regime-results section.
    Delete-ParagraphStarting -Document $document -Prefix 'The later integrated 2'

    # Delete the unavailable placeholder table and its caption.
    $captionParagraph = Find-ParagraphStarting -Document $document -Prefix 'Table 3C.'
    $captionEnd = $captionParagraph.Range.End
    $tableToDelete = $null
    foreach ($table in $document.Tables) {
        if ($table.Range.Start -ge $captionEnd) {
            $tableToDelete = $table
            break
        }
    }
    if ($null -eq $tableToDelete) {
        throw 'Table following Table 3C caption was not found.'
    }
    $tableToDelete.Delete()
    Delete-ParagraphStarting -Document $document -Prefix 'Table 3C.'

    $integratedText = 'Separately, the post-hoc integrated 2' + $times + '2 robustness analysis crossed global versus Regime-SHAP numerical inputs with absence versus presence of Local-NLP predicted news on the 2019' + $ndash + '2025 common cohort. The highest mean balanced accuracy was 54.07% for LSTM' + $ndash + 'CNN with Regime-SHAP numerical inputs without news. Adding news within the regime-selected arm improved mean balanced accuracy for two of five models (LSTM +0.13 and LSTM' + $ndash + 'Attention +1.03 points) and reduced it for the other three. No balanced-accuracy contrast survived Holm adjustment. The complete factorial is retained as post-hoc integrated evidence in Table S5 rather than promoted as a success claim.'
    $table4Note = Find-ParagraphStarting -Document $document -Prefix 'Note: The contrast is Regime-SHAP minus Regime-All.'
    $insertStart = $table4Note.Range.End
    $insertRange = $document.Range($insertStart, $insertStart)
    $insertRange.InsertAfter($integratedText + "`r")
    $inserted = $document.Range($insertStart, $insertStart + $integratedText.Length + 1)
    $inserted.Font.Name = 'Times New Roman'
    $inserted.Font.Size = 12
    $inserted.Font.Bold = 0
    $inserted.ParagraphFormat.Alignment = 3
    $inserted.ParagraphFormat.LineSpacingRule = 1
    $inserted.ParagraphFormat.SpaceBefore = 0
    $inserted.ParagraphFormat.SpaceAfter = 6

    $figure7Intro = 'Figure 7 summarizes architecture-wise point estimates across the completed audit pillars, while corresponding uncertainty and multiplicity-adjusted results remain in Tables 2, 3A, 4 and 5B.'
    Set-ParagraphText -Document $document -Prefix 'Figure 7 compares architecture-wise effects and uncertainty' -NewText $figure7Intro

    $figure7Caption = 'Figure 7. Architecture-wise point-estimate summary across completed audit pillars. Heatmap cells report paired balanced-accuracy changes in percentage points for VMD, Observed-News versus Market-Only, Regime-SHAP versus Regime-All and SET100 minus matched SET50; green denotes positive and red denotes negative point estimates. The lower panel reports the Leader''s intrinsic sentiment-accuracy gains over compute-matched controls as a separate endpoint. Confidence intervals and Holm-adjusted p-values are reported in Tables 2, 3A, 4 and 5B.'
    Set-ParagraphText -Document $document -Prefix 'Figure 7.' -NewText $figure7Caption

    $figure1Phrase = Find-TextRange -Document $document -Text 'out-of-sample Local-NLP and Bull/Bear/Leader'
    $figure1Phrase.Text = 'out-of-sample Local-NLP forecasting and intrinsic Bull/Bear/Leader evaluation'

    Replace-InlineFigureBeforeCaption -Document $document -CaptionPrefix 'Figure 5.' -ImagePath $figure5 -WidthCm 15.0
    Replace-InlineFigureBeforeCaption -Document $document -CaptionPrefix 'Figure 7.' -ImagePath $figure7 -WidthCm 15.2

    $document.Repaginate()
    $document.SaveAs2($output, 16)

    $summary = [ordered]@{
        Output = $output
        Pages = $document.ComputeStatistics(2)
        Tables = $document.Tables.Count
        InlineFigures = $document.InlineShapes.Count
        Fields = $document.Fields.Count
        Equations = $document.OMaths.Count
    }
    $summary.GetEnumerator() | ForEach-Object { Write-Output ("{0}={1}" -f $_.Key, $_.Value) }
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
