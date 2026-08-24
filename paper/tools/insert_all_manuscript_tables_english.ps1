param(
    [string]$Source = 'D:\Downloads\newest_กว่า.docx',
    [string]$Output = 'D:\SET50_direction_prediction_paper\paper\newest_with_all_tables_english.docx'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Source DOCX not found: $Source"
}

$TableRoot = 'D:\SET50_direction_prediction_paper\outputs\manuscript_tables_v1'

function Find-TextRange {
    param($Document, [string]$Text)
    $range = $Document.Content.Duplicate
    $find = $range.Find
    $find.ClearFormatting()
    $find.Text = $Text
    $find.Forward = $true
    $find.Wrap = 0
    $find.Format = $false
    if (-not $find.Execute()) {
        throw "Text not found: $Text"
    }
    return $range
}

function Get-ModelLabel {
    param([string]$Model)
    switch ($Model) {
        'lstm' { return 'LSTM' }
        'cnn' { return 'CNN' }
        'lstm_cnn' { return 'LSTM-CNN' }
        'lstm_attention' { return 'LSTM-Attention' }
        'lstm_cnn_attention' { return 'LSTM-CNN-Attention' }
        default { return $Model }
    }
}

function Format-Percent {
    param($Value)
    return ('{0:F2}' -f (100.0 * [double]$Value))
}

function Format-Number {
    param($Value, [int]$Digits = 3)
    return ([double]$Value).ToString("F$Digits", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Format-PValue {
    param($Value)
    $number = [double]$Value
    if ($number -lt 0.001) {
        return '<0.001'
    }
    return $number.ToString('F3', [System.Globalization.CultureInfo]::InvariantCulture)
}

function Format-Signed {
    param($Value, [int]$Digits = 2)
    $number = [double]$Value
    if ([Math]::Abs($number) -lt [Math]::Pow(10, -$Digits) / 2) {
        $number = 0.0
    }
    if ($number -gt 0) {
        return ('+' + $number.ToString("F$Digits", [System.Globalization.CultureInfo]::InvariantCulture))
    }
    return $number.ToString("F$Digits", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Format-EffectCI {
    param($Estimate, $Lower, $Upper)
    return ('{0} [{1}, {2}]' -f (Format-Signed $Estimate 2), (Format-Signed $Lower 2), (Format-Signed $Upper 2))
}

function Get-OrderedRows {
    param([object[]]$Rows)
    $order = @{
        'lstm' = 1
        'cnn' = 2
        'lstm_cnn' = 3
        'lstm_attention' = 4
        'lstm_cnn_attention' = 5
    }
    return @($Rows | Sort-Object @{ Expression = { $order[$_.model] } }, @{ Expression = { $_.contrast } })
}

function Set-CellText {
    param($Cell, [string]$Text, [int]$Alignment, [double]$FontSize, [bool]$Bold)
    $Cell.Range.Text = $Text
    $Cell.VerticalAlignment = 1
    $Cell.Range.Font.Name = 'Times New Roman'
    $Cell.Range.Font.Size = $FontSize
    $Cell.Range.Font.Bold = if ($Bold) { -1 } else { 0 }
    $Cell.Range.Font.Italic = 0
    $Cell.Range.HighlightColorIndex = 0
    $Cell.Range.ParagraphFormat.Alignment = $Alignment
    $Cell.Range.ParagraphFormat.LineSpacingRule = 0
    $Cell.Range.ParagraphFormat.SpaceBefore = 0
    $Cell.Range.ParagraphFormat.SpaceAfter = 0
}

function Add-PaperTable {
    param(
        $Document,
        [string]$Placeholder,
        [string]$CaptionPrefix,
        [string[]]$Headers,
        [object[]]$Rows,
        [double[]]$WidthRatios,
        [int[]]$Alignments,
        [double]$FontSize = 8.0
    )

    if ($Headers.Count -ne $WidthRatios.Count -or $Headers.Count -ne $Alignments.Count) {
        throw "Header, width and alignment counts differ for $CaptionPrefix"
    }
    foreach ($row in $Rows) {
        if ($row.Count -ne $Headers.Count) {
            throw "A row has $($row.Count) cells but $($Headers.Count) are required for $CaptionPrefix"
        }
    }

    $placeholderRange = Find-TextRange $Document $Placeholder
    $placeholderParagraph = $placeholderRange.Paragraphs.Item(1).Range
    $placeholderParagraph.Delete()

    $captionRange = Find-TextRange $Document $CaptionPrefix
    $captionParagraph = $captionRange.Paragraphs.Item(1).Range
    $captionParagraph.ParagraphFormat.KeepWithNext = -1
    $captionParagraph.ParagraphFormat.SpaceAfter = 3

    $insertRange = $Document.Range($captionParagraph.End, $captionParagraph.End)
    $table = $Document.Tables.Add($insertRange, $Rows.Count + 1, $Headers.Count)
    $table.AllowAutoFit = $false
    $table.Rows.Alignment = 0
    $table.Rows.AllowBreakAcrossPages = 0
    $table.TopPadding = 3
    $table.BottomPadding = 3
    $table.LeftPadding = 4
    $table.RightPadding = 4
    $table.Spacing = 0
    $table.Borders.Enable = 1

    $section = $captionParagraph.Sections.Item(1)
    $availableWidth = $section.PageSetup.PageWidth - $section.PageSetup.LeftMargin - $section.PageSetup.RightMargin
    $ratioTotal = ($WidthRatios | Measure-Object -Sum).Sum
    for ($columnIndex = 1; $columnIndex -le $Headers.Count; $columnIndex++) {
        $width = $availableWidth * ($WidthRatios[$columnIndex - 1] / $ratioTotal)
        $table.Columns.Item($columnIndex).SetWidth($width, 0)
    }
    $table.PreferredWidthType = 3
    $table.PreferredWidth = $availableWidth

    $headerRow = $table.Rows.Item(1)
    $headerRow.HeadingFormat = -1
    $headerRow.Shading.BackgroundPatternColor = 15132390
    for ($columnIndex = 1; $columnIndex -le $Headers.Count; $columnIndex++) {
        Set-CellText $headerRow.Cells.Item($columnIndex) $Headers[$columnIndex - 1] 1 $FontSize $true
    }

    for ($rowIndex = 0; $rowIndex -lt $Rows.Count; $rowIndex++) {
        $wordRow = $table.Rows.Item($rowIndex + 2)
        for ($columnIndex = 0; $columnIndex -lt $Headers.Count; $columnIndex++) {
            Set-CellText $wordRow.Cells.Item($columnIndex + 1) ([string]$Rows[$rowIndex][$columnIndex]) $Alignments[$columnIndex] $FontSize $false
        }
    }

    $afterTable = $Document.Range($table.Range.End, $table.Range.End)
    $afterTable.ParagraphFormat.SpaceAfter = 6
    return $table
}

$table1Source = Get-OrderedRows @(Import-Csv (Join-Path $TableRoot 'table_1_protocol_cohort.csv'))
$table1Rows = @()
foreach ($row in $table1Source) {
    $table1Rows += ,@(
        (Get-ModelLabel $row.model),
        $row.selected_sequence_window,
        $row.outer_test_years,
        $row.outer_folds,
        $row.seeds_per_fold,
        ("{0} / {1}" -f $row.numerical_features, $row.news_features),
        'Balanced accuracy',
        ("{0} / {1}" -f $row.information_timezone, $row.information_cutoff)
    )
}

$table2Source = Get-OrderedRows @(Import-Csv (Join-Path $TableRoot 'table_2_numerical_ablation.csv'))
$table2Rows = @()
foreach ($row in $table2Source) {
    $table2Rows += ,@(
        (Get-ModelLabel $row.model),
        $row.selected_sequence_window,
        (Format-Percent $row.full_ta_balanced_accuracy_mean),
        (Format-Percent $row.vmd_balanced_accuracy_mean),
        (Format-EffectCI $row.balanced_accuracy_delta_pp $row.balanced_accuracy_delta_pp_ci95_lower $row.balanced_accuracy_delta_pp_ci95_upper),
        (Format-PValue $row.balanced_accuracy_exact_sign_flip_pvalue),
        ('{0:F2} / {1:F2}' -f [double]$row.full_ta_runtime_seconds_mean, [double]$row.vmd_runtime_seconds_mean)
    )
}

$contrastLabels = @{
    'observed_news_effect' = 'Observed News - Market-Only'
    'observed_vs_shuffled' = 'Observed News - Shuffled News'
}
$table3aSource = Get-OrderedRows @(Import-Csv (Join-Path $TableRoot 'table_3a_multimodal_falsification.csv'))
$table3aRows = @()
foreach ($row in $table3aSource) {
    $table3aRows += ,@(
        (Get-ModelLabel $row.model),
        $contrastLabels[$row.contrast],
        (Format-EffectCI $row.point_estimate $row.ci95_lower $row.ci95_upper),
        (Format-PValue $row.two_sided_pvalue),
        (Format-PValue $row.two_sided_pvalue_holm),
        $row.daily_rows
    )
}

$comparisonLabels = @{
    'leader_minus_self_consistency_3' = 'Leader - SC3 (equal calls)'
    'leader_minus_self_consistency_4' = 'Leader - SC4 (near cost)'
}
$table3bSource = @(Import-Csv (Join-Path $TableRoot 'table_3b_llm_intrinsic_separate.csv'))
$table3bRows = @()
foreach ($row in $table3bSource) {
    $table3bRows += ,@(
        $comparisonLabels[$row.comparison_id],
        $row.pairs,
        $row.unique_articles,
        (Format-Percent $row.control_accuracy_x),
        (Format-Percent $row.leader_accuracy),
        (Format-EffectCI $row.accuracy_delta_pp $row.accuracy_delta_pp_ci95_lower $row.accuracy_delta_pp_ci95_upper),
        (Format-PValue $row.holm_adjusted_pvalue)
    )
}

$table3cRows = @()
$table3cRows += ,([object[]]@('Market-Only', 'Not reported', 'Not available', 'Authoritative common-cohort artifact pending'))
$table3cRows += ,([object[]]@('Local-NLP sentiment', 'Not reported', 'Not available', 'Authoritative common-cohort artifact pending'))
$table3cRows += ,([object[]]@('Leader-derived sentiment', 'Not reported', 'Not available', 'Authoritative common-cohort artifact pending'))

$table4Source = Get-OrderedRows @(
    Import-Csv (Join-Path $TableRoot 'table_4_regime_shap.csv') |
        Where-Object {
            $_.inference_type -eq 'four_fold_exact_sign_flip' -and
            $_.contrast -eq 'regime_shap_reduction'
        }
)
$table4Rows = @()
foreach ($row in $table4Source) {
    $table4Rows += ,@(
        (Get-ModelLabel $row.model),
        (Format-Signed $row.point_estimate 2),
        ('[{0}, {1}]' -f (Format-Signed $row.ci95_lower 2), (Format-Signed $row.ci95_upper 2)),
        (Format-PValue $row.raw_pvalue),
        (Format-PValue $row.holm_adjusted_pvalue)
    )
}

$objectiveLabels = @{ 'direct' = 'Direct'; 'multitask' = 'Multitask' }
$table5aSource = Get-OrderedRows @(Import-Csv (Join-Path $TableRoot 'table_5a_forward_robustness.csv'))
$table5aRows = @()
foreach ($row in $table5aSource) {
    $table5aRows += ,@(
        (Get-ModelLabel $row.model),
        $objectiveLabels[$row.objective],
        $row.rows,
        (Format-Percent $row.direction_accuracy),
        (Format-Percent $row.balanced_accuracy),
        (Format-Number $row.mcc 3),
        (Format-Number $row.auc 3),
        (Format-Number $row.brier 3)
    )
}

$table5bSource = Get-OrderedRows @(Import-Csv (Join-Path $TableRoot 'table_5b_set100_transfer.csv'))
$table5bRows = @()
foreach ($row in $table5bSource) {
    $table5bRows += ,@(
        (Get-ModelLabel $row.model),
        $row.sequence_window,
        (Format-Percent $row.balanced_accuracy_set50_mean),
        (Format-Percent $row.balanced_accuracy_set100_mean),
        (Format-EffectCI $row.balanced_accuracy_delta_pp $row.balanced_accuracy_delta_ci95_lower $row.balanced_accuracy_delta_ci95_upper),
        (Format-PValue $row.balanced_accuracy_holm_pvalue)
    )
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($Source, $false, $true, $false)
    $document.TrackRevisions = $false

    Add-PaperTable $document `
        '[Insert Table 1 here: frozen models, windows, seeds, outer folds and common SET50 cohort]' `
        'Table 1. Frozen architecture windows and common experimental settings.' `
        @('Architecture', 'Window', 'Test years', 'Outer folds', 'Seeds / fold', 'Features Num. / News', 'Primary endpoint', 'Time zone / cutoff') `
        $table1Rows `
        @(0.21, 0.07, 0.11, 0.08, 0.09, 0.14, 0.14, 0.16) `
        @(0, 1, 1, 1, 1, 1, 1, 1) `
        7.3 | Out-Null

    Add-PaperTable $document `
        '[Insert Table 2 here: corrected point-in-time numerical ablation, 2022-2025]' `
        'Table 2. Corrected point-in-time numerical ablation' `
        @('Architecture', 'W', 'Full-TA BAcc (%)', '+VMD BAcc (%)', 'Delta pp [95% CI]', 'Exact p', 'Runtime s Full / VMD') `
        $table2Rows `
        @(0.20, 0.06, 0.12, 0.12, 0.27, 0.09, 0.14) `
        @(0, 1, 1, 1, 1, 1, 1) `
        7.5 | Out-Null

    Add-PaperTable $document `
        '[Insert Table 3A here: observed-news moving-block sensitivity]' `
        'Table 3A. Moving-block bootstrap sensitivity' `
        @('Architecture', 'Contrast', 'Delta BAcc pp [95% CI]', 'Raw p', 'Holm p', 'Daily n') `
        $table3aRows `
        @(0.19, 0.25, 0.30, 0.09, 0.09, 0.08) `
        @(0, 0, 1, 1, 1, 1) `
        7.2 | Out-Null

    Add-PaperTable $document `
        '[Insert Table 3B here: locked 2023 intrinsic LLM benchmark]' `
        'Table 3B. Locked 2023 intrinsic sentiment benchmark' `
        @('Comparison', 'Pairs', 'Articles', 'Control accuracy (%)', 'Leader accuracy (%)', 'Delta pp [95% CI]', 'Holm p') `
        $table3bRows `
        @(0.23, 0.08, 0.09, 0.13, 0.13, 0.25, 0.09) `
        @(0, 1, 1, 1, 1, 1, 1) `
        7.4 | Out-Null

    Add-PaperTable $document `
        '[Insert Table 3C here: common-cohort Market-Only, Local-NLP and Leader-derived forecasting comparison]' `
        'Table 3C. Common-cohort downstream SET50 forecasting comparison' `
        @('Arm', 'Common-cohort result', 'Paired inference', 'Evidence status') `
        $table3cRows `
        @(0.21, 0.18, 0.18, 0.43) `
        @(0, 1, 1, 0) `
        8.0 | Out-Null

    Add-PaperTable $document `
        '[Insert Table 4 here: primary regime-specific SHAP reduction contrast]' `
        'Table 4. Primary regime-specific SHAP reduction contrast' `
        @('Architecture', 'Delta BAcc (pp)', '95% CI', 'Exact p', 'Holm p') `
        $table4Rows `
        @(0.26, 0.18, 0.28, 0.14, 0.14) `
        @(0, 1, 1, 1, 1) `
        8.0 | Out-Null

    Add-PaperTable $document `
        '[Insert Table 5A here: partial-2026 forward robustness]' `
        'Table 5A. Source-contingent partial-2026 forward robustness' `
        @('Architecture', 'Objective', 'n', 'Accuracy (%)', 'BAcc (%)', 'MCC', 'AUC', 'Brier') `
        $table5aRows `
        @(0.20, 0.12, 0.07, 0.13, 0.12, 0.11, 0.11, 0.14) `
        @(0, 0, 1, 1, 1, 1, 1, 1) `
        7.3 | Out-Null

    Add-PaperTable $document `
        '[Insert Table 5B here: SET100 same-exchange numerical transfer]' `
        'Table 5B. Frozen SET100 same-exchange numerical transfer.' `
        @('Architecture', 'W', 'SET50 BAcc (%)', 'SET100 BAcc (%)', 'Delta pp [95% CI]', 'Holm p') `
        $table5bRows `
        @(0.23, 0.07, 0.15, 0.15, 0.29, 0.11) `
        @(0, 1, 1, 1, 1, 1) `
        7.6 | Out-Null

    if (Test-Path -LiteralPath $Output) {
        Remove-Item -LiteralPath $Output -Force
    }
    $document.SaveAs2($Output, 16)
    $document.Close(0)
    $document = $null
    $word.Quit()
    $word = $null

    Write-Output "Saved: $Output"
}
finally {
    if ($null -ne $document) {
        $document.Close(0)
    }
    if ($null -ne $word) {
        $word.Quit()
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
