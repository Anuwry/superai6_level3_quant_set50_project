param(
    [string]$Source = 'D:\Downloads\newest.docx',
    [string]$Output = 'D:\SET50_direction_prediction_paper\paper\newest_with_numbered_figures_tables.docx'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Source DOCX not found: $Source"
}

function Find-TextRange {
    param($Document, [string]$Text)
    $range = $Document.Content.Duplicate
    $find = $range.Find
    $find.ClearFormatting()
    $find.Text = $Text
    $find.Forward = $true
    $find.Wrap = 0  # wdFindStop
    $find.Format = $false
    if (-not $find.Execute()) {
        throw "Text not found: $Text"
    }
    return $range
}

function Format-BodyParagraph {
    param($Document, [string]$UniqueText)
    $range = Find-TextRange $Document $UniqueText
    $para = $range.Paragraphs.Item(1).Range
    $para.Font.Name = 'Times New Roman'
    $para.Font.Size = 12
    $para.Font.Bold = 0
    $para.Font.Italic = 0
    $para.HighlightColorIndex = 0
    $para.ParagraphFormat.Alignment = 3  # justified
    $para.ParagraphFormat.LineSpacingRule = 1  # 1.5 lines
    $para.ParagraphFormat.SpaceBefore = 0
    $para.ParagraphFormat.SpaceAfter = 0
}

function Format-Placeholder {
    param($Document, [string]$Text)
    $range = Find-TextRange $Document $Text
    $para = $range.Paragraphs.Item(1).Range
    $para.Font.Name = 'Times New Roman'
    $para.Font.Size = 12
    $para.Font.Bold = 1
    $para.Font.Italic = 0
    $para.HighlightColorIndex = 7  # yellow
    $para.ParagraphFormat.Alignment = 1  # centered
    $para.ParagraphFormat.LineSpacingRule = 0  # single
    $para.ParagraphFormat.SpaceBefore = 6
    $para.ParagraphFormat.SpaceAfter = 0
    $para.ParagraphFormat.KeepWithNext = -1
}

function Format-Caption {
    param($Document, [string]$UniquePrefix, [string]$Label)
    $range = Find-TextRange $Document $UniquePrefix
    $para = $range.Paragraphs.Item(1).Range
    $para.Font.Name = 'Times New Roman'
    $para.Font.Size = 12
    $para.Font.Bold = 0
    $para.Font.Italic = 0
    $para.HighlightColorIndex = 0
    $para.ParagraphFormat.Alignment = 0  # left
    $para.ParagraphFormat.LineSpacingRule = 0  # single
    $para.ParagraphFormat.SpaceBefore = 4
    $para.ParagraphFormat.SpaceAfter = 0
    $labelRange = $Document.Range($para.Start, $para.Start + $Label.Length)
    $labelRange.Font.Bold = 1
}

function Append-ToParagraph {
    param($Document, [string]$AnchorText, [string]$Sentence)
    $anchor = Find-TextRange $Document $AnchorText
    $para = $anchor.Paragraphs.Item(1).Range
    $position = $para.End - 1
    $insert = $Document.Range($position, $position)
    $insert.Text = " $Sentence"
    $newRange = $Document.Range($position, $position + $Sentence.Length + 1)
    $newRange.Font.Name = 'Times New Roman'
    $newRange.Font.Size = 12
    $newRange.Font.Bold = 0
    $newRange.Font.Italic = 0
    $newRange.HighlightColorIndex = 0
}

function Insert-BlockAfterParagraph {
    param($Document, [string]$AnchorText, [string[]]$Paragraphs)
    $anchor = Find-TextRange $Document $AnchorText
    $para = $anchor.Paragraphs.Item(1).Range
    $position = $para.End
    $insert = $Document.Range($position, $position)
    $insert.Text = (($Paragraphs -join "`r") + "`r")
}

function Insert-ParagraphBefore {
    param($Document, [string]$AnchorText, [string]$Text)
    $anchor = Find-TextRange $Document $AnchorText
    $para = $anchor.Paragraphs.Item(1).Range
    $position = $para.Start
    $insert = $Document.Range($position, $position)
    $insert.Text = "$Text`r"
}

function Replace-Text {
    param($Document, [string]$OldText, [string]$NewText)
    $range = Find-TextRange $Document $OldText
    $range.Text = $NewText
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($Source, $false, $true, $false)
    $document.TrackRevisions = $false

    # Renumber the existing Results summary before adding a new Methods Figure 3.
    Replace-Text $document `
        'Figure 3. Architecture-wise effects and uncertainty across audit pillars.' `
        'Figure 7. Architecture-wise effects and uncertainty across audit pillars.'

    # Methods 3.2: news construction.
    Append-ToParagraph $document `
        'The forecasting cohort begins in 2019, the first year with out-of-sample sentiment predictions.' `
        'The complete news-to-session construction is summarized in Figure 2.'
    $fig2Placeholder = '[Insert Figure 2 here: news data and out-of-sample sentiment pipeline]'
    $fig2Caption = 'Figure 2. Point-in-time financial-news and sentiment construction. Dated StockTBSA labels from 2018-2023 and forward news from 2024-2025 are normalized, filtered for SET50 relevance and processed using only previously observed labels. Predicted sentiment is mapped from publication time to the next tradable session before daily aggregation.'
    Insert-BlockAfterParagraph $document 'The complete news-to-session construction is summarized in Figure 2.' @($fig2Placeholder, $fig2Caption)
    Format-Placeholder $document $fig2Placeholder
    Format-Caption $document 'Figure 2. Point-in-time financial-news and sentiment construction.' 'Figure 2.'

    # Methods 3.3: temporal contract and the missing Table 1.
    Append-ToParagraph $document `
        'All dates use Asia/Bangkok and the 17:00 information cutoff.' `
        'Figure 3 illustrates the point-in-time label boundary and expanding outer folds, while Table 1 summarizes the frozen model windows and common evaluation settings.'
    $fig3Placeholder = '[Insert Figure 3 here: point-in-time expanding-window evaluation]'
    $fig3Caption = 'Figure 3. Point-in-time target contract and expanding-window evaluation. Features known by the close of day t predict the direction of day t+1, whose label is observed only after t+1. Training data expand through time, while 2022, 2023, 2024 and 2025 remain separate held-out outer years. All transforms are fitted on training data only.'
    $table1Placeholder = '[Insert Table 1 here: frozen models, windows, seeds, outer folds and common SET50 cohort]'
    $table1Caption = 'Table 1. Frozen architecture windows and common experimental settings. The table reports the registered sequence window for each architecture, five fixed seeds, expanding outer folds, common SET50 cohort and primary evaluation endpoint.'
    Insert-BlockAfterParagraph $document 'Figure 3 illustrates the point-in-time label boundary and expanding outer folds' @($fig3Placeholder, $fig3Caption, $table1Placeholder, $table1Caption)
    Format-Placeholder $document $fig3Placeholder
    Format-Caption $document 'Figure 3. Point-in-time target contract and expanding-window evaluation.' 'Figure 3.'
    Format-Placeholder $document $table1Placeholder
    Format-Caption $document 'Table 1. Frozen architecture windows and common experimental settings.' 'Table 1.'

    # Methods 3.4-3.5: numerical construction and architecture panel.
    Append-ToParagraph $document `
        'min-max scaling was fitted only on the supervised training portion of each fold.' `
        'Figure 4 summarizes the numerical feature construction, causal VMD arm and five fixed neural architectures.'
    $fig4Placeholder = '[Insert Figure 4 here: VMD and registered neural architectures]'
    $fig4Caption = 'Figure 4. Numerical inputs, causal VMD and registered architectures. The 116-feature technical-analysis representation is compared with a 122-feature representation obtained by adding six variables from causal rolling VMD fitted to a trailing 60-day window. Both representations are evaluated through the same five-model panel.'
    Insert-BlockAfterParagraph $document 'Figure 4 summarizes the numerical feature construction' @($fig4Placeholder, $fig4Caption)
    Format-Placeholder $document $fig4Placeholder
    Format-Caption $document 'Figure 4. Numerical inputs, causal VMD and registered architectures.' 'Figure 4.'

    # Methods 3.6: multimodal and role-structured evaluation.
    Append-ToParagraph $document `
        'Intrinsic sentiment accuracy is not interpreted as proof of forecasting value.' `
        'Figure 5 shows the paired forecasting routes, Bull/Bear/Leader process and associated falsification controls.'
    $fig5Placeholder = '[Insert Figure 5 here: multimodal debate and falsification pipeline]'
    $fig5Caption = 'Figure 5. Multimodal forecasting and role-structured sentiment audit. The five registered models produce paired forecasts with and without point-in-time sentiment. Financial news is filtered for SET50 relevance before Bull and Bear workers provide arguments to a Leader. Forecasting falsification arms and compute-matched sentiment controls are retained as separate audit components.'
    Insert-BlockAfterParagraph $document 'Figure 5 shows the paired forecasting routes' @($fig5Placeholder, $fig5Caption)
    Format-Placeholder $document $fig5Placeholder
    Format-Caption $document 'Figure 5. Multimodal forecasting and role-structured sentiment audit.' 'Figure 5.'

    # Methods 3.7-3.8: causal regimes, SHAP and inference.
    Append-ToParagraph $document `
        'Economic summaries are exploratory and are not used to certify deployment.' `
        'Figure 6 summarizes causal regime routing, train-only SHAP selection and the temporal inference hierarchy.'
    $fig6Placeholder = '[Insert Figure 6 here: regime-aware SHAP selection and inference]'
    $fig6Caption = 'Figure 6. Causal regime-aware SHAP selection and inference. Past returns and volatility define daily Bull, Sideway and Bear states. The reference model, SHAP values and regime-specific feature subsets are fitted within each training fold, frozen and applied to the held-out year. Paired effects are aggregated across seeds and years before Holm adjustment.'
    Insert-BlockAfterParagraph $document 'Figure 6 summarizes causal regime routing' @($fig6Placeholder, $fig6Caption)
    Format-Placeholder $document $fig6Placeholder
    Format-Caption $document 'Figure 6. Causal regime-aware SHAP selection and inference.' 'Figure 6.'

    # Results tables already cited in the prose: make every insertion point explicit.
    Replace-Text $document '[ Table Here Later (full rolling 2022-2025) ]' '[Insert Table 2 here: corrected point-in-time numerical ablation, 2022-2025]'
    $table2Caption = 'Table 2. Corrected point-in-time numerical ablation for Full-TA versus Full-TA plus causal rolling VMD, 2022-2025. Balanced accuracy effects are based on four outer years after within-fold seed aggregation; runtime is reported per seed-fold cell.'
    Insert-BlockAfterParagraph $document '[Insert Table 2 here:' @($table2Caption)
    Format-Placeholder $document '[Insert Table 2 here: corrected point-in-time numerical ablation, 2022-2025]'
    Format-Caption $document 'Table 2. Corrected point-in-time numerical ablation' 'Table 2.'

    Replace-Text $document '[ Table Here Later (10 day circular moving-block ) ]' '[Insert Table 3A here: observed-news moving-block sensitivity]'
    $table3aCaption = 'Table 3A. Moving-block bootstrap sensitivity for observed-news effects relative to Market-Only and Shuffled-News controls. Contrasts use the same eligible dates, frozen windows, architectures, seeds and training budgets.'
    Insert-BlockAfterParagraph $document '[Insert Table 3A here:' @($table3aCaption)
    Format-Placeholder $document '[Insert Table 3A here: observed-news moving-block sensitivity]'
    Format-Caption $document 'Table 3A. Moving-block bootstrap sensitivity' 'Table 3A.'

    Replace-Text $document '[ Table Here Later (LLM Benchmark on locked 2023 ) ]' '[Insert Table 3B here: locked 2023 intrinsic LLM benchmark]'
    $table3bCaption = 'Table 3B. Locked 2023 intrinsic sentiment benchmark for single-pass, compute-matched self-consistency and Bull/Bear/Leader systems. Accuracy is the primary endpoint; uncertainty is clustered by article and the two Leader contrasts are Holm-adjusted.'
    Insert-BlockAfterParagraph $document '[Insert Table 3B here:' @($table3bCaption)
    Format-Placeholder $document '[Insert Table 3B here: locked 2023 intrinsic LLM benchmark]'
    Format-Caption $document 'Table 3B. Locked 2023 intrinsic sentiment benchmark' 'Table 3B.'

    # Table 3C is referenced in the manuscript but its authoritative artifact remains pending.
    $table3cPlaceholder = '[Insert Table 3C here: common-cohort Market-Only, Local-NLP and Leader-derived forecasting comparison]'
    $table3cCaption = 'Table 3C. Common-cohort downstream SET50 forecasting comparison of Market-Only, Local-NLP and Leader-derived sentiment arms. The final table must use identical overlapping dates, architectures, frozen windows, seeds, training budgets and targets and report coverage, failures, runtime, token cost and paired uncertainty.'
    Insert-BlockAfterParagraph $document 'must not be inferred from the intrinsic accuracy values.' @($table3cPlaceholder, $table3cCaption)
    Format-Placeholder $document $table3cPlaceholder
    Format-Caption $document 'Table 3C. Common-cohort downstream SET50 forecasting comparison' 'Table 3C.'

    Replace-Text $document ' [ Table Here Later (Primary regime-specific SHAP reduction contrast) ]' '[Insert Table 4 here: primary regime-specific SHAP reduction contrast]'
    Insert-ParagraphBefore $document '[Insert Table 4 here:' 'The primary Regime-SHAP minus Regime-All contrast is reported in Table 4.'
    Format-BodyParagraph $document 'The primary Regime-SHAP minus Regime-All contrast is reported in Table 4.'
    $table4Caption = 'Table 4. Primary regime-specific SHAP reduction contrast by architecture. The contrast is Regime-SHAP minus Regime-All after seeds are averaged within each outer year; SHAP values are fitted-model attributions rather than causal effects.'
    Insert-BlockAfterParagraph $document '[Insert Table 4 here:' @($table4Caption)
    Format-Placeholder $document '[Insert Table 4 here: primary regime-specific SHAP reduction contrast]'
    Format-Caption $document 'Table 4. Primary regime-specific SHAP reduction contrast' 'Table 4.'

    Replace-Text $document '[ Table Here Later (partial-2026 forward robustness) ]' '[Insert Table 5A here: partial-2026 forward robustness]'
    $table5aCaption = 'Table 5A. Source-contingent partial-2026 forward robustness across five architectures and two registered objectives. Direction accuracy, balanced accuracy, MCC, AUC, Brier score and prediction-side prevalence are reported using the validation-gate fallback threshold of 0.50.'
    Insert-BlockAfterParagraph $document '[Insert Table 5A here:' @($table5aCaption)
    Format-Placeholder $document '[Insert Table 5A here: partial-2026 forward robustness]'
    Format-Caption $document 'Table 5A. Source-contingent partial-2026 forward robustness' 'Table 5A.'

    Replace-Text $document '[ Table Here Later (SET100 same-exchange numerical transfer) ]' '[Insert Table 5B here: SET100 same-exchange numerical transfer]'
    $table5bCaption = 'Table 5B. Frozen SET100 same-exchange numerical transfer. SET100 uses the same 122 numerical features, architecture-specific windows, seeds, epochs and 2022-2025 outer years as the matched SET50 comparator, without SET100 retuning.'
    Insert-BlockAfterParagraph $document '[Insert Table 5B here:' @($table5bCaption)
    Format-Placeholder $document '[Insert Table 5B here: SET100 same-exchange numerical transfer]'
    Format-Caption $document 'Table 5B. Frozen SET100 same-exchange numerical transfer.' 'Table 5B.'

    # Results summary figure follows the six Methods figures.
    Replace-Text $document '[ Figure Here Later ]' '[Insert Figure 7 here: architecture-wise effects across audit pillars]'
    Insert-ParagraphBefore $document '[Insert Figure 7 here:' 'Figure 7 compares architecture-wise effects and uncertainty across the completed audit pillars.'
    Format-BodyParagraph $document 'Figure 7 compares architecture-wise effects and uncertainty across the completed audit pillars.'
    Format-Placeholder $document '[Insert Figure 7 here: architecture-wise effects across audit pillars]'
    Format-Caption $document 'Figure 7. Architecture-wise effects and uncertainty across audit pillars.' 'Figure 7.'

    # Save a new file. The source remains read-only and unchanged.
    if (Test-Path -LiteralPath $Output) {
        Remove-Item -LiteralPath $Output -Force
    }
    $document.SaveAs2($Output, 16)  # wdFormatDocumentDefault (.docx)
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
