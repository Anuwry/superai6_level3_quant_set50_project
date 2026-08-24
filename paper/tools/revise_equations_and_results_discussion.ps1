$ErrorActionPreference = 'Stop'

$source = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_revised_figure5_fixed.docx'
$working = 'D:\SET50_direction_prediction_paper\paper\qa\equations_discussion_working.docx'
$output = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_equations_discussion_revised.docx'

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
}

function Set-ParagraphText {
    param($Document, [string]$Prefix, [string]$NewText)
    $paragraph = Find-ParagraphStarting -Document $Document -Prefix $Prefix
    $content = $paragraph.Range.Duplicate
    $content.End = $content.End - 1
    $content.Text = $NewText
    Format-BodyParagraph -Paragraph $paragraph
}

function Insert-ParagraphAfter {
    param($Document, [string]$AnchorPrefix, [string]$Text)
    $anchor = Find-ParagraphStarting -Document $Document -Prefix $AnchorPrefix
    $start = $anchor.Range.End
    $insert = $Document.Range($start, $start)
    $insert.InsertAfter($Text + "`r")
    $paragraph = $Document.Range($start, $start + $Text.Length + 1).Paragraphs.Item(1)
    Format-BodyParagraph -Paragraph $paragraph
    return $paragraph
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
        if ($jsonStart -lt 0) { throw "Invalid Zotero field for pattern: $pattern" }
        $payload = $sourceField.Code.Text.Substring($jsonStart) | ConvertFrom-Json
        if ($null -ne $payload.schema) { $schema = $payload.schema }
        foreach ($item in $payload.citationItems) {
            $key = $null
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

function Renumber-DisplayedEquations {
    param($Document)
    $oldLabels = @('(1)', '(2)', '(3a)', '(3b)', '(3c)', '(4)', '(5a)', '(5b)', '(5c)', '(6)', '(7)', '(8a)', '(8b)', '(8c)')
    $equationParagraphs = @()
    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        if ($paragraph.Range.OMaths.Count -gt 0) {
            $equationParagraphs += $paragraph
        }
    }
    if ($equationParagraphs.Count -ne 14) {
        throw "Expected 14 displayed equations but found $($equationParagraphs.Count)."
    }
    for ($i = 0; $i -lt $equationParagraphs.Count; $i++) {
        $range = $equationParagraphs[$i].Range.Duplicate
        $find = $range.Find
        $find.ClearFormatting()
        $find.Text = $oldLabels[$i]
        $find.Forward = $true
        $find.Wrap = 0
        if (-not $find.Execute()) {
            throw "Equation label not found in sequence: $($oldLabels[$i])"
        }
        $range.Text = '(' + ($i + 1).ToString() + ')'
    }
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($working, $false, $false)

    Renumber-DisplayedEquations -Document $document

    # Equation introductions and cross-references. Each displayed equation or equation group is source-anchored.
    Set-ParagraphText -Document $document -Prefix 'For each article-ticker pair, the scalar sentiment score' -NewText 'For each article-ticker pair, the scalar sentiment score and its session-level aggregation are defined in Eq. (1). Here, N_t is the set of pairs assigned to session t, n_t is its size, and c denotes the positive, neutral, or negative class. Article count, ticker-mention count, and news availability follow directly from N_t. This signed target-based construction is consistent with the labelled financial-sentiment task used to train the local classifier [[CITE_EQ1]].'

    Set-ParagraphText -Document $document -Prefix 'For feature date t, market variables use only information' -NewText 'For feature date t, market variables use only information available through the close at t. The point-in-time regression target and the realized and predicted directions are defined in Eq. (2), following established next-period financial direction-classification formulations [[CITE_EQ2]]. Each row records both its feature Date and the next-session Label_Date on which C_(t+1) becomes observable. At every boundary, a supervised training row is retained only when Label_Date_t is earlier than the first Date in the evaluation split. The final pre-evaluation row may supply sequence context, but it is excluded from fitting and scaler estimation, transformed with the training-fitted scaler and never contributes a target. Zero-return observations remain available to price regression but are excluded from binary metrics; a predicted exact-zero move is treated as an abstention and coverage is reported.'

    Set-ParagraphText -Document $document -Prefix 'Equations (3a)-(3c) formalize' -NewText 'Equations (3)-(5) formalize the trailing decomposition window, highest-frequency noise-mode rule, denoised close, and removed-mode energy ratio. The quantities u_(k,t) and omega_(k,t) denote mode k estimated at date t and its final centre frequency, respectively. The decomposition follows the VMD formulation, while the trailing-window implementation follows the leakage-avoiding window principle used in causal financial VMD forecasting [[CITE_EQ_VMD]].'

    Set-ParagraphText -Document $document -Prefix 'All five architectures minimize the mean squared error objective' -NewText 'All five architectures minimize the mean squared error objective in Eq. (6), where n is the number of supervised training sequences. This continuous-price objective is widely used in VMD-based deep stock-series forecasting [[CITE_EQ_MSE]].'

    Set-ParagraphText -Document $document -Prefix 'Daily regimes are assigned without future information' -NewText 'Daily regimes are assigned without future information using the causal trend score in Eq. (7). H contains the six return horizons in Eq. (8); sigma_(t,v(h)) is trailing volatility, with v(h)=20 sessions for h up to 20 and 60 sessions for h=60. ADX_(14,t) scales trend strength and EWMA_3 smooths the composite score. The construction adapts bull/bear phase analysis, multi-horizon time-series momentum and ADX trend strength to a causal day-level routing rule [[CITE_EQ_REGIME]].'

    Set-ParagraphText -Document $document -Prefix 'For outer fold f, the symmetric deadband lambda_f' -NewText 'For outer fold f, the symmetric deadband lambda_f in Eq. (9) is the 35th percentile of absolute training-fold scores only. Date t is Bull when z_t exceeds lambda_f, Bear when z_t is below -lambda_f, and Sideway otherwise. The state at t routes the prediction for t+1; targets and future columns are never used in regime construction. The train-only deadband is the study-specific causal adaptation of established bull/bear and momentum definitions [[CITE_EQ_REGIME_DEADBAND]].'

    Set-ParagraphText -Document $document -Prefix 'Within regime r, feature j is ranked by the mean absolute SHAP value' -NewText 'Within regime r, feature j is ranked by the mean absolute SHAP value in Eq. (10), computed only from the training ranking set T_r. The selected set S_r(k) contains the k highest-ranked features. SHAP is an additive feature-attribution framework for assigning prediction-specific feature contributions [[CITE_EQ_SHAP]].'

    Set-ParagraphText -Document $document -Prefix 'Balanced accuracy is defined in Eq.' -NewText 'Balanced accuracy is defined in Eq. (11), giving equal weight to sensitivity and specificity and therefore avoiding inflation from the majority class [[CITE_EQ_BACC]].'

    Set-ParagraphText -Document $document -Prefix 'For paired arms A and B, Eq.' -NewText 'For paired arms A and B, Eq. (12) first averages the five seed-level balanced-accuracy differences within each architecture m and outer year y, and then averages the four year effects. Equations (13)-(14) define the 16 sign-flipped test statistics and their exact two-sided probability. The finite randomization distribution and familywise interpretation are paired with Holm adjustment, while serial dependence is checked separately using block resampling [[CITE_EQ_INFERENCE]].'

    # Results-discussion integration: result -> literature comparison -> mechanism -> defensible inference.
    $vmdDiscussion = 'Unlike decomposition studies that report lower continuous-price errors, the present result does not imply that VMD is ineffective in every forecasting setting. Liu et al. decomposed causal sliding windows and predicted and recombined multiple modal subseries with a meta-learned LSTM, whereas later work paired VMD with a dual-channel attention network for stock-series regression [[CITE_DISC_VMD]]. Our ablation instead removes only the highest-frequency mode, keeps architectures and training budgets fixed, and evaluates the sign of a one-day movement. Smoothing can therefore reduce squared price error by shrinking deviations while erasing small sign-bearing movements around zero. The isolated +0.35-point LSTM-CNN effect is plausibly an architecture interaction: convolutional filters may exploit local residual patterns left after decomposition, whereas the recurrent and attention models may lose useful short-horizon variation. Because its interval crosses zero, this mechanism explains the observed pattern but does not establish a general denoising advantage.'
    Insert-ParagraphAfter -Document $document -AnchorPrefix 'The error and direction endpoints also diverged.' -Text $vmdDiscussion | Out-Null

    $newsDiscussion = 'These findings contrast with multimodal systems that report gains from richer news representations. Sentiment-attention models learn time-varying fusion from large news collections, while interaction-based models explicitly connect news items and report delayed pricing, long-memory effects and limitations of sentiment alone [[CITE_DISC_NEWS]]. The present daily scalar aggregation deliberately omits article-to-article interaction and compresses timing, entity specificity and narrative context. That compression, together with the shorter 2024-2025 headlines and lower sentiment confidence, provides a plausible mechanism for the negative incremental effects. The shuffled-news comparison is especially informative: if dated semantic information were driving the forecast, breaking its temporal alignment should materially reduce performance. Because that did not occur after adjustment, the news channel more likely acted as a weak proxy or noisy regularizer than as stable dated information. This does not show that financial news is uninformative; it narrows the claim to this out-of-sample daily representation under a fixed compute budget.'
    Insert-ParagraphAfter -Document $document -AnchorPrefix 'The negative-control result should not be interpreted as proof that news has no financial information.' -Text $newsDiscussion | Out-Null

    $llmDiscussion = 'The Leader advantage is consistent with multi-agent debate studies in which divergent arguments improve reasoning relative to a single trajectory, while self-consistency improves performance by sampling and aggregating multiple reasoning paths [[CITE_DISC_LLM]]. The equal-call and near-cost controls matter because they separate role diversity from the simpler benefit of spending more inference calls. However, the 2023 local classifier still exceeded the Leader by 6.23 accuracy points (82.82% versus 76.59%). This gap is informative rather than contradictory: the supervised classifier is optimized on a narrow target with repeated domain-specific lexical cues, whereas the LLM allocates capacity to broader reasoning and may be less calibrated to the local label policy. The result therefore suggests that Bull/Bear role separation recovers some ambiguous cases but does not replace temporally supervised domain calibration. Recent finance-agent reviews similarly identify prompt sensitivity and limited real-time adaptability as barriers to deployment, so the intrinsic gain remains distinct from downstream market value.'
    Insert-ParagraphAfter -Document $document -AnchorPrefix 'Note: Table 3B reports intrinsic sentiment performance' -Text $llmDiscussion | Out-Null

    $regimeDiscussion = 'The regime result should also be interpreted relative to the definition of regime. Classical bull/bear work identifies sustained market phases, and time-series momentum documents persistence across multiple horizons; the present method instead converts those ideas into a train-only day-level routing variable [[CITE_DISC_REGIME]]. The CNN gain is plausibly consistent with architecture-specific compression: local convolutional filters may benefit when regime-irrelevant inputs are removed, whereas recurrent and attention models can already reweight temporal context and may be harmed when selection discards weak but complementary variables. Splitting the training data by regime further reduces the effective sample size, increasing ranking and refit variance. SHAP explains the fitted model rather than establishing a causal market mechanism, and financial-XAI research emphasizes that stability and fidelity must be checked before feature narratives are trusted [[CITE_DISC_XAI]]. The low-fidelity LIME result reinforces that caution; consequently, the CNN effect is evidence of a model-specific routing hypothesis, not a universal regime feature set.'
    Insert-ParagraphAfter -Document $document -AnchorPrefix 'LIME did not provide an independent validation of SHAP.' -Text $regimeDiscussion | Out-Null

    $forwardDiscussion = 'The majority-side failures explain why raw accuracy and directional skill diverged. Balanced accuracy averages sensitivity and specificity, so a classifier that always predicts Up has balanced accuracy of 0.50 even when the Up share makes its raw accuracy appear high [[CITE_DISC_FORWARD]]. The LSTM multitask objective illustrates the mechanism: it reached 59.42% raw accuracy but only 51.40% balanced accuracy and AUC 0.502, suggesting that the auxiliary return head shifted the decision boundary toward the dominant class without improving ranking. Direct LSTM had lower raw accuracy but higher balanced accuracy and AUC. Thus, an auxiliary regression loss can improve a headline metric while weakening two-sided discrimination under distribution shift. The Deflated-Sharpe probability of 0.441 for the selected economic cell is consistent with warnings about backtest selection and multiplicity; it supports retaining that result as a diagnostic rather than a profitability claim [[CITE_DISC_BACKTEST]].'
    Insert-ParagraphAfter -Document $document -AnchorPrefix 'The complete 10-basis-point economic grid is exploratory.' -Text $forwardDiscussion | Out-Null

    $transferDiscussion = 'The transfer pattern is more naturally interpreted as a transportability test than as a second independent market. SET50 and SET100 share the exchange and calendar, but SET100 broadens the constituent universe; using the same frozen features and architecture-specific windows therefore tests whether a SET50-selected representation survives a different cross-sectional composition [[CITE_DISC_TRANSFER]]. The uniformly negative point estimates are consistent with feature-window mismatch rather than a different macro regime, because the outer years are identical. This stricter frozen design differs from dataset-specific studies that retune each market and can thereby absorb part of the shift. At the same time, four outer-year effects imply a minimum attainable exact two-sided p-value of 0.125, so sign consistency alone cannot establish degradation. The result narrows external validity: the audited configurations are not automatically portable even within the same exchange, but the evidence is insufficient to attribute the loss to any single architecture or constituent characteristic.'
    Insert-ParagraphAfter -Document $document -AnchorPrefix 'Note: Matched SET50 is the frozen 122-feature global numerical comparator' -Text $transferDiscussion | Out-Null

    # Insert dynamic Zotero fields by cloning existing valid fields and combining their CSL citation items.
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_EQ1]]' -SourcePatterns @('Uthayopas et al.') -FormattedCitation '(Uthayopas et al., 2025)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_EQ2]]' -SourcePatterns @('Fischer & Krauss') -FormattedCitation '(Fischer & Krauss, 2018; Hoseinzade & Haratizadeh, 2019)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_EQ_VMD]]' -SourcePatterns @('Dragomiretskiy', 'T. Liu et al., 2022)') -FormattedCitation '(Dragomiretskiy & Zosso, 2014; T. Liu et al., 2022)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_EQ_MSE]]' -SourcePatterns @('T. Liu et al., 2022; Y. Liu') -FormattedCitation '(T. Liu et al., 2022; Y. Liu et al., 2024)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_EQ_REGIME]]' -SourcePatterns @('Pagan & Sossounov', 'Moskowitz', 'Wilder') -FormattedCitation '(Pagan & Sossounov, 2003; Moskowitz et al., 2012; Wilder, 1978)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_EQ_REGIME_DEADBAND]]' -SourcePatterns @('Pagan & Sossounov', 'Moskowitz') -FormattedCitation '(Pagan & Sossounov, 2003; Moskowitz et al., 2012)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_EQ_SHAP]]' -SourcePatterns @('Lundberg & Lee') -FormattedCitation '(Lundberg & Lee, 2017)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_EQ_BACC]]' -SourcePatterns @('Brodersen') -FormattedCitation '(Brodersen et al., 2010)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_EQ_INFERENCE]]' -SourcePatterns @('Holm', 'Politis & Romano') -FormattedCitation '(Holm, 1979; Politis & Romano, 1992)'

    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_DISC_VMD]]' -SourcePatterns @('Dragomiretskiy', 'T. Liu et al., 2022; Y. Liu') -FormattedCitation '(Dragomiretskiy & Zosso, 2014; T. Liu et al., 2022; Y. Liu et al., 2024)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_DISC_NEWS]]' -SourcePatterns @('W.-J. Liu et al.', 'M. Wang et al.') -FormattedCitation '(W.-J. Liu et al., 2024; M. Wang et al., 2024)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_DISC_LLM]]' -SourcePatterns @('Du et al.', 'X. Wang et al.', 'Dong et al.') -FormattedCitation '(Du et al., 2024; Liang et al., 2024; X. Wang et al., 2023; Dong et al., 2025)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_DISC_REGIME]]' -SourcePatterns @('Pagan & Sossounov', 'Moskowitz') -FormattedCitation '(Pagan & Sossounov, 2003; Moskowitz et al., 2012)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_DISC_XAI]]' -SourcePatterns @('Lundberg & Lee', 'Yeo et al.') -FormattedCitation '(Lundberg & Lee, 2017; Yeo et al., 2025)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_DISC_FORWARD]]' -SourcePatterns @('Brodersen') -FormattedCitation '(Brodersen et al., 2010)'
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_DISC_BACKTEST]]' -SourcePatterns @('Arnott', 'Bailey') -FormattedCitation ('(Arnott et al., 2019; Bailey & ' + $lopez + ' de Prado, 2014; Olorunnimbe & Viktor, 2023)')
    Insert-CombinedZoteroCitation -Document $document -Placeholder '[[CITE_DISC_TRANSFER]]' -SourcePatterns @('The Stock Exchange of Thailand', 'Bergmeir', 'Olorunnimbe & Viktor, 2023; Sezer') -FormattedCitation '(Bergmeir et al., 2018; Olorunnimbe & Viktor, 2023; Sezer et al., 2020; The Stock Exchange of Thailand, n.d.-a, n.d.-b)'

    $document.Repaginate()
    $document.SaveAs2($output, 16)

    $citationCount = 0
    foreach ($field in $document.Fields) {
        if ($field.Code.Text -match 'ZOTERO_ITEM') { $citationCount++ }
    }
    Write-Output "Output=$output"
    Write-Output "Pages=$($document.ComputeStatistics(2))"
    Write-Output "Tables=$($document.Tables.Count)"
    Write-Output "InlineFigures=$($document.InlineShapes.Count)"
    Write-Output "Equations=$($document.OMaths.Count)"
    Write-Output "CitationFields=$citationCount"
    Write-Output "TotalFields=$($document.Fields.Count)"
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
