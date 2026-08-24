param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$wdFindStop = 0
$wdNoHighlight = 0
$p = [char]13

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
    $found = $find.Execute()
    if (-not $found) {
        throw "Could not find required text: $Text"
    }
    return $range.Duplicate
}

function Format-ParagraphAsHeading {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $range = Find-TextRange -Document $Document -Text $Text
    $paragraph = $range.Paragraphs.Item(1).Range
    $paragraph.Font.Bold = 1
    $paragraph.HighlightColorIndex = $wdNoHighlight
    $paragraph.ParagraphFormat.KeepWithNext = -1
}

function Format-LeadLabel {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $range = Find-TextRange -Document $Document -Text $Text
    $range.Font.Bold = 1
    $range.HighlightColorIndex = $wdNoHighlight
}

$inputResolved = (Resolve-Path -LiteralPath $InputPath).Path
$outputDirectory = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}
$outputFull = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $outputFull) {
    throw "Output already exists; refusing to overwrite: $outputFull"
}
Copy-Item -LiteralPath $inputResolved -Destination $outputFull

$methods = @'
3.1 Study design, data governance, and evidence hierarchy

Figure 1 summarizes a five-pillar reliability audit in which point-in-time market-data governance supports numerical denoising, multimodal news, regime-aware explanation, and later robustness tests. The numerical stream spans the full historical period; news enters only on dates with defensible out-of-sample sentiment estimates; and causal regime labels and train-only explanation rankings act only on the numerical feature pool. The corrected numerical ablation is the primary architecture-paired experiment. Multimodal negative controls are frozen retrospective falsification tests, the integrated regime-news and explanation analyses are post-hoc robustness evidence, the partial-2026 module is source-contingent forward evidence, and SET100 is a frozen same-exchange transfer audit rather than independent-market replication.

[Insert Figure 1 near here]

Figure 1. Five-pillar reliability-audit pipeline. Market-data governance and point-in-time reconstruction support the causal VMD, out-of-sample Local-NLP and Bull/Bear/Leader, causal Bull/Sideway/Bear routing with train-only SHAP, partial-2026, and frozen SET100 modules. Solid arrows denote data or forecasting flow; evaluation-only links are shown separately.

Daily, weekly, and monthly SET50 and SET100 price-index observations were manually downloaded from the publicly accessible Investing.com historical-data pages (Investing.com, n.d.-a, n.d.-b). The provenance manifest records source URLs, acquisition evidence, file sizes, SHA-256 digests, row counts, date ranges, and processing method. Public accessibility is not treated as an open-data license, provider terms continue to apply, and raw rows are not redistributed. Date-only observations were interpreted as Stock Exchange of Thailand sessions in Asia/Bangkok, with features available at 17:00 after the normal close. Weekly and monthly bars were shifted by one completed period before forward filling. The study uses provider-published price-index levels and applies no researcher-side split, dividend, constituent, or corporate-action adjustment.

Forty of 42 raw-file integrity checks passed. Two OHLC-containment warnings on 11 December 2025, present in both daily index files, were retained and disclosed; no critical check failed. The aligned SET100 cohort contained 3,360 of 3,381 required SET50 reference dates (99.3789%); 21 leading dates were excluded rather than backfilled. After the 60-day indicator warm-up, SET50 features begin on 3 May 2012. The four outer SET50 tests end on 18 December 2025 and contain 241, 243, 244, and 234 rows for 2022-2025.

3.2 News data and out-of-sample sentiment construction

The labelled Bilingual StockTBSA snapshot spans 3 January 2018 to 28 December 2023 and contains 10,295 articles, 15,949 article-ticker labels, and 12,706 valid positive, neutral, or negative pairs (Uthayopas et al., 2025). Exclude, not-stock, and ambiguous labels were not relabelled as neutral. The locked 2023 intrinsic cohort contains 1,333 pairs from 738 articles: 92 negative, 585 neutral, and 656 positive. For 2024-2025, 69,824 deduplicated official SET headlines yielded 4,619 point-in-time article-symbol pairs under six non-overlapping SET50 membership versions, including the March-April 2025 transition. These headlines are unlabelled and are evaluated only through downstream forecasting. The 2015-2017 corpus lacked comparable ticker and sentiment labels, and no defensible free 2012-2014 source was identified; homogeneous news coverage over 2012-2025 is therefore not claimed.

The operational sentiment model is balanced logistic regression over character TF-IDF features with the target ticker prepended. Annual expanding predictions trained on 2018 to predict 2019 and then expanded one year at a time through a 2018-2022 fit predicting 2023. For 2024-2025, the model was fitted once on labelled 2018-2023 data and frozen; pseudo-label retraining and Optuna tuning were not used. Because reliable intraday timestamps were unavailable in the labelled source, each item dated d was assigned conservatively to the first trading session strictly after d. Eight daily variables were constructed: sentiment mean and standard deviation; positive, negative, and neutral ratios; article count; ticker-mention count; and news availability. The forecasting cohort begins in 2019, the first year with out-of-sample sentiment predictions.

3.3 Point-in-time contract, temporal splits, and frozen windows

For feature date t, market variables use only information available through the close at t. The regression target is y_t = C_(t+1), and direction is d_t = I(C_(t+1) > C_t). Each row records both its feature Date and the next-session Label_Date on which C_(t+1) becomes observable. At every boundary, a supervised training row is retained only when Label_Date_t is earlier than the first Date in the evaluation split. The final pre-evaluation row may supply sequence context, but it is excluded from fitting and scaler estimation, transformed with the training-fitted scaler, and never contributes a target. Zero-return observations remain available to price regression but are excluded from binary metrics; a predicted exact-zero move is treated as an abstention and coverage is reported.

Candidate sequence lengths were 1, 3, 5, 10, and 20 trading days. Expanding validation folds trained on 2012-2017, 2012-2018, 2012-2019, and 2012-2020 and validated on 2018-2021. Selection prioritized balanced accuracy, followed by direction accuracy, RMSE, and the shorter-window tie-breaker, and was symmetric across Full-TA and Full-TA+VMD. Frozen windows were 5 days for LSTM, 20 for CNN, 20 for LSTM-CNN, 10 for LSTM-Attention, and 20 for LSTM-CNN-Attention. Outer folds trained through 2021, 2022, 2023, and 2024 and tested 2022, 2023, 2024, and 2025 using seeds 42, 123, 456, 789, and 2025. All dates use Asia/Bangkok and the 17:00 information cutoff.

3.4 Numerical features and causal VMD

The Full-TA control contains 116 variables constructed from daily and previously completed weekly/monthly OHLCV information: price and volume lags; 1-, 3-, 5-, 10-, 20-, and 60-day returns; moving averages; volatility, momentum, rate-of-change, candlestick, cross-timeframe, and volume-ratio features; direction lags; and Stochastic, RSI, MACD, Williams %R, CCI, ADX, and directional indicators.

Causal rolling VMD adds six variables. At each t, only the 60 closes from t-59 through t are decomposed using five modes, alpha = 1000, dual-ascent step tau = 0, a DC mode, tolerance 10^-7, and at most 500 iterations. The mode with the highest final centre frequency is treated as noise. Four retained intrinsic mode functions, the denoised close, and removed-mode energy ratio produce 122 numerical inputs. VMD is tested as a causal auxiliary representation rather than assumed to improve direction.

3.5 Neural architectures and training

Historical numerical, news, and regime modules predict the next close and derive direction from predicted next close minus current close. The compact architectures were: LSTM, LSTM(16) followed by Dense(8, ReLU); CNN, causal Conv1D(32, kernel 3) followed by global-average pooling and Dense(8, ReLU); LSTM-CNN, sequence-output LSTM(16) followed by the causal convolution and pooling stack; LSTM-Attention, sequence-output LSTM(16) followed by causal two-head attention (key dimension 8), pooling, and Dense(8, ReLU); and LSTM-CNN-Attention, LSTM followed by causal convolution, causal two-head attention, pooling, and Dense(8, ReLU). Every architecture uses a linear output.

All fits used Adam, mean squared error, 20 epochs, batch size 32, and shuffle = false. Random states were reset for every cell, deterministic TensorFlow operations were enabled, the Keras session was cleared between fits, and min-max scaling was fitted only on the supervised training portion of each fold.

3.6 Multimodal falsification and Bull/Bear/Leader evaluation

Observed predicted-news features were compared with a date-matched Market-Only arm and four falsification controls: News-Only, date-shuffled news, news lagged by five rows, and matched random features. All arms retained the frozen architecture, window, cohort, seeds, and training budget. The registered family comprised 100 model-control cells and 400 fits. Primary contrasts were Observed-News minus Market-Only and Observed-News minus Shuffled-News; the remaining controls are retained in the complete evidence bundle.

Local relevance and sentiment pipelines were evaluated annually on out-of-sample pairs, while the LLM benchmark used a separate locked design. A class-stratified set of 60 pairs from 2022 was used once to validate prompts before the 2023 cohort was opened. The Bull worker, Bear worker, and Leader used the same proprietary model under distinct prompts; the Leader returned positive, neutral, and negative probabilities and a scalar sentiment score. Controls were one pass, three identical-prompt self-consistency passes matching the role-system call count, and four passes within the frozen +/-15% cost band. Accuracy was the primary intrinsic endpoint, Macro-F1 was supporting evidence, uncertainty was clustered by article, and the two Leader contrasts were Holm-adjusted.

The audited forecasting route used expanding Local-NLP daily features. A separately specified Leader-derived route aggregates only outputs available by the market-close cutoff on date t and may predict t+1. Any downstream comparison must use identical overlapping dates, architecture, frozen window, seed set, training budget, and targets for Market-Only, Local-NLP, and Leader arms, and must report coverage, missing-news handling, model/version, prompt hash, runtime, token cost, and paired uncertainty. Intrinsic sentiment accuracy is not interpreted as proof of forecasting value.

3.7 Causal market regimes and SHAP selection

SHAP rankings used Gradient Explainer with 100 training-only background sequences, at most 128 evenly spaced training ranking sequences, 200 samples, deterministic cell seeds, and float32 tensors. Candidate feature counts were 10, 20, 30, 40, 60, 80, 100, and 122. A one-standard-error rule with balanced-accuracy, model-error, RMSE, and temporal-Jaccard guardrails selected the smallest stable subset. Frozen counts were 122 globally, 30 for Bull, 122 for Sideway, and 80 for Bear; a Spearman selector used the same counts as a size-matched comparator.

Seven outer arms separated routing, capacity, and selector effects: Global-All, Global3-All, Global-SHAP, Global-Spearman, Regime-All, Regime-SHAP, and Regime-Spearman. Because the global selector retained all 122 features, Global-SHAP and Global-Spearman are identity controls; the primary selector contrast is Regime-SHAP minus Regime-All. LIME was an explanation-fidelity diagnostic only: 1,800 instance-repeat explanations used an R^2 < 0.70 threshold for low fidelity, failed repeats remained in the denominator, and LIME did not affect training or selection.

3.8 Metrics and statistical inference

Balanced accuracy is the primary forecasting endpoint. Direction accuracy, MCC, coverage, RMSE, and MAE are secondary where applicable; AUC and Brier score are reported for probabilistic 2026 objectives. Seeds are averaged within each model-year cell before temporal inference and are not treated as independent market samples.

Historical paired contrasts use exact two-sided sign-flip tests over four outer-year effects; the minimum attainable two-sided p-value is 0.125. Descriptive t intervals communicate uncertainty but do not override the exact test, and Holm adjustment controls the registered familywise error rate. A 10-day circular moving-block bootstrap provides a serial-dependence sensitivity (Politis & Romano, 1992). Intrinsic LLM intervals use 5,000 article-cluster bootstrap replicates, article-cluster sign-flip sensitivity, and Holm adjustment across the two accuracy contrasts. Economic summaries are exploratory and are not used to certify deployment.

'@

$forward2026 = @'
Before opening the 2026 observations, the five architectures, windows, features, seeds, objectives, thresholds, and economic rules were frozen. The registered Yahoo Finance source failed its overlap and availability gate, after which a dated deviation accepted the Investing.com historical interface; the 138-row sample from 5 January to 30 July 2026 is therefore source-contingent. Direct models predicted Up with sigmoid binary cross-entropy. Multitask models added a standardized next-day log-return head with direction BCE weight 1.0 and return MSE weight 0.25, using train-only return scaling. Threshold candidates 0.50, 0.55, 0.60, and 0.65 were evaluated on 2019-2021 validation data; all ten model-objective cells failed the registered gate and reverted to 0.50.

The same frozen extension evaluated an exploratory next-session open-to-close proxy for long/flat and long/short rules at 5, 10, and 20 basis points per active round trip. The full grid is retained in the Supplement. These short-sample economic outputs are descriptive diagnostics, not evidence of profitability or deployment readiness.

'@

$set100 = @'
SET100 transfer retained the same 122 numerical features, frozen windows, seeds, epochs, and 2022-2025 outer years as the matched SET50 global numerical comparator. Features, VMD, and scaling were reconstructed within each SET100 fold; no SET100 retuning, Optuna search, early stopping, or model exclusion was allowed. All 100 planned fits (five models x four years x five seeds) completed.

SET100 news was excluded because a historically point-in-time constituent and news universe was unavailable; adding SET50-oriented news would have made the comparison asymmetric. The matched SET50 arm is therefore the Full-TA+VMD global numerical comparator rather than the later integrated regime-news arm. Because SET50 is nested within SET100 and both share the exchange, macro regimes, period, and many constituents, this is a same-exchange breadth transfer rather than external-market validation (The Stock Exchange of Thailand, n.d.).

'@

$closing = @'
5 Conclusion

Under a common point-in-time protocol, the five frozen architectures showed modest but architecture-dependent next-day SET50 directional structure rather than a universally superior configuration. Causal rolling VMD changed balanced accuracy by only -0.60 to +0.35 percentage points, predicted Local-NLP news did not survive the registered falsification and multiplicity controls, and regime-specific SHAP improved CNN while producing mixed or negative effects elsewhere. The Bull/Bear/Leader system delivered a clear intrinsic sentiment advantage over compute-matched self-consistency, but that endpoint is distinct from downstream forecasting value. Partial-2026 testing exposed majority-side collapse in several objectives, and frozen SET100 transfer weakened all five models. Collectively, the findings support the paper's central contribution: a reproducible audit framework that distinguishes selected component gains from improvements that remain reliable across time, architectures, controls, and transfer settings.

Limitations. Temporal inference is based on four outer test years, so the exact sign-flip design has limited power and cannot establish small but consistent effects below its attainable p-value resolution. The 2026 sample is partial-year and source-contingent, SET100 is an overlapping same-exchange index rather than an independent market, and news coverage varies in source, length, labelling, and timestamp precision across periods. The LLM roles share one proprietary base model, and intrinsic sentiment gains do not isolate debate as the unique mechanism or guarantee forecasting value. SHAP describes fitted-model attribution rather than causal market effects, while the low-fidelity LIME diagnostic limits feature-level interpretation. These constraints bound the claims to reliability under the audited SET setting rather than universal market predictability or deployment readiness.

Future work. The next study should preregister the complete protocol before opening a full untouched forward year, evaluate at least one independent ASEAN or comparable emerging-market index, and construct a consistently licensed point-in-time news archive with verifiable publication timestamps and constituent membership. The Leader-derived forecasting arm should be frozen and assessed on an identical common cohort with article coverage, latency, cost, failures, and paired uncertainty reported alongside predictive metrics. Model development should then focus on a single prespecified architecture and calibration strategy, followed by prospective shadow evaluation, rather than further selection on the already inspected 2022-2026 outcomes.

Acknowledgements

The authors thank their respective institutions for providing the research environment in which this study was conducted.

Reproducibility and data availability

The reproducibility package contains the model and evaluation code, fixed seeds, architecture and window specifications, temporal split definitions, point-in-time data contracts, prompt and protocol hashes, aggregate outputs, runtime and cost ledgers, and non-reconstructive integrity summaries. Raw SET50 and SET100 market rows are not redistributed; they can be obtained from the publicly accessible Investing.com historical-data pages subject to the provider's terms. Official 2024-2025 headlines remain subject to the Stock Exchange of Thailand website terms, and Bilingual StockTBSA is available from its public repository under the licence stated in its dataset card. Processed artifacts are shared only when they do not permit reconstruction of restricted raw records. The archival repository URL or DOI will be added after deposit.

'@

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($outputFull, $false, $false)
    $doc.TrackRevisions = $false

    # Replace all Method subsections while retaining the section-level heading.
    $methodsTitle = Find-TextRange -Document $doc -Text '3 Materials and Methods'
    $resultsTitle = Find-TextRange -Document $doc -Text '4 Results'
    $methodsStart = $methodsTitle.Paragraphs.Item(1).Range.End
    $methodsEnd = $resultsTitle.Paragraphs.Item(1).Range.Start
    $methodsRange = $doc.Range($methodsStart, $methodsEnd)
    $methodsRange.Text = $methods.Replace("`n", $p)
    $insertedMethods = $doc.Range($methodsStart, $methodsStart + $methods.Length)
    $insertedMethods.Font.Bold = 0
    $insertedMethods.HighlightColorIndex = $wdNoHighlight

    foreach ($heading in @(
        '3.1 Study design, data governance, and evidence hierarchy',
        '3.2 News data and out-of-sample sentiment construction',
        '3.3 Point-in-time contract, temporal splits, and frozen windows',
        '3.4 Numerical features and causal VMD',
        '3.5 Neural architectures and training',
        '3.6 Multimodal falsification and Bull/Bear/Leader evaluation',
        '3.7 Causal market regimes and SHAP selection',
        '3.8 Metrics and statistical inference'
    )) {
        Format-ParagraphAsHeading -Document $doc -Text $heading
    }
    Format-LeadLabel -Document $doc -Text 'Figure 1.'

    # Move the compact forward and transfer protocols into their Results subsections.
    $h45 = Find-TextRange -Document $doc -Text '4.5 Partial-2026 performance exposed majority-side collapse'
    $position45 = $h45.Paragraphs.Item(1).Range.End
    $text45 = $forward2026.Replace("`n", $p)
    $doc.Range($position45, $position45).Text = $text45
    $range45 = $doc.Range($position45, $position45 + $text45.Length)
    $range45.Font.Bold = 0
    $range45.HighlightColorIndex = $wdNoHighlight

    $h46 = Find-TextRange -Document $doc -Text '4.6 Frozen SET100 transfer was uniformly weaker'
    $position46 = $h46.Paragraphs.Item(1).Range.End
    $text46 = $set100.Replace("`n", $p)
    $doc.Range($position46, $position46).Text = $text46
    $range46 = $doc.Range($position46, $position46 + $text46.Length)
    $range46.Font.Bold = 0
    $range46.HighlightColorIndex = $wdNoHighlight

    # Replace the conclusion placeholder and add the requested end matter.
    $conclusionHeading = Find-TextRange -Document $doc -Text '5 Conclusion'
    $referencesHeading = Find-TextRange -Document $doc -Text 'References'
    $closingStart = $conclusionHeading.Paragraphs.Item(1).Range.Start
    $closingEnd = $referencesHeading.Paragraphs.Item(1).Range.Start
    $closingRange = $doc.Range($closingStart, $closingEnd)
    $closingRange.Text = $closing.Replace("`n", $p)
    $insertedClosing = $doc.Range($closingStart, $closingStart + $closing.Length)
    $insertedClosing.Font.Bold = 0
    $insertedClosing.HighlightColorIndex = $wdNoHighlight

    foreach ($heading in @('5 Conclusion', 'Acknowledgements', 'Reproducibility and data availability')) {
        Format-ParagraphAsHeading -Document $doc -Text $heading
    }
    Format-LeadLabel -Document $doc -Text 'Limitations.'
    Format-LeadLabel -Document $doc -Text 'Future work.'

    # Remove inherited draft-only markers if any survived outside the replaced ranges.
    foreach ($draftText in @('Method 11 -> 8', 'Results')) {
        if ($draftText -eq 'Results') { continue }
        $search = $doc.Content.Duplicate
        $find = $search.Find
        $find.ClearFormatting()
        $find.Text = $draftText
        $find.Wrap = $wdFindStop
        if ($find.Execute()) {
            $search.Paragraphs.Item(1).Range.Delete()
        }
    }

    $doc.Save()
    $doc.Close()
    $doc = $null
    $word.Quit()
    $word = $null
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

Get-Item -LiteralPath $outputFull | Select-Object FullName, Length, LastWriteTime
