$ErrorActionPreference = 'Stop'

$sourcePath = 'D:\SET50_direction_prediction_paper\paper\qa\original_spacing_input.docx'
$outputPath = 'D:\SET50_direction_prediction_paper\paper\newest_original_manuscript_spacing_corrected.docx'

$wdFindStop = 0
$wdCollapseEnd = 0
$wdAlignParagraphLeft = 0
$wdAlignParagraphCenter = 1
$wdAlignParagraphRight = 2
$wdAlignParagraphJustify = 3
$wdTabAlignmentCenter = 1
$wdTabAlignmentRight = 2
$wdTabLeaderSpaces = 0
$wdLineSpaceSingle = 0
$wdLineSpaceMultiple = 5
$wdAutoFitWindow = 2
$wdCellAlignVerticalCenter = 1
$wdFieldPage = 33
$wdFormatDocumentDefault = 16
$msoFalse = 0
$msoTrue = -1

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
    if (-not $find.Execute()) {
        throw "Text anchor not found: $Text"
    }
    return $range
}

function Replace-ExactText {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$OldText,
        [Parameter(Mandatory = $true)][string]$NewText
    )
    $range = Find-TextRange -Document $Document -Text $OldText
    $range.Text = $NewText
}

function Insert-BlockAfterParagraph {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Anchor,
        [Parameter(Mandatory = $true)][string[]]$Lines
    )
    $range = Find-TextRange -Document $Document -Text $Anchor
    $paragraphRange = $range.Paragraphs.Item(1).Range.Duplicate
    $paragraphRange.Collapse($wdCollapseEnd)
    $block = ($Lines -join "`r") + "`r"
    $insertStart = $paragraphRange.Start
    $paragraphRange.InsertAfter($block)
    $insertedRange = $Document.Range($insertStart, $insertStart + $block.Length)
    $insertedRange.Style = $Document.Styles.Item('Normal')
    $insertedRange.Font.Bold = $msoFalse
    $insertedRange.Font.Italic = $msoFalse
}

function Convert-MarkerToEquation {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$Marker,
        [Parameter(Mandatory = $true)][string]$LinearEquation,
        [Parameter(Mandatory = $true)][string]$Label,
        [double]$FontSize = 10.0
    )
    $markerRange = Find-TextRange -Document $Document -Text $Marker
    $paragraphRange = $markerRange.Paragraphs.Item(1).Range
    $paragraphStart = $paragraphRange.Start
    $paragraphRange.Text = "`t$LinearEquation`t$Label`r"

    $paragraphRange = $Document.Range($paragraphStart, $paragraphStart + $LinearEquation.Length + $Label.Length + 3)
    $usableWidth = $paragraphRange.Sections.Item(1).PageSetup.PageWidth - $paragraphRange.Sections.Item(1).PageSetup.LeftMargin - $paragraphRange.Sections.Item(1).PageSetup.RightMargin
    $paragraphRange.ParagraphFormat.TabStops.ClearAll()
    [void]$paragraphRange.ParagraphFormat.TabStops.Add($usableWidth / 2.0, $wdTabAlignmentCenter, $wdTabLeaderSpaces)
    [void]$paragraphRange.ParagraphFormat.TabStops.Add($usableWidth, $wdTabAlignmentRight, $wdTabLeaderSpaces)
    $paragraphRange.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    $paragraphRange.ParagraphFormat.SpaceBefore = 4
    $paragraphRange.ParagraphFormat.SpaceAfter = 4
    $paragraphRange.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
    $paragraphRange.ParagraphFormat.KeepTogether = $msoTrue
    $paragraphRange.ParagraphFormat.WidowControl = $msoTrue
    $paragraphRange.Font.Name = 'Times New Roman'
    $paragraphRange.Font.Size = $FontSize

    $equationStart = $paragraphStart + 1
    $equationEnd = $equationStart + $LinearEquation.Length
    $equationRange = $Document.Range($equationStart, $equationEnd)
    [void]$Document.OMaths.Add($equationRange)
}

function Format-DocumentLayout {
    param([Parameter(Mandatory = $true)]$Document)

    foreach ($section in $Document.Sections) {
        $section.PageSetup.PageWidth = $Document.Application.CentimetersToPoints(21.0)
        $section.PageSetup.PageHeight = $Document.Application.CentimetersToPoints(29.7)
        $section.PageSetup.TopMargin = $Document.Application.CentimetersToPoints(2.35)
        $section.PageSetup.BottomMargin = $Document.Application.CentimetersToPoints(2.35)
        $section.PageSetup.LeftMargin = $Document.Application.CentimetersToPoints(2.5)
        $section.PageSetup.RightMargin = $Document.Application.CentimetersToPoints(2.5)
        $section.PageSetup.HeaderDistance = $Document.Application.CentimetersToPoints(1.25)
        $section.PageSetup.FooterDistance = $Document.Application.CentimetersToPoints(1.25)
    }

    $normal = $Document.Styles.Item('Normal')
    $normal.Font.Name = 'Times New Roman'
    $normal.Font.Size = 11
    $normal.ParagraphFormat.Alignment = $wdAlignParagraphJustify
    $normal.ParagraphFormat.LineSpacingRule = $wdLineSpaceMultiple
    $normal.ParagraphFormat.LineSpacing = $Document.Application.LinesToPoints(1.12)
    $normal.ParagraphFormat.SpaceAfter = 5
    $normal.ParagraphFormat.WidowControl = $msoTrue

    foreach ($paragraph in $Document.Paragraphs) {
        $range = $paragraph.Range
        if ($range.Information(12)) { continue }
        $text = ($range.Text -replace "[`r`a`v]", '').Trim()
        if ([string]::IsNullOrWhiteSpace($text)) { continue }

        $range.Font.Name = 'Times New Roman'
        $range.Font.Size = 11
        $range.ParagraphFormat.WidowControl = $msoTrue

        if ($text -match '^\d+\.\s+' -or $text -eq 'Abstract' -or $text -eq 'References') {
            $range.Font.Size = 13
            $range.Font.Bold = $msoTrue
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceBefore = 14
            $range.ParagraphFormat.SpaceAfter = 6
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^\d+\.\d+\s+') {
            $range.Font.Size = 11.5
            $range.Font.Bold = $msoTrue
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceBefore = 10
            $range.ParagraphFormat.SpaceAfter = 4
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^Figure\s+\d+\.') {
            $range.Font.Size = 9.5
            $range.Font.Bold = $msoFalse
            $range.Font.Italic = $msoFalse
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceBefore = 3
            $range.ParagraphFormat.SpaceAfter = 8
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^Table\s+\d+[A-Za-z]?\.') {
            $range.Font.Size = 9.5
            $range.Font.Bold = $msoFalse
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceBefore = 7
            $range.ParagraphFormat.SpaceAfter = 3
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }

        if ($text -notmatch '^\[\[EQ_') {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphJustify
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceMultiple
            $range.ParagraphFormat.LineSpacing = $Document.Application.LinesToPoints(1.12)
            $range.ParagraphFormat.SpaceAfter = 5
        }
    }

    foreach ($table in $Document.Tables) {
        $table.Rows.Alignment = $wdAlignParagraphCenter
        $table.Rows.AllowBreakAcrossPages = $msoFalse
        $table.AllowAutoFit = $msoTrue
        $table.AutoFitBehavior($wdAutoFitWindow)
        $table.Range.Font.Name = 'Times New Roman'
        $table.Range.Font.Size = 8.5
        $table.Range.ParagraphFormat.SpaceBefore = 0
        $table.Range.ParagraphFormat.SpaceAfter = 0
        $table.Range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
        foreach ($cell in $table.Range.Cells) {
            $cell.VerticalAlignment = $wdCellAlignVerticalCenter
        }
        if ($table.Rows.Count -ge 1) {
            $table.Rows.Item(1).HeadingFormat = $msoTrue
            $table.Rows.Item(1).Range.Font.Bold = $msoTrue
            $table.Rows.Item(1).Range.ParagraphFormat.KeepWithNext = $msoTrue
        }
    }

    if ($Document.Sections.Count -ge 1) {
        $footer = $Document.Sections.Item(1).Footers.Item(1)
        if ($footer.PageNumbers.Count -eq 0) {
            [void]$footer.PageNumbers.Add($wdAlignParagraphRight, $true)
        }
        $footer.Range.Font.Name = 'Times New Roman'
        $footer.Range.Font.Size = 9
    }
}

function Format-OriginalManuscriptSpacing {
    param([Parameter(Mandatory = $true)]$Document)

    # Preserve the author's original manuscript system: A4/Letter-like one-column
    # layout, Times New Roman 12 pt, 1.5-line body spacing and 6 pt after paragraphs.
    # This function changes paragraph rhythm only; page geometry, line numbering,
    # tables, figures, fields and the original font hierarchy remain unchanged.
    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        $range = $paragraph.Range
        if ($range.Information(12)) { continue }
        $text = (($range.Text -replace "[`r`a`v]", '') -replace '\s+', ' ').Trim()

        if ([string]::IsNullOrWhiteSpace($text)) {
            $range.ParagraphFormat.SpaceBefore = 0
            $range.ParagraphFormat.SpaceAfter = 0
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            continue
        }

        $range.ParagraphFormat.WidowControl = $msoTrue
        $range.ParagraphFormat.KeepTogether = $msoFalse
        $range.ParagraphFormat.KeepWithNext = $msoFalse
        $range.ParagraphFormat.SpaceBefore = 0
        $range.ParagraphFormat.SpaceAfter = 6
        $range.ParagraphFormat.LineSpacingRule = 1

        if ($i -eq 1) {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceBefore = 0
            $range.ParagraphFormat.SpaceAfter = 12
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($i -eq 2) {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceAfter = 6
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($i -ge 4 -and $i -le 6) {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceAfter = 0
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($i -eq 7) {
            $range.ParagraphFormat.SpaceAfter = 12
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            continue
        }
        if ($text -eq 'Abstract') {
            $range.ParagraphFormat.SpaceBefore = 6
            $range.ParagraphFormat.SpaceAfter = 4
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($text -match '^Keywords:') {
            $range.ParagraphFormat.SpaceBefore = 0
            $range.ParagraphFormat.SpaceAfter = 12
            continue
        }
        if ($text -match '^\d+\.\s+' -or $text -eq 'References') {
            $range.ParagraphFormat.SpaceBefore = 12
            $range.ParagraphFormat.SpaceAfter = 6
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^\d+\.\d+\s+') {
            $range.ParagraphFormat.SpaceBefore = 9
            $range.ParagraphFormat.SpaceAfter = 4
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^Figure\s+\d+\.') {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceBefore = 3
            $range.ParagraphFormat.SpaceAfter = 9
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^Table\s+\d+[A-Za-z]?\.') {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceBefore = 9
            $range.ParagraphFormat.SpaceAfter = 4
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^Note:') {
            $range.ParagraphFormat.SpaceBefore = 3
            $range.ParagraphFormat.SpaceAfter = 6
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            continue
        }
        if ($range.InlineShapes.Count -gt 0) {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceBefore = 6
            $range.ParagraphFormat.SpaceAfter = 3
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($range.Style.NameLocal -eq 'List Paragraph') {
            $range.ParagraphFormat.SpaceAfter = 3
        }
    }
}

$word = $null
$document = $null
try {
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Source document not found: $sourcePath"
    }
    if (Test-Path -LiteralPath $outputPath) {
        Remove-Item -LiteralPath $outputPath -Force
    }

    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($sourcePath, $false, $false)
    $document.TrackRevisions = $false

    Replace-ExactText -Document $document `
        -OldText 'The regression target is y_t = C_(t+1) and direction is d_t = I(C_(t+1) > C_t).' `
        -NewText 'The point-in-time regression target and the realized and predicted directions are defined in Eq. (2).'

    Insert-BlockAfterParagraph -Document $document -Anchor 'The complete news-to-session construction is summarized in Figure 2.' -Lines @(
        'For each article-ticker pair, the scalar sentiment score and its session-level aggregation are defined in Eq. (1). Here, N_t is the set of pairs assigned to session t, n_t is its size, and c denotes the positive, neutral, or negative class. Article count, ticker-mention count, and news availability follow directly from N_t.',
        '[[EQ_1]]'
    )

    Insert-BlockAfterParagraph -Document $document -Anchor 'The point-in-time regression target and the realized and predicted directions are defined in Eq. (2).' -Lines @(
        '[[EQ_2]]'
    )

    Insert-BlockAfterParagraph -Document $document -Anchor 'VMD is tested as a causal auxiliary representation rather than assumed to improve direction.' -Lines @(
        'Equations (3a)-(3c) formalize the trailing decomposition window, highest-frequency noise-mode rule, denoised close, and removed-mode energy ratio. The quantities u_(k,t) and omega_(k,t) denote mode k estimated at date t and its final centre frequency, respectively.',
        '[[EQ_3A]]',
        '[[EQ_3B]]',
        '[[EQ_3C]]'
    )

    Insert-BlockAfterParagraph -Document $document -Anchor 'Every architecture uses a linear output.' -Lines @(
        'All five architectures minimize the mean squared error objective in Eq. (4), where n is the number of supervised training sequences.',
        '[[EQ_4]]'
    )

    Insert-BlockAfterParagraph -Document $document -Anchor '3.7 Causal market regimes and SHAP selection' -Lines @(
        'Daily regimes are assigned without future information using the causal trend score in Eq. (5a). H contains the six return horizons in Eq. (5b); sigma_(t,v(h)) is trailing volatility, with v(h)=20 sessions for h up to 20 and 60 sessions for h=60. ADX_(14,t) scales trend strength and EWMA_3 smooths the composite score.',
        '[[EQ_5A]]',
        '[[EQ_5B]]',
        'For outer fold f, the symmetric deadband lambda_f in Eq. (5c) is the 35th percentile of absolute training-fold scores only. Date t is Bull when z_t exceeds lambda_f, Bear when z_t is below -lambda_f, and Sideway otherwise. The state at t routes the prediction for t+1; targets and future columns are never used in regime construction.',
        '[[EQ_5C]]'
    )

    Insert-BlockAfterParagraph -Document $document -Anchor 'A one-standard-error rule with balanced-accuracy, model-error, RMSE and temporal-Jaccard guardrails selected the smallest stable subset.' -Lines @(
        'Within regime r, feature j is ranked by the mean absolute SHAP value in Eq. (6), computed only from the training ranking set T_r. The selected set S_r(k) contains the k highest-ranked features.',
        '[[EQ_6]]'
    )

    Insert-BlockAfterParagraph -Document $document -Anchor 'Seeds are averaged within each model-year cell before temporal inference and are not treated as independent market samples.' -Lines @(
        'Balanced accuracy is defined in Eq. (7), giving equal weight to sensitivity and specificity.',
        '[[EQ_7]]'
    )

    Insert-BlockAfterParagraph -Document $document -Anchor 'Economic summaries are exploratory and are not used to certify deployment.' -Lines @(
        'For paired arms A and B, Eq. (8a) first averages the five seed-level balanced-accuracy differences within each architecture m and outer year y, and then averages the four year effects. Equations (8b)-(8c) define the 16 sign-flipped test statistics and their exact two-sided probability.',
        '[[EQ_8A]]',
        '[[EQ_8B]]',
        '[[EQ_8C]]'
    )

    # Keep the original one-column manuscript structure. Figure 6 arrived with
    # its caption in the image paragraph; separate it and remove visible slash
    # placeholders without moving or replacing any figure.
    $figure6Range = Find-TextRange -Document $document -Text 'Figure 6.'
    if ($figure6Range.Paragraphs.Item(1).Range.InlineShapes.Count -gt 0) {
        $figure6Range.InsertBefore("`r")
    }
    for ($paragraphIndex = $document.Paragraphs.Count; $paragraphIndex -ge 1; $paragraphIndex--) {
        $figureParagraph = $document.Paragraphs.Item($paragraphIndex)
        if ($figureParagraph.Range.InlineShapes.Count -eq 0) { continue }
        $slashRange = $figureParagraph.Range.Duplicate
        $slashRange.Find.Text = '/'
        $slashRange.Find.Wrap = $wdFindStop
        if ($slashRange.Find.Execute()) { $slashRange.Text = '' }
    }

    Write-Output 'STAGE=blocks_inserted'
    Format-OriginalManuscriptSpacing -Document $document
    Write-Output 'STAGE=original_spacing_formatted'

    $sum = [char]0x2211
    $mu = [char]0x03BC
    $sigma = [char]0x03C3
    $rho = [char]0x03C1
    $omega = [char]0x03C9
    $tau = [char]0x03C4
    $lambda = [char]0x03BB
    $phi = [char]0x03D5
    $delta = [char]0x0394
    $epsilon = [char]0x03B5
    $inSet = [char]0x2208
    $notEqual = [char]0x2260
    $greaterEqual = [char]0x2265
    $chat = 'C' + [char]0x0302
    $dhat = 'd' + [char]0x0302
    $yhat = 'y' + [char]0x0302
    $cchat = 'c' + [char]0x0302
    $ctilde = 'C' + [char]0x0303
    $deltaMean = "$delta" + 'mean'

    $eq1 = "s_i=p_i^(+)-p_i^(-), $mu`_(s,t)=1/n_t $sum`_(i$inSet`N_t)s_i, q_(t,c)=1/n_t $sum`_(i$inSet`N_t)I($cchat`_i=c)"
    $eq2 = "y_t=C_(t+1), d_t=I(C_(t+1)>C_t), $dhat`_t=I($chat`_(t+1)>C_t)"
    $eq3a = "(u_(k,t),$omega`_(k,t))_(k=1)^5=VMD(C_(t-59):C_t), k_t^*=arg max_k $omega`_(k,t)"
    $eq3b = "$ctilde`_t=$sum`_(k$notEqual`k_t^*)u_(k,t)(60)"
    $eq3c = "$rho`_t=($sum`_($tau=1)^60 u_(k_t^*,t)^2($tau))/($sum`_($tau=1)^60 C_(t-60+$tau)^2)"
    $eq4 = "L_(MSE)=1/n $sum`_(t=1)^n(y_t-$yhat`_t)^2"
    $eq5a = "z_t=EWMA_3[(ADX_(14,t)/100)$sum`_(h$inSet`H) w_h R_(t,h)/($sigma`_(t,v(h)) sqrt(h))]"
    $eq5b = "H={1,3,5,10,20,60}, w=(0.05,0.10,0.15,0.20,0.25,0.25)"
    $eq5c = "$lambda`_f=Q_(0.35)({|z_t|:t$inSet`T_f})"
    $eq6 = "I_(r,j)=1/|T_r| $sum`_(i$inSet`T_r)|$phi`_(i,j)|, S_r(k)=TopK_j(I_(r,j))"
    $eq7 = "BAcc=1/2(TP/(TP+FN)+TN/(TN+FP))"
    $eq8a = "$delta`_(m,y)=1/5 $sum`_(s=1)^5(BAcc_(m,y,s)^A-BAcc_(m,y,s)^B), $deltaMean`_m=1/4 $sum`_(y=2022)^2025 $delta`_(m,y)"
    $eq8b = "T_j=1/4 $sum`_(y=1)^4 $epsilon`_(j,y)$delta`_(m,y), $epsilon`_(j,y)$inSet`{-1,+1}, j=1,...,16"
    $eq8c = "p_(exact)=1/16 $sum`_(j=1)^16 I(|T_j|$greaterEqual`|$deltaMean`_m|)"

    Convert-MarkerToEquation -Document $document -Marker '[[EQ_1]]' -LinearEquation $eq1 -Label '(1)' -FontSize 10.5
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_2]]' -LinearEquation $eq2 -Label '(2)' -FontSize 10.5
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_3A]]' -LinearEquation $eq3a -Label '(3a)' -FontSize 10.0
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_3B]]' -LinearEquation $eq3b -Label '(3b)' -FontSize 10.5
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_3C]]' -LinearEquation $eq3c -Label '(3c)' -FontSize 10.0
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_4]]' -LinearEquation $eq4 -Label '(4)' -FontSize 10.5
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_5A]]' -LinearEquation $eq5a -Label '(5a)' -FontSize 10.0
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_5B]]' -LinearEquation $eq5b -Label '(5b)' -FontSize 10.0
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_5C]]' -LinearEquation $eq5c -Label '(5c)' -FontSize 10.5
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_6]]' -LinearEquation $eq6 -Label '(6)' -FontSize 10.0
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_7]]' -LinearEquation $eq7 -Label '(7)' -FontSize 10.5
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_8A]]' -LinearEquation $eq8a -Label '(8a)' -FontSize 9.5
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_8B]]' -LinearEquation $eq8b -Label '(8b)' -FontSize 9.5
    Convert-MarkerToEquation -Document $document -Marker '[[EQ_8C]]' -LinearEquation $eq8c -Label '(8c)' -FontSize 9.5

    Write-Output 'STAGE=equations_added'
    $document.OMaths.BuildUp()
    Write-Output 'STAGE=equations_built_up'
    $document.Repaginate()
    Write-Output 'STAGE=repaginated'
    $document.SaveAs2($outputPath, $wdFormatDocumentDefault)
    Write-Output 'STAGE=saved'

    $markerCheck = $document.Content.Duplicate
    $markerCheck.Find.Text = '[[EQ_'
    $markerCheck.Find.Wrap = $wdFindStop
    $hasMarkers = $markerCheck.Find.Execute()
    Write-Output "OUTPUT=$outputPath"
    Write-Output "PAGES=$($document.ComputeStatistics(2))"
    Write-Output "TABLES=$($document.Tables.Count)"
    Write-Output "INLINE_SHAPES=$($document.InlineShapes.Count)"
    Write-Output "FIELDS=$($document.Fields.Count)"
    Write-Output "OMATHS=$($document.OMaths.Count)"
    Write-Output "MARKERS_REMAIN=$hasMarkers"
}
finally {
    if ($document -ne $null) { $document.Close($false) }
    if ($word -ne $null) { $word.Quit() }
    if ($document -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($word -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
