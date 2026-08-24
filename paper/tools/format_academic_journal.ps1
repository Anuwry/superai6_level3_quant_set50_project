$ErrorActionPreference = 'Stop'

$originalPath = 'D:\SET50_direction_prediction_paper\paper\newest_with_all_tables_and_equations_english.docx'
$inputPath = 'D:\SET50_direction_prediction_paper\paper\qa\journal_input_copy.docx'
$outputPath = 'D:\SET50_direction_prediction_paper\paper\newest_academic_journal_style.docx'

$wdCollapseStart = 1
$wdCollapseEnd = 0
$wdSectionBreakContinuous = 3
$wdAlignParagraphLeft = 0
$wdAlignParagraphCenter = 1
$wdAlignParagraphRight = 2
$wdAlignParagraphJustify = 3
$wdLineSpaceSingle = 0
$wdTabAlignmentCenter = 1
$wdTabAlignmentRight = 2
$wdTabLeaderSpaces = 0
$wdAutoFitWindow = 2
$wdPreferredWidthPoints = 3
$wdCellAlignVerticalCenter = 1
$wdBorderTop = -1
$wdBorderLeft = -2
$wdBorderBottom = -3
$wdBorderRight = -4
$wdBorderHorizontal = -5
$wdBorderVertical = -6
$wdLineStyleNone = 0
$wdLineStyleSingle = 1
$wdLineWidth025pt = 2
$wdLineWidth050pt = 4
$wdFormatDocumentDefault = 16
$wdInTable = 12
$wdFindStop = 0
$msoFalse = 0
$msoTrue = -1

function Clean-Text([string]$Text) {
    return (($Text -replace "[`r`a`v]", '') -replace '\s+', ' ').Trim()
}

function Find-TextRange {
    param($Document, [string]$Text)
    $range = $Document.Content.Duplicate
    $find = $range.Find
    $find.ClearFormatting()
    $find.Text = $Text
    $find.Forward = $true
    $find.Wrap = $wdFindStop
    if (-not $find.Execute()) { throw "Text not found: $Text" }
    return $range
}

function Set-RangeFont {
    param($Range, [string]$Name, [double]$Size, $Bold = $null, $Italic = $null)
    $Range.Font.Name = $Name
    try { $Range.Font.NameAscii = $Name } catch {}
    $Range.Font.Size = $Size
    if ($null -ne $Bold) { $Range.Font.Bold = $Bold }
    if ($null -ne $Italic) { $Range.Font.Italic = $Italic }
}

function Remove-PlaceholderSlash {
    param($Paragraph)
    $range = $Paragraph.Range.Duplicate
    $find = $range.Find
    $find.Text = '/'
    $find.Forward = $true
    $find.Wrap = $wdFindStop
    while ($find.Execute()) {
        $range.Text = ''
        $range = $Paragraph.Range.Duplicate
        $find = $range.Find
        $find.Text = '/'
        $find.Forward = $true
        $find.Wrap = $wdFindStop
    }
}

function Add-ContinuousBreak {
    param($Document, [int]$Position)
    $range = $Document.Range($Position, $Position)
    $range.InsertBreak($wdSectionBreakContinuous)
}

function Get-DisplayBlocks {
    param($Document)
    $blocks = New-Object System.Collections.Generic.List[object]

    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        $text = Clean-Text $paragraph.Range.Text
        if ($text -match '^Figure\s+\d+\.') {
            $start = $paragraph.Range.Start
            for ($j = $i - 1; $j -ge 1; $j--) {
                $candidate = $Document.Paragraphs.Item($j)
                if ($candidate.Range.InlineShapes.Count -gt 0) {
                    $start = $candidate.Range.Start
                    break
                }
                $candidateText = Clean-Text $candidate.Range.Text
                if (-not [string]::IsNullOrWhiteSpace($candidateText)) { break }
            }
            $blocks.Add([pscustomobject]@{ Start = [int]$start; End = [int]$paragraph.Range.End; Kind = 'Figure' })
        }
    }

    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        $text = Clean-Text $paragraph.Range.Text
        if ($text -match '^Table\s+\d+[A-Za-z]?\.') {
            $tableFound = $null
            for ($t = 1; $t -le $Document.Tables.Count; $t++) {
                $table = $Document.Tables.Item($t)
                if ($table.Range.Start -ge $paragraph.Range.End) {
                    $tableFound = $table
                    break
                }
            }
            if ($null -eq $tableFound) { throw "No table found after caption: $text" }
            $end = $tableFound.Range.End
            for ($j = 1; $j -le $Document.Paragraphs.Count; $j++) {
                $candidate = $Document.Paragraphs.Item($j)
                if ($candidate.Range.Start -lt $tableFound.Range.End) { continue }
                $candidateText = Clean-Text $candidate.Range.Text
                if ([string]::IsNullOrWhiteSpace($candidateText)) { continue }
                if ($candidateText -match '^Note:') { $end = $candidate.Range.End }
                break
            }
            $blocks.Add([pscustomobject]@{ Start = [int]$paragraph.Range.Start; End = [int]$end; Kind = 'Table' })
        }
    }

    $sorted = @($blocks | Sort-Object Start, End)
    $merged = New-Object System.Collections.Generic.List[object]
    foreach ($block in $sorted) {
        if ($merged.Count -eq 0) {
            $merged.Add([pscustomobject]@{ Start = $block.Start; End = $block.End })
            continue
        }
        $last = $merged[$merged.Count - 1]
        if ($block.Start -le ($last.End + 2)) {
            if ($block.End -gt $last.End) { $last.End = $block.End }
        }
        else {
            $merged.Add([pscustomobject]@{ Start = $block.Start; End = $block.End })
        }
    }
    return $merged | ForEach-Object { $_ }
}

function Configure-SectionPage {
    param($Document, $Section, [bool]$TwoColumns)
    $setup = $Section.PageSetup
    $setup.PageWidth = $Document.Application.CentimetersToPoints(21.0)
    $setup.PageHeight = $Document.Application.CentimetersToPoints(29.7)
    $setup.TopMargin = $Document.Application.CentimetersToPoints(1.8)
    $setup.BottomMargin = $Document.Application.CentimetersToPoints(1.8)
    $setup.LeftMargin = $Document.Application.CentimetersToPoints(2.0)
    $setup.RightMargin = $Document.Application.CentimetersToPoints(2.0)
    $setup.HeaderDistance = $Document.Application.CentimetersToPoints(0.8)
    $setup.FooterDistance = $Document.Application.CentimetersToPoints(0.8)
    try { $setup.LineNumbering.Active = $false } catch {}
    if ($TwoColumns) {
        $setup.TextColumns.SetCount(2)
        $setup.TextColumns.Spacing = $Document.Application.CentimetersToPoints(0.65)
        $setup.TextColumns.LineBetween = $false
    }
    else {
        $setup.TextColumns.SetCount(1)
    }
}

function Set-MinimalTableBorders {
    param($Table)
    foreach ($borderId in @($wdBorderTop, $wdBorderLeft, $wdBorderBottom, $wdBorderRight, $wdBorderHorizontal, $wdBorderVertical)) {
        $Table.Borders.Item($borderId).LineStyle = $wdLineStyleNone
    }
    $Table.Borders.Item($wdBorderTop).LineStyle = $wdLineStyleSingle
    $Table.Borders.Item($wdBorderTop).LineWidth = $wdLineWidth050pt
    $Table.Borders.Item($wdBorderBottom).LineStyle = $wdLineStyleSingle
    $Table.Borders.Item($wdBorderBottom).LineWidth = $wdLineWidth050pt
    $Table.Borders.Item($wdBorderHorizontal).LineStyle = $wdLineStyleSingle
    $Table.Borders.Item($wdBorderHorizontal).LineWidth = $wdLineWidth025pt
    $Table.Borders.Item($wdBorderHorizontal).Color = 12632256
    if ($Table.Rows.Count -ge 1) {
        $header = $Table.Rows.Item(1)
        $header.Range.Font.Bold = $msoTrue
        $header.HeadingFormat = $msoTrue
        $header.Shading.BackgroundPatternColor = 15132390
        $header.Borders.Item($wdBorderBottom).LineStyle = $wdLineStyleSingle
        $header.Borders.Item($wdBorderBottom).LineWidth = $wdLineWidth050pt
    }
}

Copy-Item -LiteralPath $originalPath -Destination $inputPath -Force
if (Test-Path -LiteralPath $outputPath) { Remove-Item -LiteralPath $outputPath -Force }

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($inputPath, $false, $false)
    $document.TrackRevisions = $false

    # Remove the accidentally duplicated figure from the title paragraph.
    $titleParagraph = $document.Paragraphs.Item(1)
    while ($titleParagraph.Range.InlineShapes.Count -gt 0) {
        $titleParagraph.Range.InlineShapes.Item(1).Delete()
    }
    $titleParagraph.Range.Text = "Multimodal and Regime-Aware Deep Learning for Next-Day SET Index Direction Forecasting`r"

    # Remove image placeholder slashes while keeping the inline figures.
    for ($i = $document.Paragraphs.Count; $i -ge 1; $i--) {
        $paragraph = $document.Paragraphs.Item($i)
        if ($paragraph.Range.InlineShapes.Count -gt 0) { Remove-PlaceholderSlash -Paragraph $paragraph }
    }

    # Separate Figure 6 caption from the image paragraph if they share one paragraph.
    $figure6 = Find-TextRange -Document $document -Text 'Figure 6.'
    if ($figure6.Paragraphs.Item(1).Range.InlineShapes.Count -gt 0) {
        $figure6.InsertBefore("`r")
    }

    # Clear drafting highlights, comments display residue and line-number formatting.
    $document.Content.HighlightColorIndex = 0

    # Identify full-width figure/table blocks, then insert continuous section breaks in reverse order.
    $blocks = Get-DisplayBlocks -Document $document
    foreach ($block in @($blocks | Sort-Object Start -Descending)) {
        Add-ContinuousBreak -Document $document -Position $block.End
        Add-ContinuousBreak -Document $document -Position $block.Start
    }

    # Title/abstract remain one column; the article begins in two columns at Introduction.
    $introRange = Find-TextRange -Document $document -Text '1. Introduction'
    Add-ContinuousBreak -Document $document -Position $introRange.Paragraphs.Item(1).Range.Start

    # Configure every section: display blocks are full width; narrative sections are two-column.
    for ($s = 1; $s -le $document.Sections.Count; $s++) {
        $section = $document.Sections.Item($s)
        $isFront = (Clean-Text $section.Range.Text) -match '^Multimodal and Regime-Aware'
        $isDisplay = ($section.Range.InlineShapes.Count -gt 0 -or $section.Range.Tables.Count -gt 0)
        Configure-SectionPage -Document $document -Section $section -TwoColumns:(-not $isFront -and -not $isDisplay)
    }

    # Resolve a journal-style typography system through real Word styles.
    $normal = $document.Styles.Item('Normal')
    Set-RangeFont -Range $normal -Name 'Times New Roman' -Size 9.5
    $normal.ParagraphFormat.Alignment = $wdAlignParagraphJustify
    $normal.ParagraphFormat.SpaceBefore = 0
    $normal.ParagraphFormat.SpaceAfter = 3
    $normal.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
    $normal.ParagraphFormat.WidowControl = $msoTrue

    $listStyle = $document.Styles.Item('List Paragraph')
    Set-RangeFont -Range $listStyle -Name 'Times New Roman' -Size 9.2
    $listStyle.ParagraphFormat.Alignment = $wdAlignParagraphJustify
    $listStyle.ParagraphFormat.SpaceAfter = 2
    $listStyle.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle

    $inReferences = $false
    for ($i = 1; $i -le $document.Paragraphs.Count; $i++) {
        $paragraph = $document.Paragraphs.Item($i)
        $range = $paragraph.Range
        if ($range.Information($wdInTable)) { continue }
        $text = Clean-Text $range.Text
        if ([string]::IsNullOrWhiteSpace($text)) { continue }

        Set-RangeFont -Range $range -Name 'Times New Roman' -Size 9.5
        $range.ParagraphFormat.Alignment = $wdAlignParagraphJustify
        $range.ParagraphFormat.SpaceBefore = 0
        $range.ParagraphFormat.SpaceAfter = 3
        $range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
        $range.ParagraphFormat.WidowControl = $msoTrue
        $range.ParagraphFormat.KeepTogether = $msoFalse
        $range.ParagraphFormat.KeepWithNext = $msoFalse
        $range.ParagraphFormat.LeftIndent = 0
        $range.ParagraphFormat.RightIndent = 0
        $range.ParagraphFormat.FirstLineIndent = 0

        if ($i -eq 1) {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 16 -Bold $msoTrue
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceBefore = 5
            $range.ParagraphFormat.SpaceAfter = 6
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($i -eq 2) {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 10.5
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceAfter = 3
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($i -ge 4 -and $i -le 6) {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 8.5
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceAfter = 1
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($i -eq 7) {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 8.5 -Italic $msoTrue
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceAfter = 7
            continue
        }
        if ($text -eq 'Abstract') {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 10 -Bold $msoTrue
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceBefore = 4
            $range.ParagraphFormat.SpaceAfter = 2
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($i -eq 9) {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 9
            $range.ParagraphFormat.Alignment = $wdAlignParagraphJustify
            $range.ParagraphFormat.SpaceAfter = 4
            continue
        }
        if ($text -match '^Keywords:') {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 8.5
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceAfter = 8
            continue
        }
        if ($text -eq 'References') { $inReferences = $true }

        if ($text -match '^\d+\.\s+' -or $text -eq 'References') {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 11.5 -Bold $msoTrue
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceBefore = 8
            $range.ParagraphFormat.SpaceAfter = 4
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^\d+\.\d+\s+') {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 10 -Bold $msoTrue
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceBefore = 6
            $range.ParagraphFormat.SpaceAfter = 2
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^Figure\s+\d+\.') {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 8
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceBefore = 2
            $range.ParagraphFormat.SpaceAfter = 5
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^Table\s+\d+[A-Za-z]?\.') {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 8.2
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceBefore = 4
            $range.ParagraphFormat.SpaceAfter = 2
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            $range.ParagraphFormat.KeepTogether = $msoTrue
            continue
        }
        if ($text -match '^Note:') {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 7.5 -Italic $msoTrue
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.SpaceBefore = 2
            $range.ParagraphFormat.SpaceAfter = 4
            continue
        }
        if ($range.InlineShapes.Count -gt 0) {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $range.ParagraphFormat.SpaceBefore = 4
            $range.ParagraphFormat.SpaceAfter = 2
            $range.ParagraphFormat.KeepWithNext = $msoTrue
            continue
        }
        if ($inReferences) {
            Set-RangeFont -Range $range -Name 'Times New Roman' -Size 8
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.LeftIndent = $document.Application.CentimetersToPoints(0.45)
            $range.ParagraphFormat.FirstLineIndent = -$document.Application.CentimetersToPoints(0.45)
            $range.ParagraphFormat.SpaceAfter = 2
        }
    }

    # Superscript author/affiliation indices in the author line.
    $authorRange = $document.Paragraphs.Item(2).Range.Duplicate
    foreach ($token in @('1,2', '3')) {
        $find = $authorRange.Duplicate
        $find.Find.Text = $token
        $find.Find.Wrap = $wdFindStop
        if ($find.Find.Execute()) { $find.Font.Superscript = $msoTrue }
    }

    # Scale and center figures across the full display width.
    $maxFigureWidth = $document.Application.CentimetersToPoints(16.6)
    for ($i = 1; $i -le $document.InlineShapes.Count; $i++) {
        $shape = $document.InlineShapes.Item($i)
        $shape.LockAspectRatio = $msoTrue
        if ($shape.Width -gt $maxFigureWidth) { $shape.Width = $maxFigureWidth }
        $shape.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
    }

    # Journal tables: full display width, compact typography, repeated headers and restrained rules.
    $tableWidth = $document.Application.CentimetersToPoints(16.6)
    for ($i = 1; $i -le $document.Tables.Count; $i++) {
        $table = $document.Tables.Item($i)
        $table.AllowAutoFit = $msoTrue
        $table.AutoFitBehavior($wdAutoFitWindow)
        $table.PreferredWidthType = $wdPreferredWidthPoints
        $table.PreferredWidth = $tableWidth
        $table.Rows.AllowBreakAcrossPages = $msoFalse
        $table.Rows.Alignment = $wdAlignParagraphCenter
        $table.TopPadding = 2
        $table.BottomPadding = 2
        $table.LeftPadding = 3
        $table.RightPadding = 3
        Set-RangeFont -Range $table.Range -Name 'Times New Roman' -Size 7.5
        $table.Range.ParagraphFormat.SpaceBefore = 0
        $table.Range.ParagraphFormat.SpaceAfter = 0
        $table.Range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
        foreach ($cell in $table.Range.Cells) {
            $cell.VerticalAlignment = $wdCellAlignVerticalCenter
        }
        Set-MinimalTableBorders -Table $table
    }

    # Reset equation tab stops to the actual column width and retain right-aligned numbering.
    for ($i = 1; $i -le $document.OMaths.Count; $i++) {
        $equation = $document.OMaths.Item($i)
        $equationParagraph = $equation.Range.Paragraphs.Item(1).Range
        $section = $equationParagraph.Sections.Item(1)
        $usableWidth = $section.PageSetup.PageWidth - $section.PageSetup.LeftMargin - $section.PageSetup.RightMargin
        if ($section.PageSetup.TextColumns.Count -eq 2) {
            $usableWidth = ($usableWidth - $section.PageSetup.TextColumns.Spacing) / 2.0
            $equationParagraph.Font.Size = 8.0
        }
        else {
            $equationParagraph.Font.Size = 8.8
        }
        $equationParagraph.ParagraphFormat.TabStops.ClearAll()
        [void]$equationParagraph.ParagraphFormat.TabStops.Add($usableWidth / 2.0, $wdTabAlignmentCenter, $wdTabLeaderSpaces)
        [void]$equationParagraph.ParagraphFormat.TabStops.Add($usableWidth, $wdTabAlignmentRight, $wdTabLeaderSpaces)
        $equationParagraph.ParagraphFormat.Alignment = $wdAlignParagraphLeft
        $equationParagraph.ParagraphFormat.SpaceBefore = 2
        $equationParagraph.ParagraphFormat.SpaceAfter = 2
        $equationParagraph.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
        $equationParagraph.ParagraphFormat.KeepTogether = $msoTrue
    }

    # Use one unobtrusive page number and remove inherited duplicate headers/footers.
    for ($s = 1; $s -le $document.Sections.Count; $s++) {
        $section = $document.Sections.Item($s)
        $section.PageSetup.DifferentFirstPageHeaderFooter = $false
        if ($s -eq 1) {
            $section.Headers.Item(1).LinkToPrevious = $false
            $section.Footers.Item(1).LinkToPrevious = $false
            $section.Headers.Item(1).Range.Text = ''
            $section.Footers.Item(1).Range.Text = ''
            [void]$section.Footers.Item(1).PageNumbers.Add($wdAlignParagraphCenter, $true)
            Set-RangeFont -Range $section.Footers.Item(1).Range -Name 'Times New Roman' -Size 8
        }
        else {
            $section.Headers.Item(1).LinkToPrevious = $true
            $section.Footers.Item(1).LinkToPrevious = $true
        }
    }

    $document.OMaths.BuildUp()
    $document.Repaginate()
    $document.SaveAs2($outputPath, $wdFormatDocumentDefault)

    Write-Output "OUTPUT=$outputPath"
    Write-Output "PAGES=$($document.ComputeStatistics(2))"
    Write-Output "SECTIONS=$($document.Sections.Count)"
    Write-Output "TABLES=$($document.Tables.Count)"
    Write-Output "FIGURES=$($document.InlineShapes.Count)"
    Write-Output "OMATHS=$($document.OMaths.Count)"
    Write-Output "FIELDS=$($document.Fields.Count)"
}
finally {
    if ($document -ne $null) { $document.Close($false) }
    if ($word -ne $null) { $word.Quit() }
    if ($document -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($word -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
