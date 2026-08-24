param(
    [string]$ProjectRoot = "D:\SET50_direction_prediction_paper"
)

$ErrorActionPreference = "Stop"

function Rgb([int]$r, [int]$g, [int]$b) {
    return $r + (256 * $g) + (65536 * $b)
}

$INK = Rgb 22 23 24
$MUTED = Rgb 82 85 88
$LIGHT = Rgb 245 245 242
$PANEL = Rgb 235 235 230
$RULE = Rgb 188 191 196
$ORANGE = Rgb 255 107 53
$PALE_ORANGE = Rgb 255 235 226
$PALE_BLUE = Rgb 220 231 241
$PALE_GREEN = Rgb 225 235 218
$WHITE = Rgb 255 255 255

function Add-Text {
    param($Slide, [string]$Text, [double]$X, [double]$Y, [double]$W, [double]$H,
          [double]$Size = 18, [bool]$Bold = $false, [int]$Color = $INK,
          [int]$Align = 1, [int]$VAlign = 1, [string]$Font = "Arial")
    $shape = $Slide.Shapes.AddTextbox(1, $X, $Y, $W, $H)
    $shape.Line.Visible = 0
    $shape.Fill.Visible = 0
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.VerticalAnchor = $VAlign
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = $Font
    $shape.TextFrame.TextRange.Font.Size = $Size
    $shape.TextFrame.TextRange.Font.Bold = $(if ($Bold) { -1 } else { 0 })
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Align
    return $shape
}

function Add-Panel {
    param($Slide, [double]$X, [double]$Y, [double]$W, [double]$H,
          [int]$Fill = $PANEL, [int]$Line = $RULE, [double]$RadiusType = 5)
    $shape = $Slide.Shapes.AddShape($RadiusType, $X, $Y, $W, $H)
    $shape.Fill.ForeColor.RGB = $Fill
    $shape.Fill.Solid()
    $shape.Line.ForeColor.RGB = $Line
    $shape.Line.Weight = 0.8
    return $shape
}

function Add-Line {
    param($Slide, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2,
          [int]$Color = $RULE, [double]$Weight = 1.0)
    $line = $Slide.Shapes.AddLine($X1, $Y1, $X2, $Y2)
    $line.Line.ForeColor.RGB = $Color
    $line.Line.Weight = $Weight
    return $line
}

function Add-FitImage {
    param($Slide, [string]$Path, [double]$X, [double]$Y, [double]$W, [double]$H,
          [bool]$Panel = $true)
    if ($Panel) {
        $bg = Add-Panel $Slide ($X - 6) ($Y - 6) ($W + 12) ($H + 12) $WHITE $RULE 1
        $bg.Shadow.Visible = 0
    }
    $pic = $Slide.Shapes.AddPicture($Path, 0, -1, $X, $Y, -1, -1)
    $pic.LockAspectRatio = -1
    $scale = [Math]::Min($W / $pic.Width, $H / $pic.Height)
    $pic.Width = $pic.Width * $scale
    $pic.Height = $pic.Height * $scale
    $pic.Left = $X + (($W - $pic.Width) / 2)
    $pic.Top = $Y + (($H - $pic.Height) / 2)
    return $pic
}

function Add-Header {
    param($Slide, [string]$Title, [int]$Number, [string]$Kicker)
    $bar = $Slide.Shapes.AddShape(1, 42, 27, 52, 4)
    $bar.Fill.ForeColor.RGB = $ORANGE
    $bar.Fill.Solid()
    $bar.Line.Visible = 0
    Add-Text $Slide $Kicker 106 19 300 18 11 $true $MUTED 1 3 | Out-Null
    Add-Text $Slide ("{0:D2}" -f $Number) 877 18 40 20 12 $true $MUTED 3 3 | Out-Null
    Add-Text $Slide $Title 42 45 860 66 35 $true $INK 1 1 | Out-Null
}

function Add-Footer {
    param($Slide, [string]$Text = "SET next-day direction reliability framework")
    Add-Line $Slide 42 515 918 515 $RULE 0.6 | Out-Null
    Add-Text $Slide $Text 42 520 700 13 9 $false $MUTED 1 1 | Out-Null
}

function Add-Note {
    param($Slide, [string]$Text)
    try {
        foreach ($shape in $Slide.NotesPage.Shapes) {
            if ($shape.Type -eq 14 -and $shape.PlaceholderFormat.Type -eq 2) {
                $shape.TextFrame.TextRange.Text = $Text
                return
            }
        }
    } catch {
        Write-Warning "Could not add notes to slide $($Slide.SlideIndex): $($_.Exception.Message)"
    }
}

function Add-BulletLine {
    param($Slide, [string]$Number, [string]$Title, [string]$Body,
          [double]$X, [double]$Y, [double]$W, [int]$Fill = $LIGHT)
    $circle = $Slide.Shapes.AddShape(9, $X, $Y + 1, 30, 30)
    $circle.Fill.ForeColor.RGB = $ORANGE
    $circle.Fill.Solid()
    $circle.Line.Visible = 0
    Add-Text $Slide $Number $X ($Y + 2) 30 28 13 $true $WHITE 2 3 | Out-Null
    Add-Text $Slide $Title ($X + 43) $Y ($W - 43) 22 17 $true $INK 1 1 | Out-Null
    Add-Text $Slide $Body ($X + 43) ($Y + 23) ($W - 43) 31 13 $false $MUTED 1 1 | Out-Null
}

$assets = @{
    Overall = Join-Path $ProjectRoot "paper\figures\methods_v2_simple\figure1_overall_reliability_audit_pipeline.png"
    PIT = Join-Path $ProjectRoot "paper\figures\methods_v2_simple\figure3_point_in_time_expanding_window_design.png"
    VMD = Join-Path $ProjectRoot "paper\figures\methods_v2_simple\figure4_vmd_and_registered_architectures.png"
    LLM = Join-Path $ProjectRoot "paper\assets\figure5_separated_audits_v2.png"
    XAI = Join-Path $ProjectRoot "paper\assets\figure10_shap_lime_result_audit.png"
    Heatmap = Join-Path $ProjectRoot "paper\assets\figure7_heatmap_summary.png"
    Scatter = Join-Path $ProjectRoot "outputs\final_five_model_prediction_visuals_v1\observed_vs_predicted_scatter_oos_2022_2025.png"
    Timeline = Join-Path $ProjectRoot "outputs\final_five_model_prediction_visuals_v1\actual_vs_predicted_oos_2025_zoom.png"
}

foreach ($entry in $assets.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value)) {
        throw "Missing required asset: $($entry.Value)"
    }
}

$contentPath = Join-Path $ProjectRoot "paper\tools\set50_progress_deck_content.json"
$notes = (Get-Content -LiteralPath $contentPath -Raw -Encoding UTF8 | ConvertFrom-Json).notes
$outputDir = Join-Path $ProjectRoot "outputs\presentation_progress_v1"
$renderDir = Join-Path $outputDir "rendered_slides"
New-Item -ItemType Directory -Force -Path $outputDir, $renderDir | Out-Null
$pptxPath = Join-Path $outputDir "SET50_reliability_framework_progress_presentation.pptx"
$pdfPath = Join-Path $outputDir "SET50_reliability_framework_progress_presentation.pdf"
if (Test-Path -LiteralPath $pptxPath) { Remove-Item -LiteralPath $pptxPath -Force }
if (Test-Path -LiteralPath $pdfPath) { Remove-Item -LiteralPath $pdfPath -Force }

$powerPoint = New-Object -ComObject PowerPoint.Application
$powerPoint.Visible = -1
$presentation = $powerPoint.Presentations.Add()
$presentation.PageSetup.SlideSize = 15

try {
    # Slide 1: Cover
    $s = $presentation.Slides.Add(1, 12)
    $s.FollowMasterBackground = 0
    $s.Background.Fill.ForeColor.RGB = $WHITE
    $s.Background.Fill.Solid()
    $accent = $s.Shapes.AddShape(1, 0, 0, 15, 540)
    $accent.Fill.ForeColor.RGB = $ORANGE
    $accent.Fill.Solid(); $accent.Line.Visible = 0
    Add-Text $s "RESEARCH PROGRESS" 48 43 300 20 13 $true $ORANGE 1 1 | Out-Null
    Add-Text $s "Evaluating Multimodal and Regime-Aware Deep Learning for Next-Day SET Index Direction Forecasting" 48 88 530 210 50 $true $INK 1 1 | Out-Null
    Add-Text $s "A point-in-time reliability framework for numerical, news, regime and transfer testing" 50 317 505 60 22 $false $MUTED 1 1 | Out-Null
    Add-Line $s 50 398 550 398 $RULE 1.0 | Out-Null
    Add-Text $s "arsanchai.su@wu.ac.th  |  10-minute engineering presentation" 50 415 520 25 15 $false $MUTED 1 1 | Out-Null
    Add-Panel $s 612 55 294 420 $LIGHT $RULE 5 | Out-Null
    Add-FitImage $s $assets.Overall 628 82 262 330 $false | Out-Null
    Add-Text $s "INPUT  ->  AUDIT  ->  EVIDENCE" 635 432 250 22 14 $true $INK 2 3 | Out-Null
    Add-Note $s $notes[0]

    # Slide 2: Problem
    $s = $presentation.Slides.Add(2, 12)
    Add-Header $s "A close price forecast can still miss tomorrow's direction" 2 "PROBLEM"
    Add-Panel $s 43 131 276 330 $LIGHT $RULE 5 | Out-Null
    Add-Text $s "Daily direction is a weak, non-stationary signal" 64 158 230 60 23 $true $INK 1 1 | Out-Null
    Add-Line $s 64 229 284 229 $ORANGE 2 | Out-Null
    Add-Text $s "Level error and directional accuracy measure different behavior." 64 249 224 60 17 $false $MUTED 1 1 | Out-Null
    Add-Text $s "Temporal leakage, repeated selection and regime dependence can create fragile gains." 64 332 224 76 17 $false $MUTED 1 1 | Out-Null
    Add-Text $s "Question: which improvements remain reliable?" 64 418 224 28 16 $true $ORANGE 1 1 | Out-Null
    Add-FitImage $s $assets.Scatter 345 132 570 330 $true | Out-Null
    Add-Footer $s "Observed versus predicted SET50 levels, out-of-sample 2022-2025"
    Add-Note $s $notes[1]

    # Slide 3: Framework
    $s = $presentation.Slides.Add(3, 12)
    Add-Header $s "One framework audits five enhancement classes under a common protocol" 3 "SYSTEM"
    Add-FitImage $s $assets.Overall 43 135 585 335 $true | Out-Null
    Add-Text $s "Five reliability dimensions" 665 137 240 26 20 $true $INK 1 1 | Out-Null
    Add-BulletLine $s "1" "Point-in-time data" "No future information in features or labels" 665 178 245 | Out-Null
    Add-BulletLine $s "2" "Causal denoising" "VMD tested against matched raw inputs" 665 237 245 | Out-Null
    Add-BulletLine $s "3" "News and LLM" "Downstream value separated from intrinsic sentiment" 665 296 245 | Out-Null
    Add-BulletLine $s "4" "Regime and XAI" "Bull/Sideway/Bear routing with SHAP and LIME" 665 355 245 | Out-Null
    Add-BulletLine $s "5" "Forward and transfer" "Later-period and SET100 robustness" 665 414 245 | Out-Null
    Add-Footer $s
    Add-Note $s $notes[2]

    # Slide 4: PIT design
    $s = $presentation.Slides.Add(4, 12)
    Add-Header $s "Every forecast uses only information available by the close of day t" 4 "DATA CONTRACT"
    $metrics = @(
        @("2012-2025", "SET50 daily market data"),
        @("2022-2025", "Four held-out outer years"),
        @("962 sessions", "Evaluated with five fixed seeds")
    )
    $yy = 139
    foreach ($m in $metrics) {
        Add-Panel $s 43 $yy 265 88 $LIGHT $RULE 5 | Out-Null
        Add-Text $s $m[0] 62 ($yy + 15) 225 32 25 $true $INK 1 1 | Out-Null
        Add-Text $s $m[1] 62 ($yy + 50) 225 22 14 $false $MUTED 1 1 | Out-Null
        $yy += 101
    }
    Add-FitImage $s $assets.PIT 345 139 570 304 $true | Out-Null
    Add-Panel $s 345 456 570 41 $PALE_ORANGE $ORANGE 5 | Out-Null
    Add-Text $s "Labels are purged by observation date; every transform is fitted on training data only." 365 466 530 22 15 $true $INK 2 3 | Out-Null
    Add-Footer $s
    Add-Note $s $notes[3]

    # Slide 5: Numerical and VMD
    $s = $presentation.Slides.Add(5, 12)
    Add-Header $s "Causal VMD was tested across five frozen architectures" 5 "NUMERICAL TRACK"
    Add-FitImage $s $assets.VMD 43 135 585 335 $true | Out-Null
    Add-Panel $s 661 140 250 110 $PALE_ORANGE $ORANGE 5 | Out-Null
    Add-Text $s "-0.60 to +0.35 pp" 680 160 212 38 27 $true $ORANGE 2 3 | Out-Null
    Add-Text $s "Balanced-accuracy effect across architectures" 683 204 206 34 14 $false $MUTED 2 1 | Out-Null
    Add-Text $s "Registered architectures" 665 277 230 24 18 $true $INK 1 1 | Out-Null
    Add-Text $s "LSTM\nCNN\nLSTM-CNN\nLSTM-Attention\nLSTM-CNN-Attention" 665 311 230 126 17 $false $INK 1 1 | Out-Null
    Add-Text $s "No conclusive general VMD gain" 665 449 235 24 16 $true $ORANGE 1 1 | Out-Null
    Add-Footer $s
    Add-Note $s $notes[4]

    # Slide 6: News and debate
    $s = $presentation.Slides.Add(6, 12)
    Add-Header $s "News forecasting and LLM debate answer different questions" 6 "MULTIMODAL + LLM"
    Add-FitImage $s $assets.LLM 52 130 856 292 $true | Out-Null
    Add-Panel $s 54 446 407 52 $LIGHT $RULE 5 | Out-Null
    Add-Text $s "Downstream market forecast" 70 455 190 20 16 $true $INK 1 1 | Out-Null
    Add-Text $s "No robust incremental BAcc gain" 70 476 360 16 13 $false $MUTED 1 1 | Out-Null
    Add-Panel $s 499 446 409 52 $PALE_ORANGE $ORANGE 5 | Out-Null
    Add-Text $s "Locked intrinsic sentiment" 515 455 190 20 16 $true $INK 1 1 | Out-Null
    Add-Text $s "Leader +5.93 / +6.00 pp vs matched controls" 515 476 365 16 13 $false $MUTED 1 1 | Out-Null
    Add-Footer $s
    Add-Note $s $notes[5]

    # Slide 7: XAI
    $s = $presentation.Slides.Add(7, 12)
    Add-Header $s "Regime-SHAP helped CNN, but explanations did not generalize" 7 "REGIME + XAI"
    Add-Panel $s 43 134 275 330 $LIGHT $RULE 5 | Out-Null
    Add-Text $s "+1.46 pp" 63 158 235 42 31 $true $ORANGE 1 1 | Out-Null
    Add-Text $s "CNN balanced-accuracy change with regime-SHAP" 63 204 230 51 16 $false $MUTED 1 1 | Out-Null
    Add-Line $s 63 271 292 271 $RULE 1 | Out-Null
    Add-Text $s "Mixed or negative effects for the other architectures" 63 292 225 62 18 $true $INK 1 1 | Out-Null
    Add-Text $s "No contrast survived Holm adjustment" 63 364 225 38 15 $false $MUTED 1 1 | Out-Null
    Add-Text $s "LIME low-fidelity: 71.83%" 63 420 225 23 15 $true $ORANGE 1 1 | Out-Null
    Add-FitImage $s $assets.XAI 351 134 562 330 $true | Out-Null
    Add-Footer $s
    Add-Note $s $notes[6]

    # Slide 8: Results table
    $s = $presentation.Slides.Add(8, 12)
    Add-Header $s "The integrated arm peaked at 53.64% mean balanced accuracy" 8 "FINAL FIVE-MODEL PANEL"
    Add-Panel $s 44 129 872 52 $INK $INK 1 | Out-Null
    $cols = @(60, 148, 492, 663, 808)
    $widths = @(74, 320, 150, 132, 90)
    $heads = @("Rank", "Model", "Window", "Mean BAcc", "Direction Acc")
    for ($i = 0; $i -lt $heads.Count; $i++) {
        Add-Text $s $heads[$i] $cols[$i] 145 $widths[$i] 23 15 $true $WHITE $(if($i -eq 1){1}else{2}) 3 | Out-Null
    }
    $rows = @(
        @("1", "LSTM-CNN-Attention", "20", "53.64 +/- 2.42", "53.43 +/- 2.96"),
        @("2", "LSTM-CNN", "20", "52.81 +/- 2.29", "52.11 +/- 2.58"),
        @("3", "LSTM-Attention", "10", "52.62 +/- 1.46", "52.40 +/- 0.80"),
        @("4", "LSTM", "5", "52.01 +/- 1.64", "51.98 +/- 0.43"),
        @("5", "CNN", "20", "51.49 +/- 1.83", "51.76 +/- 1.75")
    )
    $yy = 186
    for ($r = 0; $r -lt $rows.Count; $r++) {
        $fill = $(if ($r -eq 0) { $PALE_ORANGE } elseif ($r % 2 -eq 0) { $LIGHT } else { $WHITE })
        Add-Panel $s 44 $yy 872 50 $fill $(if ($r -eq 0) { $ORANGE } else { $RULE }) 1 | Out-Null
        for ($i = 0; $i -lt 5; $i++) {
            $bold = ($r -eq 0 -or $i -eq 1)
            Add-Text $s $rows[$r][$i] $cols[$i] ($yy + 13) $widths[$i] 23 16 $bold $INK $(if($i -eq 1){1}else{2}) 3 | Out-Null
        }
        $yy += 55
    }
    Add-Panel $s 44 474 872 30 $LIGHT $RULE 5 | Out-Null
    Add-Text $s "Common evaluation: four held-out years x five seeds. Differences are modest and architecture-dependent." 60 481 840 18 14 $true $MUTED 2 3 | Out-Null
    Add-Footer $s
    Add-Note $s $notes[7]

    # Slide 9: Prediction behavior
    $s = $presentation.Slides.Add(9, 12)
    Add-Header $s "Level tracking and directional accuracy are not the same objective" 9 "MODEL BEHAVIOR"
    Add-FitImage $s $assets.Timeline 54 124 852 333 $true | Out-Null
    Add-Panel $s 54 470 255 31 $LIGHT $RULE 5 | Out-Null
    Add-Text $s "LSTM: closer level tracking" 65 477 235 18 14 $true $INK 2 3 | Out-Null
    Add-Panel $s 353 470 255 31 $PALE_GREEN $RULE 5 | Out-Null
    Add-Text $s "CNN: smoother trend tracking" 364 477 235 18 14 $true $INK 2 3 | Out-Null
    Add-Panel $s 652 470 254 31 $PALE_BLUE $RULE 5 | Out-Null
    Add-Text $s "Hybrids: larger level spread" 663 477 234 18 14 $true $INK 2 3 | Out-Null
    Add-Footer $s "Actual versus predicted SET50 levels in the 2025 out-of-sample window"
    Add-Note $s $notes[8]

    # Slide 10: Close
    $s = $presentation.Slides.Add(10, 12)
    Add-Header $s "The framework exposes fragile gains before deployment" 10 "ENGINEERING OUTCOME"
    Add-FitImage $s $assets.Heatmap 43 130 500 346 $true | Out-Null
    Add-Text $s "What the project delivers" 580 133 330 28 22 $true $INK 1 1 | Out-Null
    Add-BulletLine $s "1" "Evidence that worked" "Leader intrinsic sentiment; CNN regime-SHAP" 580 184 325 | Out-Null
    Add-BulletLine $s "2" "Fragility made visible" "VMD, predicted news and transfer did not generalize" 580 258 325 | Out-Null
    Add-BulletLine $s "3" "Reusable engineering assets" "PIT data contract, paired ablations, robustness tests and runtime logs" 580 332 325 | Out-Null
    Add-Panel $s 580 417 325 59 $PALE_ORANGE $ORANGE 5 | Out-Null
    Add-Text $s "Trust improvements that survive the audit - not merely the configuration that wins once." 598 429 290 34 16 $true $INK 2 3 | Out-Null
    Add-Footer $s
    Add-Note $s $notes[9]

    $presentation.SaveAs($pptxPath, 24)
    $presentation.SaveAs($pdfPath, 32)
    foreach ($slide in $presentation.Slides) {
        $png = Join-Path $renderDir ("slide_{0:D2}.png" -f $slide.SlideIndex)
        if (Test-Path -LiteralPath $png) { Remove-Item -LiteralPath $png -Force }
        $slide.Export($png, "PNG", 1600, 900)
    }
} finally {
    $presentation.Close()
    $powerPoint.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output $pptxPath
Write-Output $pdfPath
Write-Output $renderDir
