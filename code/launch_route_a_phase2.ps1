$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
. .\env_cache.ps1

$tag = "route_a_fsna_rf_lolv1"
$parent = Split-Path $Root -Parent
$weights = Join-Path $parent "third_party\Retinexformer\pretrained_weights\LOL_v1.pth"
$logDir = Join-Path $parent "results"
$ckptDir = Join-Path $parent "checkpoints\$tag"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "route_a_phase2_train.log"
$err = Join-Path $logDir "route_a_phase2_train.log.err"

if (-not (Test-Path $weights)) { throw "Missing pretrained weights: $weights" }

# Archive previous collapsed / plateaued run artifacts.
if (Test-Path $ckptDir) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $bak = Join-Path $parent ("checkpoints\${tag}_bak_$stamp")
  Move-Item $ckptDir $bak -Force
  New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
}
$rk = Join-Path $logDir "route_a_kill.json"
if (Test-Path $rk) { Move-Item $rk ($rk + ".bak") -Force }

$env:PYTHONUNBUFFERED = "1"
$py = (Get-Command python).Source
# Freeze RF backbone (keep 25.16). Train FRB/NAF/heads only with direct fusion loss.
# No --official_hold_lr. No mixup (cleaner residual learning on frozen BB).
$arg = @(
  "-u train_route_a.py",
  "--model fsna_rf",
  "--dataset lolv1",
  "--tag $tag",
  "--pretrained `"$weights`"",
  "--freeze_backbone",
  "--official_augs",
  "--no_mixup",
  "--iters 80000",
  "--eval_every 1000",
  "--batch_size 8",
  "--patch_size 128",
  "--lr 1.5e-4",
  "--backbone_lr_mult 0.0",
  "--w_base 0.0",
  "--w_fused 0.4",
  "--w_fre 0.2",
  "--w_ssim 0.25",
  "--n_feat 40",
  "--num_workers 4",
  "--gate_20k 25.10",
  "--gate_50k 25.25",
  "--target_psnr 25.45"
) -join " "

Write-Host "Launching Route A Phase-2 FSNA-on-RF (frozen BB + anti-collapse fuse): $tag"
Write-Host "Pretrained: $weights"
Write-Host "freeze+detach BB; fused+delta path; gate floor=0.15; lr=1.5e-4; gates 25.10/25.25 target 25.45"
if (Test-Path $log) { Move-Item $log ($log + ".prev") -Force }
if (Test-Path $err) { Move-Item $err ($err + ".prev") -Force }
$p = Start-Process -FilePath $py -ArgumentList $arg -WorkingDirectory $Root `
  -RedirectStandardOutput $log -RedirectStandardError $err `
  -PassThru -WindowStyle Hidden
Write-Host "Started PID=$($p.Id) -> $log"
Wait-Process -Id $p.Id
Write-Host "ExitCode=$($p.ExitCode)"
