$branches = @(
  "apply-burst-prefetch",
  "apply-caller-prefetch",
  "apply-dispatch-both",
  "apply-dispatch-prefetch",
  "apply-dispatch-template",
  "apply-fuse-copy",
  "apply-fuse-copy-psqt",
  "apply-fuse-copy-sub",
  "apply-fuse-dispatch",
  "apply-fuse-prefetch",
  "apply-interleave",
  "apply-memcpy-copy",
  "apply-prefetch-first-t0",
  "apply-prefetch-initial",
  "apply-prefetch-removed",
  "apply-prefetch-rolling",
  "apply-psqt-prefetch",
  "apply-stream-store",
  "apply-swap-loops",
  "apply-swapped-loops",
  "apply-template-prefetch",
  "apply-template-prefetch-t2",
  "apply-template-unroll",
  "apply-unified-initpf",
  "apply-write-prefetch"
)

foreach ($branch in $branches) {
    Write-Host "Pushing branch: $branch"
    git push origin $branch --force
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to push $branch" -ForegroundColor Red
    } else {
        Write-Host "Successfully pushed $branch" -ForegroundColor Green
    }
}
