cd /path/to/zips

files=(
  cifar10_preactresnet18_badnet_0_1.zip
  cifar10_preactresnet18_blind_0_1.zip
  cifar10_preactresnet18_bpp_0_1.zip
  cifar10_preactresnet18_inputaware_0_1.zip
  cifar10_preactresnet18_lc_0_1.zip
  cifar10_preactresnet18_ssba_0_1.zip
  cifar10_preactresnet18_trojannn_0_1.zip
  cifar10_preactresnet18_wanet_0_1.zip
)

for f in "${files[@]}"; do
  echo "==> Unzipping: $f"
  [ -f "$f" ] || { echo "!! not found: $f"; exit 1; }
  out="${f%.zip}"
  mkdir -p "$out"
  unzip -o "$f" -d "$out"
done
