pretrain_ckpt=""

while getopts "p:" opt; do
  case "$opt" in
    p) pretrain_ckpt=$OPTARG ;;
  esac
done

echo "pretrain_ckpt: $pretrain_ckpt"
if [ -z "$pretrain_ckpt" ]; then
  echo "Usage: $0 -p <pretrain_ckpt>"
  exit 1
fi

python -m pdb -m scripts.test --pretrain_ckpt $pretrain_ckpt 