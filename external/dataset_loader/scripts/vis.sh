# foundationpose for interactive, bvdex for offline

# export QT_DEBUG_PLUGINS=1
# export QT_PLUGIN_PATH=/home/zbh/anaconda3/envs/foundationpose/lib/python3.8/site-packages/cv2/qt/plugins
# export QT_QPA_PLATFORM=offscreen
# export DISPLAY=:0


vis_way="interactive"
dataset="oakink2"

while getopts "d:w:" opt; do
  case "$opt" in
    d) dataset=$OPTARG ;;
    w) vis_way=$OPTARG ;;
  esac
done

echo "dataset: $dataset, vis_way: $vis_way"

python -m dataset_visualizer.general_${vis_way}_visualizer -d $dataset
