Environment
```bash
pip install pydantic av numpydantic open3d natsort
```

pull the entire repo MHMP-visualizer into a subfolder named mhmp_vis/
```bash
# clean
git rm -r mhmp_vis
git commit -m "Remove previous subtree folders"
rm -rf .git/modules/mhmp_vis
git remote remove mhmp_vis 2>/dev/null
# git subtree
git remote add mhmp-vis https://github.com/zhoubohan0/MHMP-visualizer.git
git fetch mhmp-vis
git subtree add --prefix=mhmp_vis mhmp-vis master --squash
git subtree pull --prefix=mhmp_vis mhmp-vis master --squash
# git submodule
git submodule add https://github.com/zhoubohan0/MHMP-visualizer.git mhmp_vis
git commit -m "Add MHMP-visualizer as submodule under mhmp_vis"
git submodule update --init --recursive  # for first-time or after clone
cd mhmp_vis # Edit and Push to Submodule (2-way sync) by add commit and push
```

Prepare Dataset
```bash
python -m mhmp_vis.dataset_loader.dataset_fpha
```

Calculate Statistics
```bash
bash bash/statistics.sh -dataset_list taco
```

Training
```bash
# MHMP
bash bash/train.sh -dataset_list taco -depthanything_path "" -incorporate_depth 0 -exp_name RGB 
# MHMP+

```
