pip install pydantic av numpydantic open3d natsort
pull the entire repo MHMP-visualizer into a subfolder named mhmp-vis/
```bash
git rm -r mhmp-vis
git commit -m "Remove previous subtree folders"
git push
git remote add mhmp-vis https://github.com/zhoubohan0/MHMP-visualizer.git
git fetch mhmp-vis
git subtree add --prefix=mhmp_vis mhmp-vis master --squash
# mhmp-vis/dataset_loader and mhmp-vis/utils are used
```