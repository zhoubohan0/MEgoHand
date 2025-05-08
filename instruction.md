Run locally
```bash
bash bash/train.sh -dataset_path /media/ps/zhan/DATASET/ -depth_model_path "" -multimodal "tv" -exp_name ZY_RGB_fpha -lora_rank 0 -tune_llm 0 -tune_visual 0 -device 0 -dataset_list fpha -debug 1 -batch_size 1 
```

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

VSCode 
    for entry in os.scandir(join(prefix, "etc", "conda", "deactivate.d"))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "<frozen posixpath>", line 76, in join
TypeError: expected str, bytes or os.PathLike object, not NoneType
```bash
export CONDA_SHLVL=0
```

push:
```bash
commit_message="fix tiny" 
cd mhmp_vis 
git add . ;  git commit -m $commit_message ; git push 
cd - 
git add . ;  git commit -m $commit_message ; git push
```
pull:
```bash
cd mhmp_vis
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
git pull
cd - 
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
git pull
```
