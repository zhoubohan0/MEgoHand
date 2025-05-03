killname=${1:-"gr1"}
ps aux | grep python | grep "$killname" | awk '{print $2}' | xargs kill