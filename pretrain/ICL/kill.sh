ps -ef | grep 'sim' | grep -v grep | awk '{print "kill -9 "$2}' | sh
