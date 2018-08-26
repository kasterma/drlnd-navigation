ENVIRON_URL=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip

.PHONY: get-environment
get-environment:
	wget -O Banana.app.zip ${ENVIRON_URL}
	unzip Banana.app.zip

.PHONY: clean
clean:
	rm -f Banana.app.zip
	rm -rf Banana.app/
