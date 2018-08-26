# ENVIRON_URL=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip        # for Linux
# ENVIRON_URL=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip  # for Linux with no (virtual) screen, e.g. AWS
ENVIRON_URL=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip          # for Mac OSX
ENVIRON_FILE=$(notdir ${ENVIRON_URL})
ENVIRON_DIR=$(ENVIRON_FILE:%.zip=%)
EXAMPLE_URL=https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/p1_navigation/Navigation.ipynb
EXAMPLE_FILE=Navigation.ipynb

$(info $$ENVIRON_URL is ${ENVIRON_URL})
$(info $$ENVIRON_FILE is ${ENVIRON_FILE})
$(info $$ENRIFON_DIR is ${ENVIRON_DIR})

.PHONY: get-environment
get-environment:
	wget -O ${ENVIRON_FILE} ${ENVIRON_URL}
	unzip ${ENVIRON_FILE}

.PHONY: get-example
get-example:
	wget -O ${EXAMPLE_FILE} ${EXAMPLE_URL}

.PHONY: clean
clean:
	rm -f ${ENVIRON_FILE} ${EXAMPLE_FILE}
	rm -rf ${ENVIRON_DIR}
