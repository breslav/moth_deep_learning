#!/usr/bin/env sh

GLOG_logtostderr=1 ../caffe/build/tools/caffe train -gpu 0 -solver ./vgg_regression_solver.prototxt -weights /research/bats3/Breslav/deeplearning/moth_example/VGG_ILSVRC_16_layers.caffemodel 2>&1 | tee /research/bats3/Breslav/deeplearning/moth_example/experiments/solver_test_log.txt
echo "Done."
