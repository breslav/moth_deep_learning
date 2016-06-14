%compute features for data......


load('moth_example/SavedAnnotations/cam1_annotations_pts.mat');
cam1_state_pts = cell2mat(cam_state_save);

caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

model_dir = '/research/bats3/Breslav/deeplearning/moth_example/';
net_model = [model_dir 'vgg_regression_deploy.prototxt'];
net_weights = [model_dir 'vgg_16_regression_aug_32_2/ts/_iter_10000.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

d = hdf5read('/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/moth_test_1.hdf5','data');
l = hdf5read('/research/bats3/Breslav/deeplearning/moth_example/hdf5_data/moth_test_1.hdf5','label');

load('moth_example/test_label_shifts_new_split.mat');
final_detections = zeros(400,11); 

label_scale = repmat([224/600 224/400],1,4); 
label_scale_inverted = 1./label_scale;

testing_idx = 401:1:800;

for i=1:1:400
    
output_regression = net.forward({d(:,:,:,i)});

im = d(:,:,:,i);
im_b = im(:,:,1);


output_landmarks = output_regression{1};

final_detections(i,1) = cam1_state_pts(testing_idx(i),1);
final_detection = (output_landmarks').*label_scale_inverted + label_shifts(i,:);

final_detections(i,4:5) = final_detection(1:2); %head
final_detections(i,6:7) = final_detection(5:6); %abdomen tip
final_detections(i,8:9) = final_detection(7:8); %left wing tip
final_detections(i,10:11) = final_detection(3:4); %right wing tip

%plot 224 x 224 result
% figure(1); clf; imagesc(im_b); colormap gray; hold on;
% plot(output_landmarks(1),output_landmarks(2),'or','MarkerFaceColor','r'); hold on;
% plot(output_landmarks(3),output_landmarks(4),'om','MarkerFaceColor','m'); hold on;
% plot(output_landmarks(5),output_landmarks(6),'og','MarkerFaceColor','g'); hold on;
% plot(output_landmarks(7),output_landmarks(8),'ob','MarkerFaceColor','b'); hold on;
% print(gcf,'-dpng',['vgg_regression_',num2str(i),'.png']);

%plot original result
im_orig = uint16(imread(['Cam1_Images/',num2str(testing_idx(i)),'.png']));
figure(1); clf; imagesc(im_orig); colormap gray; hold on;
plot(final_detection(1),final_detection(2),'or','MarkerFaceColor','r'); hold on;
plot(final_detection(3),final_detection(4),'om','MarkerFaceColor','m'); hold on;
plot(final_detection(5),final_detection(6),'og','MarkerFaceColor','g'); hold on;
plot(final_detection(7),final_detection(8),'ob','MarkerFaceColor','b'); hold on;
print(gcf,'-dpng',['vgg_regression_',num2str(i),'.png']);

end

save('moth_example/vgg_regression_aug_ts_32_2_iter_10k_new_split.mat','final_detections');