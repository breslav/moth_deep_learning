%Evaluate existing deep neural network on test data

%MAKE SURE caffe folder is on your path!

%caffe configuration
caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

%specify path to model and weights
model_dir = 'moth_example/demo/';
net_model = [model_dir 'vgg_regression_deploy.prototxt'];
net_weights = [model_dir '_iter_10000.caffemodel'];
phase = 'test'; % run with phase test

% initialize a network
net = caffe.Net(net_model, net_weights, phase);

%read in test data (in this case it is all stored in an hdf5 file)
d = hdf5read([model_dir 'moth_test_1.hdf5'],'data');

%test images are random 400 x 600 crops from original image which are then resized to 224 x 224
%the network therefore predicts locations in the 224 x 224 image space, which then needs to be transformed
%back to the original image space


%specify/load transformation parameters
load([model_dir 'test_label_shifts_new_split.mat']);

label_scale = repmat([224/600 224/400],1,4); 
label_scale_inverted = 1./label_scale;


final_detections = zeros(400,8); %storage for landmark detections [hx,hy, rwtx,rwty, atx, aty, lwtx, lwty]

%for all 400 test images
for i=1:1:400
    
output_regression = net.forward({d(:,:,:,i)}); %last dimension of d specifies test image, which is propagated forward through the network

im = d(:,:,:,i);
im_b = im(:,:,1); %take the first channel as they are all the same for grayscale

figure(1); clf; imagesc(im_b); colormap gray; hold on;
output_landmarks = output_regression{1};

final_detection = (output_landmarks').*label_scale_inverted + label_shifts(i,:);
final_detections(i,:) = final_detection;

%plot in the 224 x 224 space
plot(output_landmarks(1),output_landmarks(2),'or','MarkerFaceColor','r'); hold on;
plot(output_landmarks(3),output_landmarks(4),'om','MarkerFaceColor','m'); hold on;
plot(output_landmarks(5),output_landmarks(6),'og','MarkerFaceColor','g'); hold on;
plot(output_landmarks(7),output_landmarks(8),'ob','MarkerFaceColor','b'); hold on;
% print(gcf,'-dpng',['vgg_regression_',num2str(i),'.png']); 

end

%save final detections
save([model_dir 'final_detections.mat'],'final_detections');