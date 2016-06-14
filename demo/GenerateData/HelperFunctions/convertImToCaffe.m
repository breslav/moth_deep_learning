%need a function to turn rows x column single channel grayscale image to caffe/vgg compatible format

function [b,g,r] = convertImToCaffe(im_8bit,vgg_16_bgr)


    b_channel = im_8bit - vgg_16_bgr(1); 
    g_channel = im_8bit - vgg_16_bgr(2);
    r_channel = im_8bit - vgg_16_bgr(3);

    b = imresize(b_channel,[224 224],'bilinear'); 
    g = imresize(g_channel,[224 224],'bilinear');
    r = imresize(r_channel,[224 224],'bilinear');

        
end